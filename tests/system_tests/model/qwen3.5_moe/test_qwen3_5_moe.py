import os
import time

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.quantization.cache import clear_weight_cache
from fsdp_turbo.fsdp_turbo_config import (
    FSDPTurboConfig,
    FSDPPlanConfig,
    EPPlanConfig,
    CPPlanConfig,
    DataConfig,
    ModelConfig,
    TPPlanConfig,
    DistributedConfig,
    MemoryConfig,
)
from fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5_moe_cp import qwen3_5_moe_model_forward
from fsdp_turbo.training.clip_grads import compute_grad_norm
from fsdp_turbo.utils.device import set_accelerator_compatible
from fsdp_turbo.utils.log import set_log_level
from fsdp_turbo.utils.model import init_submodule_weights, set_num_hidden_layers
from fsdp_turbo.utils.random import set_seed


# ============================================================
# Configuration
# ============================================================
MODEL_PATH = os.environ.get('MODEL_PATH', 'Qwen/Qwen3.5-35B-A3B')
DATASET_PATH = os.environ.get('DATASET_PATH', '/home/dataset/coco/llava_instruct_150k_fsdp_turbo_100.json')
DATASET_SPLIT = os.environ.get('DATASET_SPLIT', 'train')
SEED = 42
LOG_LEVEL = 'INFO'

# Training hyperparameters
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '1'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', '512'))
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', '0'))
LR = 1e-4
NUM_STEPS = int(os.environ.get('NUM_STEPS', '10'))
# Reduce language-model layers and randomly initialize (CI mode, no weights).
NUM_HIDDEN_LAYERS = int(os.environ.get('NUM_HIDDEN_LAYERS', '4'))

# Parallel configuration (examples/qwen3_5_moe/config.yaml uses FSP=8/TP=1/ULYSSES=1/EP=8/EFP=1;
# st enables Ulysses CP, rebalanced to FSP=4/TP=1/ULYSSES=2/EP=8/EFP=1 on 8 NPUs).
TENSOR_PARALLEL_SIZE = int(os.environ.get('TENSOR_PARALLEL_SIZE', '1'))
FULLY_SHARD_PARALLEL_SIZE = int(os.environ.get('FULLY_SHARD_PARALLEL_SIZE', '4'))
ULYSSES_PARALLEL_SIZE = int(os.environ.get('ULYSSES_PARALLEL_SIZE', '2'))
EXPERT_PARALLEL_SIZE = int(os.environ.get('EXPERT_PARALLEL_SIZE', '8'))
EXPERT_FULLY_SHARD_PARALLEL_SIZE = int(os.environ.get('EXPERT_FULLY_SHARD_PARALLEL_SIZE', '1'))

# Profiler configuration
ENABLE_PROFILER = os.environ.get('ENABLE_PROFILER', '0').lower() in ('1', 'true', 'yes', 'on')
PROFILE_MEMORY = os.environ.get('PROFILE_MEMORY', '0').lower() in ('1', 'true', 'yes', 'on')
PROFILE_DIR = os.environ.get('PROFILE_DIR', './profile')

# Patch targets for the Qwen3.5-MoE multimodal model + GatedDeltaNet layers.
QWEN3_5_MOE_EAGER_ATTENTION_FORWARD = 'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.eager_attention_forward'
FOR_CAUSAL_LM_LOSS = 'transformers.loss.loss_utils.ForCausalLMLoss'
QWEN3_5_MOE_GATED_DELTA_NET_FORWARD = (
    'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet.forward'
)
QWEN3_5_MOE_MODEL_FORWARD = 'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeModel.forward'


def initialization():
    """Initialize distributed environment, accelerator compatibility, seed, and logging."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(local_rank)

    set_accelerator_compatible()
    set_seed(SEED)
    set_log_level(LOG_LEVEL)


def get_model_tokenizer(config):
    """Build a reduced-layer, randomly-initialized Qwen3.5-MoE model (CI mode).

    Mirrors examples/qwen3_5_moe/train.py ``build_model``: the architecture is
    taken from ``config.model.model_name_or_path`` (config.json +
    tokenizer/processor only, no weights), the language decoder is reduced via
    ``config.model.num_hidden_layers``, and the model is built on the ``meta``
    device. trainer_with_config then materializes it via ``to_empty`` and
    randomly initializes it via ``_init_weights`` BEFORE FSDPTurbo wrapping.
    The processor is loaded for multimodal data (images + conversations).
    """
    model_config = AutoConfig.from_pretrained(config.model.model_name_or_path)
    model_config._attn_implementation = "eager"
    set_num_hidden_layers(model_config, config.model.num_hidden_layers)

    with torch.device('meta'):
        model = AutoModelForImageTextToText.from_config(model_config, torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(config.model.tokenizer_name_or_path)
    return model, processor


def get_data_config():
    """Mirror examples/qwen3_5_moe/config.yaml data section (multimodal)."""
    return DataConfig(
        dataset_path=DATASET_PATH,
        is_multimodal=True,
        text_column='text',
        image_column='images',
        conversation_column='messages',
        split=DATASET_SPLIT,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_LENGTH,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
    )


def get_dataloader(processor):
    """Build the multimodal dataloader (conversations + images via processor)."""
    from fsdp_turbo.training.dataloader import build_dataloader

    return build_dataloader(get_data_config(), tokenizer=processor.tokenizer, processor=processor)


# ============================================================
# Complete configurations
# ============================================================
def get_complete_custom_config():
    """
    Get complete FSDP configuration for Qwen3.5-MoE with all features enabled.

    Features:
    - FSDP (custom implementation) over visual + language + embeddings + lm_head
    - Expert Parallel + Expert Fully Sharded Data Parallel over MoE experts
    - Tensor Parallel
    - Ulysses Context Parallel: full_attention, causal_lm_loss, and GatedDeltaNet
      (GatedDeltaNet common-class forward) patches under cp_plan
    - Module patch: Qwen3_5MoeModel.forward
    - Recompute for memory optimization
    - Mixed Precision (BF16 param/output, FP32 reduce)
    """
    return FSDPTurboConfig(
        model=ModelConfig(
            model_name_or_path=MODEL_PATH,
            tokenizer_name_or_path=MODEL_PATH,
            torch_dtype=torch.bfloat16,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
        ),
        distributed=DistributedConfig(
            fully_shard_parallel_size=FULLY_SHARD_PARALLEL_SIZE,
            fsdp_plan=FSDPPlanConfig(
                ignored_modules=[],
                apply_modules={
                    'model.visual.blocks.{*}': {},
                    'model.visual.patch_embed': {},
                    'model.visual.pos_embed': {},
                    'model.language_model.layers.{*}': {},
                    'model.language_model.embed_tokens': {},
                    'lm_head': {},
                },
                param_dtype='bf16',
                reduce_dtype='fp32',
                output_dtype='bf16',
                hook_modules=['model.language_model.layers.{*}'],
                fsdp_implementation='native',
                num_to_forward_prefetch=1,
                num_to_backward_prefetch=1,
            ),
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            tp_plan=TPPlanConfig(
                colwise_parallel=['*.q_proj', '*.k_proj', '*.v_proj'],
                rowwise_parallel=['*.o_proj'],
            ),
            ulysses_parallel_size=ULYSSES_PARALLEL_SIZE,
            cp_plan=CPPlanConfig(
                ulysses_function_patches=[
                    {'target_functions': [QWEN3_5_MOE_EAGER_ATTENTION_FORWARD], 'type': 'full_attention'},
                    {'target_functions': [QWEN3_5_MOE_GATED_DELTA_NET_FORWARD], 'type': 'gated_delta_net'},
                ],
                loss_function_patches=[
                    {'target_functions': [FOR_CAUSAL_LM_LOSS], 'type': 'causal_lm_loss'},
                ],
            ),
            expert_parallel_size=EXPERT_PARALLEL_SIZE,
            expert_fully_shard_parallel_size=EXPERT_FULLY_SHARD_PARALLEL_SIZE,
            ep_plan=EPPlanConfig(
                apply_modules=['model.language_model.layers.{*}.mlp.experts'],
                dispatcher='all2all',
                use_fused_ops=True,
                apply_efsdp_modules=['model.language_model.layers.{*}.mlp.experts'],
            ),
        ),
        memory=MemoryConfig(
            recompute=True,
            recompute_plan=['model.language_model.layers.{*}', 'model.visual.blocks.{*}'],
        ),
        module_patches=[
            {'target': QWEN3_5_MOE_MODEL_FORWARD, 'replacement': qwen3_5_moe_model_forward},
        ],
    )


# ============================================================
# Training loop
# ============================================================
def train_step(model, batch, optimizer):
    """Execute a single training step: forward, backward, optimizer step, zero_grad."""
    batch['use_cache'] = False
    batch = {k: (v.npu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    outputs = model.forward(**batch)
    loss = outputs.loss
    loss.backward()
    grad_norm = compute_grad_norm(list(model.parameters()))

    optimizer.step()
    clear_weight_cache()
    model.zero_grad()

    return loss, grad_norm


def trainer_with_config(config, config_name, description, num_steps=NUM_STEPS, enable_profiler=False):
    """
    Train with a specific configuration.

    Args:
        config: FSDPTurboConfig instance.
        config_name: Name of the configuration.
        description: Description of the configuration.
        num_steps: Number of training steps to run.
        enable_profiler: Whether to enable NPU profiler.
    """
    initialization()

    if torch.accelerator.current_device_index() == 0:
        print(f'\n{"=" * 60}')
        print(f'Test: {config_name}')
        print(f'Description: {description}')
        print(f'{"=" * 60}\n')

    model, processor = get_model_tokenizer(config)
    model.to_empty(device='npu')
    init_submodule_weights(model.model.language_model, name="language_model")
    init_submodule_weights(model.model.visual, name="visual")
    model.tie_weights()
    model = FSDPTurbo(config, model)
    dataloader = get_dataloader(processor)

    if torch.accelerator.current_device_index() == 0:
        print(model)
    model.model.train(True)

    optimizer = AdamW(model.model.parameters(), lr=LR)

    if enable_profiler:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AicMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
        )
        prof_ctx = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU],
            record_shapes=False,
            profile_memory=PROFILE_MEMORY,
            with_stack=False,
            experimental_config=experimental_config,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=5000),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(PROFILE_DIR),
        )
    else:
        from contextlib import nullcontext

        prof_ctx = nullcontext()

    with prof_ctx as prof:
        for i, batch in enumerate(dataloader):
            if i >= num_steps:
                break

            torch.distributed.barrier()
            torch.npu.reset_peak_memory_stats()

            step_start = time.perf_counter()
            loss, grad_norm = train_step(model, batch, optimizer)
            torch.npu.synchronize()
            step_time = time.perf_counter() - step_start

            if enable_profiler:
                prof.step()

            torch.distributed.barrier()
            if torch.accelerator.current_device_index() == 0:
                print(
                    f'[Rank {torch.accelerator.current_device_index()}] step {i} '
                    f'step time: {step_time:.4f}s '
                    f'loss: {loss.item():.4f} '
                    f'grad norm: {grad_norm:.4f} '
                    f'memory: {torch.npu.memory_allocated() / 1024**3:.4f} GB '
                    f'max memory: {torch.npu.max_memory_allocated() / 1024**3:.4f} GB',
                )

    if torch.accelerator.current_device_index() == 0:
        print(f'\n Test completed: {config_name}\n')


# ============================================================
# Unified test entry point
# ============================================================
def test_qwen3_5_moe():
    """
    Test Qwen3.5-MoE with complete FSDP configurations.

    This test will run the following configuration.:
    FSDP + EP + TP + Ulysses CP (GDN) + Module Patches + Recompute + Mixed Precision + Hook Module
    """

    trainer_with_config(
        config=get_complete_custom_config(),
        config_name='complete_custom',
        description=(
            'FSDP + EP + TP + Ulysses CP (GDN) + Recompute + Mixed Precision + Hook Module + Prefetch (Custom)'
            f' + Ulysses={ULYSSES_PARALLEL_SIZE}, EP={EXPERT_PARALLEL_SIZE}, EFP={EXPERT_FULLY_SHARD_PARALLEL_SIZE}'
        ),
        enable_profiler=ENABLE_PROFILER,
    )

    if torch.accelerator.current_device_index() == 0:
        print(f'\n{"#" * 60}')
        print('All tests completed successfully!')
        print(f'{"#" * 60}\n')


if __name__ == '__main__':
    test_qwen3_5_moe()

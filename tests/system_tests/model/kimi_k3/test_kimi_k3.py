import os
import time

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import (
    DataConfig,
    DistributedConfig,
    EPPlanConfig,
    FSDPPlanConfig,
    FSDPTurboConfig,
    MemoryConfig,
    ModelConfig,
    TPPlanConfig,
)
from fsdp_turbo.quantization.cache import clear_weight_cache
from fsdp_turbo.training.clip_grads import compute_grad_norm
from fsdp_turbo.utils.device import set_accelerator_compatible
from fsdp_turbo.utils.log import set_log_level
from fsdp_turbo.utils.model import init_submodule_weights, set_num_hidden_layers
from fsdp_turbo.utils.random import set_seed


# ============================================================
# Configuration
# ============================================================
MODEL_PATH = os.environ.get('MODEL_PATH', 'moonshotai/Kimi-K3')
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

# Parallel configuration. Kimi-K3 uses a 8-way EP layout in its example.
TENSOR_PARALLEL_SIZE = int(os.environ.get('TENSOR_PARALLEL_SIZE', '1'))
FULLY_SHARD_PARALLEL_SIZE = int(os.environ.get('FULLY_SHARD_PARALLEL_SIZE', '8'))
EXPERT_PARALLEL_SIZE = int(os.environ.get('EXPERT_PARALLEL_SIZE', '8'))
EXPERT_FULLY_SHARD_PARALLEL_SIZE = int(os.environ.get('EXPERT_FULLY_SHARD_PARALLEL_SIZE', '1'))

# Profiler configuration
ENABLE_PROFILER = os.environ.get('ENABLE_PROFILER', '0').lower() in ('1', 'true', 'yes', 'on')
PROFILE_MEMORY = os.environ.get('PROFILE_MEMORY', '0').lower() in ('1', 'true', 'yes', 'on')
PROFILE_DIR = os.environ.get('PROFILE_DIR', './profile')


def initialization():
    """Initialize distributed environment, accelerator compatibility, seed, and logging."""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.npu.set_device(local_rank)

    set_accelerator_compatible()
    set_seed(SEED)
    set_log_level(LOG_LEVEL)


def _set_attn_implementation(model_config, attn_implementation):
    """Set eager attention on the top-level, text, and vision configurations."""
    for config in (
        model_config,
        getattr(model_config, 'text_config', None),
        getattr(model_config, 'vision_config', None),
    ):
        if config is not None:
            config._attn_implementation = attn_implementation


def get_model_processor(config):
    """Build a reduced-layer, randomly initialized Kimi-K3 model for ST."""
    from fsdp_turbo.models.kimi import register_kimi_k3
    from fsdp_turbo.models.kimi.kimi_k3_processor import KimiK3Processor

    register_kimi_k3()
    model_config = AutoConfig.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=False,
    )
    _set_attn_implementation(model_config, 'eager')
    set_num_hidden_layers(model_config, config.model.num_hidden_layers)

    model_config.text_config.num_experts = 16
    model_config.text_config.num_shared_experts = 2
    with torch.device('meta'):
        model = AutoModel.from_config(
            model_config,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
        )

    processor = KimiK3Processor.from_pretrained(
        config.model.tokenizer_name_or_path,
        trust_remote_code=True,
    )
    return model, processor


def get_data_config():
    """Mirror examples/kimi_k3/config.yaml multimodal data settings."""
    return DataConfig(
        dataset_path=DATASET_PATH,
        is_multimodal=True,
        text_column='text',
        image_column='images',
        media_key='medias',
        conversation_column='messages',
        split=DATASET_SPLIT,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_LENGTH,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
    )


def get_dataloader(processor):
    """Build the Kimi-K3 multimodal dataloader."""
    from fsdp_turbo.training.dataloader import build_dataloader

    return build_dataloader(get_data_config(), tokenizer=processor.tokenizer, processor=processor)


def initialize_materialized_model(model, dtype):
    """Deterministically initialize a Kimi-K3 model after ``to_empty``."""
    # Constructors run on meta in this ST, so constructor-time initializers do
    # not populate the parameters materialized later by ``to_empty``. Re-run
    # local reset hooks first for custom MoE, convolution, router, and vision
    # parameters, then apply each pretrained submodel's canonical initializer.
    for module in model.modules():
        reset_parameters = getattr(module, 'reset_parameters', None)
        if callable(reset_parameters):
            reset_parameters()

    init_submodule_weights(model.language_model, name='language_model')
    init_submodule_weights(model.vision_tower, name='vision_tower')
    init_submodule_weights(model.mm_projector, name='mm_projector')

    # Kimi's RMSNorm modules and KDA state parameters have no reset hook and
    # are not covered by the generic Linear/Embedding initializers.
    for module in model.modules():
        if type(module).__name__ in {'KimiRMSNorm', 'KimiK_3_MoeRMSNormGated'}:
            torch.nn.init.ones_(module.weight)
        elif type(module).__name__ == 'KimiDeltaAttention':
            torch.nn.init.uniform_(module.A_log, 1, 16)
            module.A_log.data.log_()
            torch.nn.init.zeros_(module.dt_bias)
        elif type(module).__name__ == 'KimiMoEGate':
            torch.nn.init.zeros_(module.e_score_correction_bias)

    model.tie_weights()
    model.to(dtype=dtype)


# ============================================================
# Complete configuration
# ============================================================
def get_complete_custom_config():
    """Get the Kimi-K3 FSDP + EP + TP system-test configuration."""
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
                    'vision_tower.encoder.blocks.{*}': {},
                    'mm_projector': {},
                    'language_model.model.embed_tokens': {},
                    'language_model.model.layers.{*}': {},
                    'language_model.lm_head': {},
                },
                param_dtype='bf16',
                reduce_dtype='fp32',
                output_dtype='bf16',
                cast_forward_inputs=True,
                hook_modules=['language_model.model.layers.{*}'],
                fsdp_implementation='native',
                num_to_forward_prefetch=1,
                num_to_backward_prefetch=1,
            ),
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            tp_plan=TPPlanConfig(
                colwise_parallel=['*.q_proj', '*.k_proj', '*.v_proj'],
                rowwise_parallel=['*.o_proj'],
            ),
            expert_parallel_size=EXPERT_PARALLEL_SIZE,
            expert_fully_shard_parallel_size=EXPERT_FULLY_SHARD_PARALLEL_SIZE,
            ep_plan=EPPlanConfig(
                apply_modules=['language_model.model.layers.{*}.block_sparse_moe.experts'],
                dispatcher='custom_ep_forward_eager',
            ),
        ),
        memory=MemoryConfig(
            recompute=True,
            recompute_plan=[
                'language_model.model.layers.{*}',
                'vision_tower.encoder.blocks.{*}',
            ],
        ),
    )


# ============================================================
# Training loop
# ============================================================
def train_step(model, batch, optimizer):
    """Execute a single forward, backward, and optimizer step."""
    batch['use_cache'] = False
    batch = {key: (value.npu() if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}

    outputs = model.forward(**batch)
    loss = outputs.loss
    if loss is None:
        raise RuntimeError('Kimi-K3 did not return a loss; ensure the batch contains labels.')
    if not torch.isfinite(loss).all().item():
        raise FloatingPointError(f'Kimi-K3 produced a non-finite loss: {loss.detach().float().item()}.')

    loss.backward()
    grad_norm = compute_grad_norm(list(model.parameters()))
    if not torch.isfinite(torch.tensor(grad_norm)).item():
        raise FloatingPointError(f'Kimi-K3 produced a non-finite gradient norm: {grad_norm}.')

    optimizer.step()
    clear_weight_cache()
    model.zero_grad()
    return loss, grad_norm


def trainer_with_config(config, config_name, description, num_steps=NUM_STEPS, enable_profiler=False):
    """Train Kimi-K3 for a few steps with the supplied configuration."""
    initialization()

    if torch.accelerator.current_device_index() == 0:
        print(f'\n{"=" * 60}')
        print(f'Test: {config_name}')
        print(f'Description: {description}')
        print(f'{"=" * 60}\n')

    model, processor = get_model_processor(config)
    model.to_empty(device='npu')
    initialize_materialized_model(model, config.model.torch_dtype)

    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    if parameter_dtypes != {config.model.torch_dtype}:
        raise RuntimeError(
            f'Kimi-K3 parameters must have a uniform dtype before FSDP wrapping, but found {parameter_dtypes}.'
        )

    model = FSDPTurbo(config, model)
    dataloader = get_dataloader(processor)

    if torch.accelerator.current_device_index() == 0:
        print(model)
    model.model.train(True)
    optimizer = AdamW(model.model.parameters(), lr=LR)

    if enable_profiler:
        if torch_npu is None:
            raise RuntimeError('ENABLE_PROFILER requires torch_npu.')
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
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
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            torch.distributed.barrier()
            torch.npu.reset_peak_memory_stats()
            print(f'[Rank {torch.accelerator.current_device_index()}] start step {step}')

            step_start = time.perf_counter()
            loss, grad_norm = train_step(model, batch, optimizer)
            torch.npu.synchronize()
            step_time = time.perf_counter() - step_start

            if enable_profiler:
                prof.step()

            torch.distributed.barrier()
            if torch.accelerator.current_device_index() == 0:
                print(
                    f'[Rank {torch.accelerator.current_device_index()}] step {step} '
                    f'step time: {step_time:.4f}s '
                    f'loss: {loss.item():.4f} '
                    f'grad norm: {grad_norm:.4f} '
                    f'memory: {torch.npu.memory_allocated() / 1024**3:.4f} GB '
                    f'max memory: {torch.npu.max_memory_allocated() / 1024**3:.4f} GB'
                )

    if torch.accelerator.current_device_index() == 0:
        print(f'\nTest completed: {config_name}\n')


def test_kimi_k3():
    """Run the Kimi-K3 complete system-test configuration."""
    trainer_with_config(
        config=get_complete_custom_config(),
        config_name='complete_custom',
        description=(
            'FSDP + EP + TP + Recompute + Mixed Precision + Hook Module + Prefetch (Native)'
            f' + EP={EXPERT_PARALLEL_SIZE}, EFP={EXPERT_FULLY_SHARD_PARALLEL_SIZE}'
        ),
        enable_profiler=ENABLE_PROFILER,
    )

    if torch.accelerator.current_device_index() == 0:
        print(f'\n{"#" * 60}')
        print('All tests completed successfully!')
        print(f'{"#" * 60}\n')


if __name__ == '__main__':
    test_kimi_k3()

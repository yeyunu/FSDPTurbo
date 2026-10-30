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
from fsdp_turbo.fsdp_turbo_config import (
    DataConfig,
    DistributedConfig,
    EPPlanConfig,
    FSDPPlanConfig,
    FSDPTurboConfig,
    MemoryConfig,
    ModelConfig,
)
from fsdp_turbo.quantization.cache import clear_weight_cache
from fsdp_turbo.training.clip_grads import compute_grad_norm
from fsdp_turbo.utils.device import set_accelerator_compatible
from fsdp_turbo.utils.log import set_log_level
from fsdp_turbo.utils.model import init_submodule_weights, set_attn_implementation, set_num_hidden_layers
from fsdp_turbo.utils.random import set_seed


# ============================================================
# Configuration
# ============================================================
MODEL_PATH = os.environ.get('MODEL_PATH', '/home/dataset/GLM-5.3-Flash')
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

# DSA/DSA-indexer implementation knobs. The defaults (dense attention + eager
# indexer) form the numerically safest baseline; set DSA_IMPL=sfa together with
# INDEXER_IMPL=triton to exercise the production sparse path on NPU.
DSA_IMPL = os.environ.get('DSA_IMPL', 'dense')
INDEXER_IMPL = os.environ.get('INDEXER_IMPL', 'eager')
USE_FUSED_LIGHTNING_INDEXER = os.environ.get('USE_FUSED_LIGHTNING_INDEXER', '0').lower() in ('1', 'true', 'yes', 'on')

# Parallel configuration. The GLM-5.3-Flash example uses a 16-way EP/FSDP
# layout; the ST mirrors it on 8 NPUs.
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


def get_model_processor(config):
    """Build a reduced-layer, randomly initialized GLM-5.3-Flash model for ST."""
    from fsdp_turbo.models.glm5_next import register_glm5_next

    register_glm5_next()
    model_config = AutoConfig.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=False,
    )
    set_attn_implementation(model_config, 'eager')
    set_num_hidden_layers(model_config, config.model.num_hidden_layers)

    text_config = model_config.text_config
    # Upstream checkpoints may name DSA layers "indexed_attention"; normalize to
    # the key used by causal_mask_mapping. Instance-level fix because the
    # Glm5NextTextConfig.__post_init__ normalization does not reliably run when
    # the nested text_config is rebuilt from its checkpoint dict.
    if text_config.layer_types:
        text_config.layer_types = [
            'deepseek_sparse_attention' if lt in ('full_attention', 'indexed_attention') else lt
            for lt in text_config.layer_types
        ]
    # Reduce the expert count so the randomly initialized ST fits 8 NPUs.
    text_config.n_routed_experts = 16
    text_config.n_shared_experts = 1
    text_config.num_experts_per_tok = min(text_config.num_experts_per_tok, 8)
    # DSA implementation knobs (default: dense + eager, the ST baseline).
    text_config.dsa_implementation = DSA_IMPL
    text_config.indexer_implementation = INDEXER_IMPL
    text_config.use_fused_lightning_indexer = USE_FUSED_LIGHTNING_INDEXER

    with torch.device('meta'):
        model = AutoModelForImageTextToText.from_config(
            model_config,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
        )

    processor = AutoProcessor.from_pretrained(config.model.tokenizer_name_or_path, trust_remote_code=True)
    return model, processor


def get_data_config():
    """Mirror examples/glm5.3_flash/config.yaml multimodal data settings."""
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
    """Build the GLM-5.3-Flash multimodal dataloader."""
    from fsdp_turbo.training.dataloader import build_dataloader

    return build_dataloader(get_data_config(), tokenizer=processor.tokenizer, processor=processor)


def _submodule(model, name):
    """Resolve a top-level submodule on the ForConditionalGeneration or its inner model."""
    inner = getattr(model, 'model', model)
    return getattr(inner, name, None) or getattr(model, name, None)


def initialize_materialized_model(model, dtype):
    """Deterministically initialize a GLM-5.3-Flash model after ``to_empty``."""
    # Constructors run on meta in this ST, so constructor-time initializers do
    # not populate the parameters materialized later by ``to_empty``. Re-run
    # local reset hooks first for custom MoE, convolution, router, KDA, and
    # vision parameters, then apply each pretrained submodel's canonical
    # initializer.
    for module in model.modules():
        reset_parameters = getattr(module, 'reset_parameters', None)
        if callable(reset_parameters):
            reset_parameters()

    text_model = _submodule(model, 'language_model')
    vision_model = _submodule(model, 'visual')
    lm_head = _submodule(model, 'lm_head')
    if text_model is not None:
        init_submodule_weights(text_model, name='language_model')
    if vision_model is not None:
        init_submodule_weights(vision_model, name='visual')
    if lm_head is not None:
        init_submodule_weights(lm_head, name='lm_head')

    # GLM-5.3-Flash RMSNorm and gated-norm weights plus KDA state parameters
    # have no reset hook and are not covered by the generic Linear/Embedding
    # initializers. Glm5NextTextUnweightedRMSNorm / HyperHead are parameter-free.
    # e_score_correction_bias is an fp32 Buffer kept out of bf16 casting.
    for module in model.modules():
        if type(module).__name__ in {'Glm5NextTextRMSNorm', 'Glm5NextTextRMSNormGated'}:
            torch.nn.init.ones_(module.weight)
        elif type(module).__name__ == 'Glm5NextTextForgetGate':
            # KDA state parameters live on the forget gate: A_log controls the
            # per-head decay rate (exp(A_log)), dt_bias offsets the raw gate.
            torch.nn.init.uniform_(module.A_log, 1, 16)
            module.A_log.data.log_()
            torch.nn.init.zeros_(module.dt_bias)
        elif type(module).__name__ == 'Glm5NextTextTopkRouter':
            torch.nn.init.zeros_(module.e_score_correction_bias)

    model.tie_weights()
    model.to(dtype=dtype)


# ============================================================
# Complete configuration
# ============================================================
def get_complete_custom_config():
    """Get the GLM-5.3-Flash FSDP + EP system-test configuration."""
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
                apply_modules={
                    'model.visual': {},
                    'model.language_model.embed_tokens': {},
                    'model.language_model.layers.{*}': {},
                    'model.language_model.lm_head': {},
                },
                param_dtype='bf16',
                reduce_dtype='fp32',
                hook_modules=['model.language_model.layers.{*}'],
                num_to_forward_prefetch=1,
                num_to_backward_prefetch=1,
            ),
            expert_parallel_size=EXPERT_PARALLEL_SIZE,
            expert_fully_shard_parallel_size=EXPERT_FULLY_SHARD_PARALLEL_SIZE,
            ep_plan=EPPlanConfig(
                apply_modules=['model.language_model.layers.{*}.mlp.experts'],
                dispatcher='custom_ep_forward_eager',
            ),
        ),
        memory=MemoryConfig(
            recompute=True,
            recompute_plan=[
                'model.visual.blocks.{*}',
                'model.language_model.layers.{*}',
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
        raise RuntimeError('GLM-5.3-Flash did not return a loss; ensure the batch contains labels.')
    if not torch.isfinite(loss).all().item():
        raise FloatingPointError(f'GLM-5.3-Flash produced a non-finite loss: {loss.detach().float().item()}.')

    loss.backward()
    grad_norm = compute_grad_norm(list(model.parameters()))
    if not torch.isfinite(torch.tensor(grad_norm)).item():
        raise FloatingPointError(f'GLM-5.3-Flash produced a non-finite gradient norm: {grad_norm}.')

    optimizer.step()
    clear_weight_cache()
    model.zero_grad()
    return loss, grad_norm


def trainer_with_config(config, config_name, description, num_steps=NUM_STEPS, enable_profiler=False):
    """Train GLM-5.3-Flash for a few steps with the supplied configuration."""
    initialization()

    if torch.accelerator.current_device_index() == 0:
        print(f'\n{"=" * 60}')
        print(f'Test: {config_name}')
        print(f'Description: {description}')
        print(f'{"=" * 60}\n')

    model, processor = get_model_processor(config)
    model.to_empty(device='npu')
    initialize_materialized_model(model, config.model.torch_dtype)

    # _keep_in_fp32_modules_strict (e_score_correction_bias / conv1d / dt_bias /
    # A_log) legitimately keeps a small set of parameters in fp32, so the dtype
    # set must be a subset of {bf16, fp32} instead of a single dtype.
    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    if not parameter_dtypes.issubset({config.model.torch_dtype, torch.float32}):
        raise RuntimeError(
            f'GLM-5.3-Flash parameters must be bf16 (plus the fp32 keep-in-fp32 set) before FSDP wrapping, '
            f'but found {parameter_dtypes}.'
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


def test_glm5_3_flash():
    """Run the GLM-5.3-Flash complete system-test configuration."""
    trainer_with_config(
        config=get_complete_custom_config(),
        config_name='complete_custom',
        description=(
            'FSDP + EP + Recompute + Mixed Precision + Hook Module + Prefetch'
            f' + EP={EXPERT_PARALLEL_SIZE}, EFP={EXPERT_FULLY_SHARD_PARALLEL_SIZE}'
            f' + dsa_implementation={DSA_IMPL}, indexer_implementation={INDEXER_IMPL}'
        ),
        enable_profiler=ENABLE_PROFILER,
    )

    if torch.accelerator.current_device_index() == 0:
        print(f'\n{"#" * 60}')
        print('All tests completed successfully!')
        print(f'{"#" * 60}\n')


if __name__ == '__main__':
    test_glm5_3_flash()

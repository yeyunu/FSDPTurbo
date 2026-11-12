# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import logging
import os
import time

import torch
from torch.optim import AdamW
from transformers import AutoConfig, AutoProcessor

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
from fsdp_turbo.utils.env import get_local_rank
from fsdp_turbo.utils.log import print_rank, set_log_level
from fsdp_turbo.utils.model import convert_model_dtype, init_submodule_weights, set_num_hidden_layers
from fsdp_turbo.utils.random import set_seed

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = os.environ.get('MODEL_PATH', 'Qwen/Qwen3.8-Flash-Next')
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

# Parallel configuration (8 NPU smoke: FSP=4, TP=1, EP=8, EFP=1; CP off).
FULLY_SHARD_PARALLEL_SIZE = int(os.environ.get('FULLY_SHARD_PARALLEL_SIZE', '4'))
EXPERT_PARALLEL_SIZE = int(os.environ.get('EXPERT_PARALLEL_SIZE', '8'))
EXPERT_FULLY_SHARD_PARALLEL_SIZE = int(os.environ.get('EXPERT_FULLY_SHARD_PARALLEL_SIZE', '1'))

# Profiler configuration
ENABLE_PROFILER = os.environ.get('ENABLE_PROFILER', '0').lower() in ('1', 'true', 'yes', 'on')
PROFILE_MEMORY = os.environ.get('PROFILE_MEMORY', '0').lower() in ('1', 'true', 'yes', 'on')
PROFILE_DIR = os.environ.get('PROFILE_DIR', './profile')


def _apply_fused_knobs(model_config, config):
    """Apply qwen4_exp fused-operator knobs shared by real and synthetic paths.

    GDN/QSA backends, grouped-expert matmul and recompute skips are not part of
    the generic ``ModelConfig`` dataclass and must be set directly on the
    HuggingFace config (mirrors examples/qwen3_8/train.py). PLE/MTP knobs are
    added by ``_overwrite_qwen4_exp_config`` on the real-weight path only.
    """
    text_cfg = getattr(model_config, "text_config", model_config)
    vision_cfg = getattr(model_config, "vision_config", None)

    text_cfg.gdn_implementation = "triton"
    text_cfg.causal_conv1d_implementation = "triton"
    text_cfg.qsa_implementation = "triton"
    cp_plan = config.distributed.cp_plan
    qsa_chunk_size = getattr(cp_plan, "qsa_cp_query_chunk_size", None)
    text_cfg.qsa_cp_query_chunk_size = (
        qsa_chunk_size if qsa_chunk_size is not None else getattr(text_cfg, "qsa_cp_query_chunk_size", 128)
    )
    qsa_sparse_gather = getattr(cp_plan, "qsa_sparse_gather", None)
    text_cfg.qsa_sparse_gather = (
        qsa_sparse_gather if qsa_sparse_gather is not None else getattr(text_cfg, "qsa_sparse_gather", True)
    )
    text_cfg.skip_gdn_recompute = True
    text_cfg.skip_flash_attn_recompute = True
    if vision_cfg is not None:
        vision_cfg.skip_flash_attn_recompute = True

    text_cfg.use_grouped_expert_matmul = True
    text_cfg._experts_implementation = "eager"
    model_config._experts_implementation = "eager"

    text_cfg.enable_chunk_loss = config.memory.chunk_loss
    text_cfg.chunk_loss_size = config.memory.chunk_loss_size
    text_cfg.enable_dynamic_chunk_loss = False

    text_cfg.enable_ep_balance = False
    text_cfg.max_dup_experts_num = 2


def _overwrite_qwen4_exp_config(model_config, config):
    """Apply qwen4_exp-specific config overrides on the HF model_config.

    Mirrors examples/qwen3_8/train.py::_overwrite_qwen4_exp_config: model-specific
    knobs (GDN/QSA backends, expert matmul, PLE CPU offload, chunk_loss, MTP) are
    not part of the generic ``ModelConfig`` dataclass, so they are set directly on
    the HuggingFace config here.
    """
    _apply_fused_knobs(model_config, config)
    text_cfg = getattr(model_config, "text_config", model_config)

    # PLE n-gram embedding: CPU-offload variant (zero-size table at __init__;
    # materialized on host after FSDPTurbo via materialize_ngram_embedding_on_cpu).
    text_cfg.ple_embedding_on_cpu = True
    text_cfg._ngram_hf_dir = config.model.model_name_or_path

    ngram_vocab_size_base = os.environ.get("NGRAM_VOCAB_SIZE_BASE", "").strip()
    if ngram_vocab_size_base:
        text_cfg.ngram_vocab_size_base = int(ngram_vocab_size_base)
        if text_cfg.ngram_vocab_size_base <= 0:
            raise ValueError(f"Invalid NGRAM_VOCAB_SIZE_BASE='{ngram_vocab_size_base}'.")

    # MTP is disabled for the 8-NPU smoke test; enable via model.mtp_num_layers>0.
    text_cfg.mtp_num_layers = getattr(config.model, "mtp_num_layers", 0)
    if text_cfg.mtp_num_layers and text_cfg.mtp_num_layers < 0:
        raise ValueError(f"Invalid mtp_num_layers='{text_cfg.mtp_num_layers}'. Must be a non-negative integer.")


# Synthesis dimensions for the no-checkpoint self-bootstrap path. Tuned so the
# tiny model fits 8 NPUs under EP=8 and exercises the fused MoE / QSA paths.
_SYNTHETIC_VOCAB_SIZE = 64
_SYNTHETIC_HIDDEN_SIZE = 64
_SYNTHETIC_NUM_EXPERTS = 16
_SYNTHETIC_SEQ_LEN = 32


def _build_synthetic_config(num_layers):
    """Build a self-contained ``Qwen4ExpConfig`` for the no-checkpoint path.

    PLE (``ple_layer_ids=[]``) and MTP (``mtp_num_layers=0``) are disabled so the
    model needs no n-gram shards / no MTP block — the smoke test covers the fused
    decoder, MoE dispatch and QSA indexer without external assets. Mirrors the
    tiny-config shape used by tests/unit_tests/models/test_qwen4_exp_merge.py.
    """
    from fsdp_turbo.models.qwen.modeling_qwen4_exp import Qwen4ExpConfig

    return Qwen4ExpConfig.from_dict(
        {
            'text_config': {
                'vocab_size': _SYNTHETIC_VOCAB_SIZE,
                'hidden_size': _SYNTHETIC_HIDDEN_SIZE,
                'num_hidden_layers': num_layers,
                'num_attention_heads': 4,
                'num_key_value_heads': 2,
                'head_dim': 16,
                'layer_types': ['full_attention'] * num_layers,
                'moe_intermediate_size': _SYNTHETIC_HIDDEN_SIZE,
                'shared_expert_intermediate_size': _SYNTHETIC_HIDDEN_SIZE,
                'num_experts': _SYNTHETIC_NUM_EXPERTS,
                'num_experts_per_tok': 4,
                'hc_count': 2,
                'hc_lowrank': 4,
                'ple_layer_ids': [],
                'use_cache': False,
                'indexer_n_heads': 4,
                'indexer_kv_heads': 1,
                'indexer_head_dim': 16,
                'indexer_budget': 4,
                'indexer_compress_ratio': 2,
                'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.0, 'mrope_section': [1, 1, 2]},
            },
            'vision_config': {
                'depth': 1,
                'hidden_size': _SYNTHETIC_HIDDEN_SIZE,
                'intermediate_size': _SYNTHETIC_HIDDEN_SIZE,
                'num_heads': 2,
                'out_hidden_size': _SYNTHETIC_HIDDEN_SIZE,
                'num_position_embeddings': 16,
            },
        }
    )


def initialization():
    """Initialize distributed environment, accelerator compatibility, seed, and logging."""
    local_rank = get_local_rank()
    torch.accelerator.set_device_index(local_rank)

    set_accelerator_compatible()
    set_seed(SEED)
    set_log_level(LOG_LEVEL)


def get_model_processor(config):
    """Build a reduced-layer Qwen3.8-Flash-Next model for ST.

    Two paths:
      - Real checkpoint dir (``MODEL_PATH`` points at a directory holding
        config.json + tokenizer/processor): load via ``AutoConfig.from_pretrained``
        and ``AutoProcessor.from_pretrained``; PLE/MTP knobs are applied so the
        full production path runs.
      - No checkpoint dir (``MODEL_PATH`` unset / absent — the CI default): build
        a self-contained ``Qwen4ExpConfig`` (PLE/MTP disabled) and return
        ``processor=None``; the caller drives the training loop with synthetic
        text batches, exercising the fused decoder + MoE dispatch + QSA indexer
        without any external asset.

    In both paths the FSDPTurbo subclass ``Qwen4ExpForConditionalGeneration`` is
    built on the ``meta`` device so it carries no real parameters; the caller
    materializes it via ``to_empty`` and initializes it before FSDPTurbo wrapping.

    Returns ``(model, processor, is_synthetic)``.
    """
    model_path = config.model.model_name_or_path
    is_synthetic = not model_path or not os.path.isdir(model_path)

    if is_synthetic:
        model_config = _build_synthetic_config(config.model.num_hidden_layers)
    else:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        set_num_hidden_layers(model_config, config.model.num_hidden_layers)
        _overwrite_qwen4_exp_config(model_config, config)

    # eager attention is the numerically safe reference; NPU fused attention is
    # exercised by the qwen4_exp ops-dispatched RoPE / indexer paths regardless.
    text_cfg = getattr(model_config, "text_config", model_config)
    vision_cfg = getattr(model_config, "vision_config", None)
    for sub_cfg in (model_config, text_cfg, vision_cfg):
        if sub_cfg is not None:
            sub_cfg._attn_implementation = "eager"

    if is_synthetic:
        _apply_fused_knobs(model_config, config)

    from fsdp_turbo.models.qwen.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration

    with torch.device('meta'):
        model = Qwen4ExpForConditionalGeneration(model_config)

    processor = None
    if not is_synthetic:
        processor = AutoProcessor.from_pretrained(config.model.tokenizer_name_or_path, trust_remote_code=True)
    return model, processor, is_synthetic


def get_data_config():
    """Mirror examples/qwen3_8/config.yaml multimodal data settings."""
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


def get_dataloader(processor, config, is_synthetic):
    """Build the Qwen3.8-Flash-Next dataloader.

    Real path: multimodal dataloader from ``build_dataloader`` + processor.
    Synthetic path: a generator of plain-text batches (``input_ids`` /
    ``attention_mask`` / ``labels``) sized to the synthetic config — no tokenizer
    or dataset asset is needed.
    """
    if is_synthetic:
        return _synthetic_batches()

    from fsdp_turbo.training.dataloader import build_dataloader

    return build_dataloader(get_data_config(), tokenizer=processor.tokenizer, processor=processor)


def _synthetic_batches():
    """Yield ``NUM_STEPS`` synthetic text batches for the no-checkpoint path.

    Each batch has ``input_ids`` / ``attention_mask`` / ``labels`` shaped
    ``[batch, seq]`` with values drawn from ``[0, vocab_size)`` (labels mask the
    first token to -100 to mirror the standard LM shift). ``pixel_values`` is left
    absent so the model runs its text-only path.
    """
    batch = BATCH_SIZE
    seq = _SYNTHETIC_SEQ_LEN
    generator = torch.Generator()
    generator.manual_seed(SEED)
    for _ in range(NUM_STEPS):
        input_ids = torch.randint(0, _SYNTHETIC_VOCAB_SIZE, (batch, seq), generator=generator, dtype=torch.long)
        attention_mask = torch.ones((batch, seq), dtype=torch.long)
        labels = input_ids.clone()
        labels[:, 0] = -100
        yield {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def initialize_materialized_model(model, dtype, is_synthetic):
    """Deterministically initialize a Qwen3.8-Flash-Next model after ``to_empty``."""
    text_model = getattr(model.model, 'language_model', None)
    vision_model = getattr(model.model, 'visual', None)
    lm_head = getattr(model, 'lm_head', None)
    if text_model is not None:
        init_submodule_weights(text_model, name='language_model')
    if vision_model is not None:
        init_submodule_weights(vision_model, name='visual')
    if lm_head is not None:
        init_submodule_weights(lm_head, name='lm_head')

    model.tie_weights()
    convert_model_dtype(model, dtype)

    if not is_synthetic:
        # convert_model_dtype casts the PLE hashing buffers to bf16; they must
        # stay torch.long for bitwise/modular ops (aclnnBitwiseXorTensor rejects
        # floats). The synthetic path disables PLE, so there are no such buffers.
        from fsdp_turbo.models.qwen.modeling_qwen4_exp import restore_ngram_integer_buffers

        restore_ngram_integer_buffers(model)

    # Qwen4ExpTextNGramEmbeddingCpu.__init__ calls get_parallel_state(), creating a
    # Singleton with default FSDP=1/EP=1; clear it so FSDPTurbo re-inits correctly.
    # (Only the real path builds the PLE module, but the clear is a harmless no-op
    # when there is nothing to clear.)
    from fsdp_turbo.distributed.parallel_state import ParallelState, Singleton

    Singleton._instances.pop(ParallelState, None)


# ============================================================
# Complete configuration
# ============================================================
def get_complete_custom_config():
    """Get the Qwen3.8-Flash-Next FSDP + EP system-test configuration.

    Features:
    - FSDP over visual + language + embeddings + lm_head (post-order nested sharding)
    - Expert Parallel + Expert Fully Sharded Data Parallel over MoE experts
    - PLE n-gram embedding excluded from FSDP, materialized frozen on CPU
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
                # PLE table excluded from FSDP: frozen on CPU via
                # materialize_ngram_embedding_on_cpu after FSDPTurbo.
                ignored_modules=['model.language_model.layers.1.ple.ple_embedding'],
                apply_modules={
                    'model.visual': {},
                    'model.visual.blocks.{*}': {},
                    'model.language_model.embed_tokens': {},
                    'model.language_model': {},
                    'model.language_model.layers.{*}': {},
                    'model.language_model.layers.{*}.linear_attn': {},
                    'model.language_model.layers.{*}.mlp.experts': {},
                    'lm_head': {},
                },
                param_dtype='bf16',
                reduce_dtype='fp32',
                output_dtype='bf16',
                cast_forward_inputs=True,
                hook_modules=['model.language_model.layers.{*}'],
                fsdp_implementation='native',
                num_to_forward_prefetch=1,
                num_to_backward_prefetch=1,
            ),
            expert_parallel_size=EXPERT_PARALLEL_SIZE,
            expert_fully_shard_parallel_size=EXPERT_FULLY_SHARD_PARALLEL_SIZE,
            ep_plan=EPPlanConfig(
                apply_modules=['model.language_model.layers.{*}.mlp.experts'],
                dispatcher='all2all',
                use_fused_ops=True,
                reuse_dispatch_splits=True,
                apply_efsdp_modules=['model.language_model.layers.{*}.mlp'],
            ),
        ),
        memory=MemoryConfig(
            recompute=True,
            recompute_plan=[
                'model.visual.blocks.{*}',
                'model.language_model.layers.{*}',
            ],
            chunk_loss=False,
            chunk_loss_size=None,
        ),
    )


# ============================================================
# Training loop
# ============================================================
def train_step(model, batch, optimizer):
    """Execute a single forward, backward, and optimizer step."""
    batch['use_cache'] = False
    batch['output_hidden_states'] = True
    accelerator = torch.accelerator.current_accelerator()
    batch = {key: (value.to(accelerator) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}

    outputs = model.forward(**batch)
    loss = outputs.loss
    if loss is None:
        raise RuntimeError('Qwen3.8-Flash-Next did not return a loss; ensure the batch contains labels.')
    if not torch.isfinite(loss).all().item():
        raise FloatingPointError(f'Qwen3.8-Flash-Next produced a non-finite loss: {loss.detach().float().item()}.')

    loss.backward()
    grad_norm = compute_grad_norm(list(model.parameters()))
    if not torch.isfinite(torch.tensor(grad_norm)).item():
        raise FloatingPointError(f'Qwen3.8-Flash-Next produced a non-finite gradient norm: {grad_norm}.')

    optimizer.step()
    clear_weight_cache()
    model.zero_grad()
    return loss, grad_norm


def trainer_with_config(config, config_name, description, num_steps=NUM_STEPS, enable_profiler=False):
    """Train Qwen3.8-Flash-Next for a few steps with the supplied configuration."""
    initialization()

    is_rank0 = torch.accelerator.current_device_index() == 0
    print_rank(logger.info, f'\n{"=" * 60}', ranks=0)
    print_rank(logger.info, f'Test: {config_name}', ranks=0)
    print_rank(logger.info, f'Description: {description}', ranks=0)
    print_rank(logger.info, f'{"=" * 60}\n', ranks=0)

    model, processor, is_synthetic = get_model_processor(config)
    model.to_empty(device=torch.accelerator.current_accelerator())
    initialize_materialized_model(model, config.model.torch_dtype, is_synthetic)

    # qwen3_8's fsdp_plan.apply_modules has parent-child nesting (model.visual ->
    # blocks.{*}, model.language_model -> layers.{*} -> linear_attn/mlp.experts).
    # FSDP2 requires children to be sharded before parents, so patch get_fsdp_modules
    # to use post-order traversal during FSDPTurbo init (mirrors examples/qwen3_8/train.py).
    from fsdp_turbo.distributed.fully_shard_parallel import (
        fully_shard_parallel as _fsp,
    )

    _orig_get_fsdp_modules = _fsp.get_fsdp_modules

    def _get_fsdp_modules_post_order(model_arg, fsdp_plan, ignored_modules):
        fsdp_modules = {}
        for name, module in _fsp._post_order_traverse(model_arg):
            for pattern, plan in fsdp_plan.apply_modules.items():
                if _fsp.module_name_match(pattern, name) and name not in ignored_modules:
                    if module not in fsdp_modules:
                        fsdp_modules[module] = {}
                    fsdp_modules.get(module).update(plan)
        if len(fsdp_modules) == 0:
            raise RuntimeError(f'[FSDP2] No module named {fsdp_plan.apply_modules.keys()}.')
        return fsdp_modules

    _fsp.get_fsdp_modules = _get_fsdp_modules_post_order
    try:
        model = FSDPTurbo(config, model)
    finally:
        _fsp.get_fsdp_modules = _orig_get_fsdp_modules

    if not is_synthetic:
        # VP off (CPU offload): materialize the frozen PLE n-gram tables on CPU
        # after FSDPTurbo so model weights are already on device and CPU RAM is
        # freed for the (shrunk via NGRAM_VOCAB_SIZE_BASE) table. With no
        # checkpoint shards present the table is random-initialized
        # (strict_weights=False), sufficient for a smoke test. The synthetic path
        # disables PLE, so there is nothing to materialize.
        from fsdp_turbo.models.qwen.modeling_qwen4_exp import materialize_ngram_embedding_on_cpu

        materialize_ngram_embedding_on_cpu(
            model,
            dtype=torch.bfloat16,
            hf_dir=config.model.model_name_or_path,
            freeze=True,
        )

    dataloader = get_dataloader(processor, config, is_synthetic)

    if is_rank0:
        print_rank(logger.info, f'{model}', ranks=0)
    model.model.train(True)
    optimizer = AdamW(model.model.parameters(), lr=LR)

    if enable_profiler:
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError('ENABLE_PROFILER requires torch_npu.') from exc
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
            torch.accelerator.reset_peak_memory_stats()

            step_start = time.perf_counter()
            loss, grad_norm = train_step(model, batch, optimizer)
            torch.accelerator.synchronize()
            step_time = time.perf_counter() - step_start

            if enable_profiler:
                prof.step()

            torch.distributed.barrier()
            if is_rank0:
                # NOTE: intentionally ``print`` (not print_rank) — acquire_json.py
                # parses this exact line from stdout; print_rank prefixes would
                # break baseline extraction. See tests/README.md §ST ②.
                print(
                    f'[Rank {torch.accelerator.current_device_index()}] step {step} '
                    f'step time: {step_time:.4f}s '
                    f'loss: {loss.item():.4f} '
                    f'grad norm: {grad_norm:.4f} '
                    f'memory: {torch.accelerator.memory_allocated() / 1024**3:.4f} GB '
                    f'max memory: {torch.accelerator.max_memory_allocated() / 1024**3:.4f} GB'
                )

    if is_rank0:
        print_rank(logger.info, f'\nTest completed: {config_name}\n', ranks=0)


def test_qwen3_8_flash_next():
    """
    Author: FSDPTurbo Team
    Date: 2026-11-11
    Description: Qwen3.8-Flash-Next (qwen4_exp) reduced-layer smoke system test on 8 NPUs.
        Covers FSP=4, EP=8, EFP=1, post-fusion block-aligned FSDP, MoE all2all dispatch
        with E-FSDP, PLE n-gram embedding CPU offload, and recompute.
    Remarks: When MODEL_PATH points at a real checkpoint dir (config.json + tokenizer/
        processor, no weights) the test runs the production path with PLE CPU offload
        and a multimodal dataloader; NGRAM_VOCAB_SIZE_BASE shrinks the PLE table for CI.
        When MODEL_PATH is unset or absent (the CI default for a brand-new model) the
        test self-bootstraps a tiny Qwen4ExpConfig (PLE/MTP disabled) and synthetic
        text batches with no external asset. No baseline JSON is shipped; the case is
        function-only (training runs = PASS).
    """
    trainer_with_config(
        config=get_complete_custom_config(),
        config_name='complete_custom',
        description=(
            'FSDP + EP + E-FSDP + Recompute + Mixed Precision + Hook Module + Prefetch'
            f' + FSP={FULLY_SHARD_PARALLEL_SIZE}, EP={EXPERT_PARALLEL_SIZE},'
            f' EFP={EXPERT_FULLY_SHARD_PARALLEL_SIZE}'
        ),
        enable_profiler=ENABLE_PROFILER,
    )

    if torch.accelerator.current_device_index() == 0:
        print_rank(logger.info, f'\n{"#" * 60}', ranks=0)
        print_rank(logger.info, 'All tests completed successfully!', ranks=0)
        print_rank(logger.info, f'{"#" * 60}\n', ranks=0)


if __name__ == '__main__':
    test_qwen3_8_flash_next()

# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Keep the configuration schema and its __post_init__ validation together.
# pylint: disable=too-many-lines
import logging
import math
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Callable, Literal, Union, Optional
from pathlib import Path

import torch
import yaml

from fsdp_turbo.utils.dtype import get_dtype
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match

from fsdp_turbo.memory.recompute.recompute import RecomputePlanConfig
from fsdp_turbo.memory.op_replay.op_replay import OpReplayScopeConfig
from fsdp_turbo.memory.swap.swap_manager import SwapPlanConfig
from fsdp_turbo.memory.swap.quantizer import QUANTIZERS, Fp8BlockwiseQuantizer
from fsdp_turbo.ops.quantize import fp8_payload_quant_supported

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name_or_path: str = ""
    # Optional architecture JSON; DeepSeek-V4.1 otherwise reads model_name_or_path/config.json.
    config_path: Optional[str] = None
    tokenizer_name_or_path: str = ""
    torch_dtype: torch.dtype = torch.bfloat16
    load_pretrained_weights: bool = False
    pretrained_strict: bool = True
    pretrained_engram_rows_per_chunk: int = 65_536
    # Optional demo/debug knob for building a shallower language model.
    num_hidden_layers: Optional[int] = None
    engram_storage_backend: Optional[Literal["host_offload", "elastic_buffer"]] = None
    use_sparse_flash_attn: bool = False
    use_fused_lightning_indexer: bool = False
    use_fused_lightning_indexer_loss: bool = False
    indexer_loss_coeff: float = 0.0
    track_indexer_kl_loss: bool = False
    use_fused_indexer_kl_loss: bool = True
    # GLM-5.3-Flash DSA/indexer/attention dispatch knobs (model_config-driven;
    # train.py injects them into the HF text/vision sub-configs).
    # dsa_implementation: 'dense' (mask-based) or 'sfa' (npu_sparse_flash_attention).
    dsa_implementation: str = "dense"
    # indexer_implementation: 'eager' or 'triton' (lightning_indexer_score).
    indexer_implementation: str = "eager"
    # 1 routes MLA/DSA text attention and the vision tower through
    # npu_fusion_attention; 0 keeps the bit-exact eager path.
    npu_fused_attention: bool = True
    use_fused_compressor: bool = False
    use_fused_mhc: bool = False
    use_fused_clipped_swiglu: bool = False
    # Use the triton in-place RoPE kernel (mindspeed_ops.api.triton.inplace_rope)
    # over the trailing rotary block on NPU.
    use_triton_inplace_rope: bool = False
    use_deepgemm_wo_a: bool = False
    use_fused_vision_flash_attn: bool = False
    # Use CANN InplacePartialRotaryMul/Grad for the large per-layer q/o RoPE.
    use_fused_partial_rotary: bool = False
    # Fuse the Engram n-gram memory gating via the PyPTO-Pro kernels on Ascend950DT.
    use_fused_engram: bool = False
    # DeepSeek-V4.1 head projection precision; cross entropy remains FP32.
    head_compute_dtype: Literal["fp32", "bf16", "hf32"] = "fp32"
    attn_implementation: str = 'eager'
    # MTP (Multi-Token Prediction) layers to train. 0 disables MTP.
    mtp_num_layers: int = 0
    # Scaling factor applied to the mean MTP loss before adding to the main loss.
    mtp_loss_scaling_factor: float = 0.1
    use_fused_causal_conv1d: bool = False
    use_fused_gdn: bool = False
    tilekernels: Dict[str, bool] = field(default_factory=lambda: {"tilekernels_mhc_post": False})
    # Perf-test switch: skip checkpoint reading, build the network on the
    # meta device and zero-fill per-rank shards after FSDP sharding. Values
    # are meaningless; keep False whenever real weights are needed.
    skip_weight_load: bool = False


@dataclass
class OptimizerConfig:
    optimizer_type: str = "AdamW"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    lr: float = 1e-4
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "cosine"
    min_lr: float = 0.0
    clip_grad: float = 1.0
    clip_grad_norm_type: float = 2.0
    # Swap optimizer (AdamWSwap) only: fraction of free memory used for batched H2D transfers
    mem_fraction_static: Optional[float] = None
    # Swap optimizer (AdamWSwap) only: state keys participating in swap; None uses the default
    state_keys: Optional[List[str]] = None
    # Swap optimizer (AdamWSwap) only: prefetch one param to device buffer if it sets True
    param_prefetch: Optional[bool] = False
    # Re-point CPU-offloaded gradients at their clipped device copies so
    # AdamWSwap consumes them directly, skipping the clip D2H write-back and
    # the step H2D upload. The host gradient is released, so no extra copy
    # is kept. Only effective when cpu_offload and swap_optimizer are both
    # enabled.
    clip_grad_keep_on_device: Optional[bool] = False
    # Swap optimizer (AdamWSwap) only: module name pattern whose backward pre-hook
    # triggers states prefetch during backward; None = H2D inside step().
    prefetch_states_at_backward: Optional[str] = None
    # Swap optimizer (AdamWSwap) only: chunk budget cap in numel; effective
    # budget is min(swap_chunk_numel, swap_numel // 2) under ping-pong. Default 4M.
    swap_chunk_numel: Optional[int] = None
    # Swap optimizer (AdamWSwap) only: smallest params (ascending numel) keep
    # their states permanently on device; 0 disables.
    resident_states_ratio: Optional[float] = None
    # Swap optimizer (AdamWSwap) only: fraction of free memory at the backward
    # prefetch trigger point used for prefetching. Default 0.7.
    swap_prefetch_budget_ratio: Optional[float] = None
    # Swap optimizer (AdamWSwap) only: number of ordered sub-batches the backward
    # prefetch is split into. Default 8.
    swap_prefetch_batches: Optional[int] = None


@dataclass
class ProfileConfig:
    enabled: bool = False
    # Collect the profiler only on the listed global ranks (zero-overhead
    # nullcontext elsewhere); None = all ranks.
    ranks: Optional[List[int]] = None
    wait_steps: int = 0
    warmup_steps: int = 0
    active_steps: int = 1
    repeat: int = 1
    skip_first: int = 0
    output_dir: str = "./profile"
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False


@dataclass
class MemorySnapshotConfig:
    """Device memory snapshot collection (``run.memory_snapshot``).

    ``enable`` turns snapshot dumps on: with ``end_step`` set, allocator
    history is recorded from ``start_step`` and one snapshot is written at
    ``end_step``; without one, one snapshot is written per step from
    ``start_step``. ``mem_info`` independently prints a per-step memory line.
    Steps are 0-based micro steps and the window is inclusive on both ends.
    The OOM trap (``OOM_SNAPSHOT_ENABLE``/``OOM_SNAPSHOT_PATH``) stays
    environment-variable controlled and is not part of this schema.
    """

    enable: bool = False
    start_step: int = 0
    end_step: Optional[int] = None
    # Snapshot directory; None means <checkpoint.output_dir>/memory_snapshot.
    save_path: Optional[str] = None
    # Global ranks that record and dump; None means all ranks.
    dump_ranks: Optional[List[int]] = None
    stacks: str = "all"
    max_entries: Optional[int] = None
    mem_info: bool = False


@dataclass
class DataConfig:
    dataset_path: str = ""
    dataset_config: Optional[str] = None
    text_column: str = "text"
    split: str = "train"
    batch_size: int = 1
    max_seq_length: int = 4096
    num_workers: int = 0
    pin_memory: bool = True
    shuffle: bool = True
    is_multimodal: bool = False
    image_column: str = "images"
    media_key: str = "images"
    conversation_column: str = "messages"
    # DeepSeek V4.1 packed JSONL can run with documents on the token (TND)
    # or batch (BSND) axis. None preserves the record's layout.
    sfa_layout: Optional[str] = None


@dataclass
class TrainRunConfig:
    seed: int = 42
    deterministic: bool = False
    max_steps: int = -1
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 1
    logging_steps: int = 1
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    memory_snapshot: MemorySnapshotConfig = field(default_factory=MemorySnapshotConfig)


@dataclass
class CheckpointConfig:
    output_dir: str = "./output"
    resume_from_checkpoint: Optional[str] = None
    save_steps: int = 500
    strict: bool = True
    save_optim: bool = True
    load_optim: bool = True


@dataclass
class FSDPPlanConfig:
    ignored_modules: List[str] = field(default_factory=list)
    apply_modules: Dict[str, Any] = None

    # mp_policy settings
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    output_dtype: Optional[str] = None
    cast_forward_inputs: bool = True

    # offload_policy settings
    cpu_offload: bool = False
    pin_memory: bool = True

    # prefetch settings
    num_to_forward_prefetch: Optional[int] = 0
    num_to_backward_prefetch: Optional[int] = 0
    prefetch_wait_a2a: bool = False
    backward_prefetch_on_moe_input: bool = False

    # fsdp2 hook manager
    hook_modules: Optional[List[str]] = None

    # FSDP implementation strategy
    # 'custom': Use FSDPTurbo custom FSDP implementation
    # 'native': Use PyTorch native FSDP implementation (default)
    fsdp_implementation: Literal["custom", "native"] = "native"

    # apply_modules patterns in forward execution order; backward uses the reverse.
    # None preserves module registration order.
    prefetch_modules: Optional[List[str]] = None


@dataclass
class TPPlanConfig:
    colwise_parallel: List[str] = None
    rowwise_parallel: List[str] = None
    sequence_parallel: List[str] = None


@dataclass
class CPPlanConfig:
    """Legacy CP targets and KDA/MLA backend selection."""

    # None means the key is not provided; an explicit empty list opts out of
    # patching for models whose Ulysses forwards are built into the modeling
    # classes (validate_cp_config keeps the two cases apart).
    ulysses_function_patches: Optional[List[Dict[str, Any]]] = None
    kcp_function_patches: List[Dict[str, Any]] = field(default_factory=list)
    kv_allgather_function_patches: List[Dict[str, Any]] = field(default_factory=list)
    loss_function_patches: List[Dict[str, Any]] = field(default_factory=list)
    kda_backend: str = "kcp"  # hybrid_p2p selects the Hybrid P2P subclass path.
    mla_backend: str | None = None
    qsa_cp_query_chunk_size: Optional[int] = None
    qsa_sparse_gather: Optional[bool] = None
    # Derived by validate_cp_config from the input layout; never set by hand.
    tnd_enabled: bool = field(default=False, init=False, repr=False)
    # full_prefix gathers the whole KV prefix; halo_prefix only the reachable tail.
    tnd_exchange_style: str = "full_prefix"


def uses_hybrid_cp_zigzag_layout(cp_plan: CPPlanConfig | None) -> bool:
    """Return whether a normalized CP plan selects Hybrid CP front/back shards.

    A missing plan selects contiguous shards. Non-empty plans must have passed
    FSDPTurboConfig validation; this query neither normalizes nor mutates them.
    """
    return cp_plan is not None and cp_plan.kda_backend == "hybrid_p2p" and cp_plan.mla_backend != "ulysses"


@dataclass
class MoonEPConfig:
    """MoonEP communication, memory and static-shape tuning options.

    ``tokens_per_rank`` and ``top_k`` may be omitted on CUDA, where they are
    inferred on the first expert forward and then locked. The Ascend backend
    requires them up front, because every symmetric allocation must be
    registered before its symmetric heap is initialized.

    MoonEP training always reserves one prefetch slot per local expert: that is
    the ``E/R`` bound proved tight in Appendix E of the Kimi K3 report, so
    planning is always feasible and training is never interrupted.
    """

    num_sms: int = 32
    token_padding: int = 128
    comm_stream_priority: int = -1
    # Let the collectives complete asynchronously on a side stream where the
    # backend can. Off by default because Ascend cannot: its kernels carry
    # device-side all-rank barriers whose generation counters desynchronise on a
    # side stream, so the request is masked off there and setting it has no
    # effect. Only the CUDA backend acts on it.
    async_finish: bool = False
    tokens_per_rank: Optional[int] = None
    top_k: Optional[int] = None
    # Trade one extra all-to-all in the backward pass for K times less saved
    # activation memory on the first expert GEMM (K3 report S5.2.2). Leave this
    # off when full-layer activation checkpointing is enabled, because the
    # dispatch is already replayed by the recompute.
    recompute_dispatch: bool = False
    # Run the device-side plan and routing assertions on every step instead of
    # only the first one. Useful during bring-up; costs a few extra elementwise
    # kernels per MoE layer.
    validate_plan_invariants: bool = False
    # CANN SHMEM bootstrap endpoint for a multi-node EP group on Ascend, for
    # example "tcp://10.0.0.1:29501". Derived from MASTER_ADDR/MASTER_PORT when
    # left unset.
    shmem_ip_port: Optional[str] = None
    # Ascend only: use CANN SHMEM's stream-ordered global barrier before the
    # weight-gradient reduction instead of a host synchronize plus dist.barrier,
    # saving one pipeline drain per MoE layer in the backward pass. On by
    # default since it measured 2.3-3.8% faster with bit-identical losses and
    # passed the per-tensor correctness check at 3 and 20 steps. Kept as a knob
    # because it is the synchronization that a past data race turned on, and 20
    # steps is not a proof.
    shmem_stream_barrier: bool = True
    trace_enabled: bool = False
    trace_sync_enabled: bool = False
    vmm_probe_enabled: bool = False
    npu_async_comm: bool = False


@dataclass
class KimiRouterDebugConfig:
    apply_custom_router: bool = False
    target_max_vio: float = -1.0
    router_mode: str = "lognormal"
    record_load_balance: bool = False

    def __post_init__(self):
        if not self.apply_custom_router:
            return
        target = float(self.target_max_vio)
        if math.isnan(target) or math.isinf(target) or target < 0:
            raise ValueError("apply_custom_router requires a finite target_max_vio >= 0.")
        if self.router_mode not in ("exact", "lognormal"):
            raise ValueError("router_mode must be exact or lognormal.")


@dataclass
class EPPlanConfig:
    """Configuration for Expert Parallelism (EP) plan for MoE models."""

    apply_modules: List[str] = field(default_factory=list)
    dispatcher: Union[Literal["all2all", "allgather", "mc2", "domino", "chunkmoe", "moonep"], Callable, str] = "all2all"
    fixed_router: bool = False
    apply_efsdp_modules: List[str] = field(default_factory=list)
    moonep_config: Optional[MoonEPConfig] = None
    router_debug: Optional[KimiRouterDebugConfig] = None
    # Gradient divide factor for expert HSDP. When unset (None) it is
    # computed in the config validation as ep * edp * efsdp.
    gradient_divide_factor: Optional[float] = None
    use_fused_ops: bool = True
    use_swiglu: bool = True
    # Reuse the forward's MoE a2a split sizes in recompute and the backward
    # wgrad group_list, skipping host-sync points; defaults to False.
    reuse_dispatch_splits: bool = False
    # Max entries per expert-module split-cache stack; steady-state depth
    # equals the number of in-flight micro-batches. Evicted entries fall
    # back to the original recompute path (perf only).
    dispatch_split_cache_cap: int = 8
    # Fixed chunk count takes precedence over token-based chunk size. Zero
    # disables that mode; with both zero, execute one chunk.
    moe_chunk_size: int = field(
        default=0,
        metadata={
            "help": "Chunk size for chunked MoE dispatch; 0 means no chunking (the whole token sequence is one chunk)."
        },
    )
    moe_num_chunks: int = field(
        default=0,
        metadata={
            "help": "Fixed number of chunks for chunked MoE dispatch; chunk length is derived as ceil(num_tokens / moe_num_chunks) and tail chunks are padded empty so every EP rank has the same chunk count. 0 means disabled; mutually exclusive with moe_chunk_size."
        },
    )
    chunk_moe_recompute: bool = field(
        default=False,
        metadata={"help": "Whether to enable Gradient Checkpointing (Activation Recomputation)."},
    )

    enable_ep_balance: bool = False
    max_dup_experts_num: int = 2

    def __post_init__(self):
        if self.moe_chunk_size > 0 and self.moe_num_chunks > 0:
            logger.warning(
                "moe_chunk_size (%s) and moe_num_chunks (%s) are both set; "
                "moe_num_chunks takes precedence and moe_chunk_size is ignored.",
                self.moe_chunk_size,
                self.moe_num_chunks,
            )
        if self.max_dup_experts_num < 0:
            raise ValueError("max_dup_experts_num must be non-negative.")
        cap = self.dispatch_split_cache_cap
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError("ep_plan.dispatch_split_cache_cap must be an integer >= 1.")
        if self.enable_ep_balance and self.dispatcher not in ("all2all", "allgather"):
            raise ValueError("EP balance only supports all2all, allgather dispatchers.")


@dataclass
class VPPlanConfig:
    """Configuration for vocabulary-parallel embedding modules.

    A ``[vocab, dim]`` embedding weight is sharded along the vocab dim
    (``Shard(0)``) over the ``vp`` mesh and the module forward is replaced
    with an all-to-all dispatcher.
    """

    apply_modules: List[str] = None
    apply_vfsdp_modules: List[str] = None
    # Gradient divide factor for the return all-to-all. When unset (None) it
    # defaults to world size, matching the FSDP gradient averaging applied to
    # dense parameters.
    gradient_divide_factor: Optional[float] = None


@dataclass
class QuantizeConfig:
    quant_format: Optional[str] = None
    quant_recipe: Optional[str] = None
    block_size: int = 32
    quant_apply_modules: List[str] = None
    quant_ignored_modules: List[str] = None
    converters: List[str] = None
    enable_fsdp_low_precision_all_gather: bool = True
    fsdp_low_precision_all_gather_mode: str = "on-demand"
    _recompute_aware: bool = field(default=False, init=False, repr=False)
    fsdp_world_size: int = 1
    quant_gmm: bool = False
    gemm_gradient_accumulation_fusion: bool = False
    extra_args: Dict[str, Any] = field(default_factory=dict)  # for future extensibility


@dataclass
class DistributedConfig:
    data_parallel_size: int = 1

    fully_shard_parallel_size: int = 1
    fsdp_plan: FSDPPlanConfig = None

    tensor_parallel_size: int = 1
    tp_plan: TPPlanConfig = None

    kv_allgather_parallel_size: int = 1
    kcp_parallel_size: int = 1
    ulysses_parallel_size: int = 1
    cp_plan: CPPlanConfig = None

    expert_parallel_size: int = 1
    expert_fully_shard_parallel_size: int = 1
    expert_data_parallel_size: int = 1
    ep_plan: EPPlanConfig = None

    # VP/VFSDP for huge embedding tables (e.g. the qwen4_exp PLE n-gram table).
    # vocab_data_parallel_size * vocab_fully_shard_parallel_size *
    # vocab_parallel_size must equal world size. When
    # vocab_fully_shard_parallel_size is None it is auto-computed as
    # world_size // (vocab_parallel_size * vocab_data_parallel_size).
    vocab_parallel_size: int = 1
    vocab_fully_shard_parallel_size: Optional[int] = None
    vocab_data_parallel_size: int = 1
    vp_plan: VPPlanConfig = None
    # Give independent parallel dimensions separate communication groups.
    split_com_groups: bool = False
    kda_p2p_parallel_size: int = 1


@dataclass
class ChunkBatchPlanConfig:
    """Configuration for chunked micro-batch execution.

    The feature wraps selected module forwards and executes them with smaller
    slices along the configured batch dimension. This is useful when a large
    module's activation or temporary workspace peak is proportional to batch
    size.

    Example:
        ``chunk_mbs=1`` with an input tensor shaped ``[4, 2048, hidden]`` runs
        the selected forward four times with tensors shaped ``[1, 2048, hidden]``
        and then concatenates the outputs back to ``[4, 2048, hidden]``.

        For Qwen-style decoder layers, a typical plan is:
            apply_modules=["model.layers.{*}"]
            batch_dim=0
            chunk_arg_indexes=[0]
            chunk_kwarg_names=["hidden_states", "attention_mask"]
    """

    chunk_mbs: int = 1
    # Module-name patterns to patch, for example "model.layers.{*}".
    apply_modules: List[str] = None
    # Tensor dimension that represents batch. HF decoder layers usually use 0.
    batch_dim: int = 0
    # Positional forward arguments that should be sliced by batch.
    chunk_arg_indexes: Optional[List[int]] = None
    # Keyword forward arguments that should be sliced by batch.
    chunk_kwarg_names: Optional[List[str]] = None


@dataclass
class SwapActivationPlanConfig:
    """Configure which modules offload activations.

    Example: ``apply_modules=["model.layers.{*}"]`` applies to all decoder layers.
    Matching rules are the same as recompute_plan, using module_name_match.
    """

    apply_modules: List[str] = None


@dataclass
class MemoryConfig:
    """Memory-saving feature switches.

    ``recompute`` reduces saved activations by re-running module forwards during
    backward. ``chunk_batch`` reduces per-module forward/backward peaks by
    splitting the batch dimension into smaller micro batches.
    ``swap_activation`` offloads saved activations to host memory and fetches
    them back during backward, trading PCIe bandwidth for device memory.
    """

    recompute: bool = False
    recompute_plan: Union[List[str], RecomputePlanConfig] = None
    op_replay_scopes: List[OpReplayScopeConfig] = field(default_factory=list)
    swap_plan: SwapPlanConfig = field(default_factory=SwapPlanConfig)
    chunk_batch: bool = False
    chunk_batch_plan: Optional[ChunkBatchPlanConfig] = None
    chunk_loss: bool = False
    chunk_loss_size: int = 512
    # Multimodal image-text merge: move the text embeddings in blocks of this
    # many tokens instead of one full-sequence advanced-index read; 0 keeps the
    # single assignment. Takes effect only where an example wires the attribute
    # onto the model config.
    mm_merge_chunk_tokens: int = 0
    swap_activation: bool = False
    swap_activation_plan: Optional[SwapActivationPlanConfig] = None


@dataclass
class QuantizationConfig:
    quantization_plan: Optional[QuantizeConfig] = None


@dataclass
class FSDPTurboConfig:
    """Training configuration normalized and validated on construction."""

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: TrainRunConfig = field(default_factory=TrainRunConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    module_patches: List[Dict[str, Any]] = field(default_factory=list)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    def __post_init__(self):
        if not isinstance(self.distributed.split_com_groups, bool):
            raise ValueError("distributed.split_com_groups must be a boolean.")
        self.validate_model_config()
        self.validate_indexer_loss_config()
        self.validate_data_config()
        self.validate_tp_config()
        self.validate_swap_config()
        self.validate_recompute_config()
        self.validate_chunk_loss_config()
        self.validate_mm_merge_chunk_tokens_config()
        self.validate_cp_config()
        self.validate_module_patches()
        self.validate_ep_config()
        self.validate_engram_config()
        self.validate_chunk_batch_config()
        self.validate_swap_activation_config()
        self.validate_quantization_config()
        self.validate_moonep_config()
        self.validate_fsdp_config()
        self.validate_vp_config()
        self.validate_memory_snapshot_config()
        self.validate_profile_config()
        self.validate_optimizer_config()

    def validate_optimizer_config(self):
        """Validate swap-optimizer (AdamWSwap) specific fields."""
        chunk_numel = self.optimizer.swap_chunk_numel
        if isinstance(chunk_numel, bool) or (
            chunk_numel is not None and (not isinstance(chunk_numel, int) or chunk_numel <= 0)
        ):
            raise ValueError("optimizer.swap_chunk_numel must be a positive integer when provided.")
        pattern = self.optimizer.prefetch_states_at_backward
        if pattern is not None and (not isinstance(pattern, str) or not pattern.strip()):
            raise ValueError("optimizer.prefetch_states_at_backward must be a non-empty string when provided.")
        ratio = self.optimizer.resident_states_ratio
        if ratio is not None and (
            isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0.0 <= ratio <= 1.0
        ):
            raise ValueError("optimizer.resident_states_ratio must be a number within [0.0, 1.0] when provided.")
        fsdp_plan = self.distributed.fsdp_plan if self.distributed is not None else None
        if ratio is not None and ratio > 0 and fsdp_plan is not None and fsdp_plan.cpu_offload:
            raise ValueError(
                "optimizer.resident_states_ratio cannot be combined with fsdp_plan.cpu_offload: "
                "resident states assume params live on device. Disable cpu_offload or set "
                "resident_states_ratio to null."
            )
        ratio = self.optimizer.swap_prefetch_budget_ratio
        if ratio is not None and (
            isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0.0 < ratio <= 1.0
        ):
            raise ValueError("optimizer.swap_prefetch_budget_ratio must be a number within (0.0, 1.0] when provided.")
        batches = self.optimizer.swap_prefetch_batches
        if isinstance(batches, bool) or (batches is not None and (not isinstance(batches, int) or batches <= 0)):
            raise ValueError("optimizer.swap_prefetch_batches must be a positive integer when provided.")

    def validate_chunk_loss_config(self) -> None:
        """Validate chunked loss and reject recompute or swap hooks on its head."""
        if not isinstance(self.memory.chunk_loss, bool):
            raise ValueError("memory.chunk_loss must be a boolean.")
        chunk_size = self.memory.chunk_loss_size
        if chunk_size is None and not self.memory.chunk_loss:
            return
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("memory.chunk_loss_size must be a positive integer.")
        if not self.memory.chunk_loss:
            return
        # Context parallelism is supported: Kimi K3 routes the chunked loss
        # through cp_chunk_lm_loss (global label shift + cross-rank
        # normalization). A model whose chunk-loss path cannot run under CP must
        # reject the combination in its own wiring.
        recompute_plan = self.memory.recompute_plan if self.memory.recompute else []
        recompute_patterns = (
            recompute_plan.apply_modules if isinstance(recompute_plan, RecomputePlanConfig) else recompute_plan
        )

        swap_plan = self.memory.swap_activation_plan
        swap_patterns = (swap_plan.apply_modules or []) if self.memory.swap_activation and swap_plan else []
        if any(
            module_name_match(pattern, name)
            for pattern in [*recompute_patterns, *swap_patterns]
            for name in ("", "model.head")
        ):
            raise ValueError("memory.chunk_loss cannot wrap the head or root in recompute/swap_activation hooks.")

    def validate_mm_merge_chunk_tokens_config(self):
        mm_merge_chunk_tokens = self.memory.mm_merge_chunk_tokens
        if (
            isinstance(mm_merge_chunk_tokens, bool)
            or not isinstance(mm_merge_chunk_tokens, int)
            or mm_merge_chunk_tokens < 0
        ):
            raise ValueError("memory.mm_merge_chunk_tokens must be a non-negative integer.")

    def validate_model_config(self):
        """Validate model loading configuration."""
        if not isinstance(self.model.tilekernels, dict):
            raise ValueError("model.tilekernels must be a dict / YAML mapping.")
        for name, enabled in self.model.tilekernels.items():
            if name != "tilekernels_mhc_post":
                raise ValueError(f"Unknown model.tilekernels key '{name}'; use tilekernels_mhc_post: true/false.")
            if not isinstance(enabled, bool):
                raise ValueError(f"model.tilekernels.{name} must be a boolean.")
        if self.model.config_path is not None and (
            not isinstance(self.model.config_path, str) or not self.model.config_path.strip()
        ):
            raise ValueError("model.config_path must be a non-empty string when provided.")
        if not isinstance(self.model.skip_weight_load, bool):
            raise ValueError("model.skip_weight_load must be a boolean.")
        if not isinstance(self.model.use_fused_clipped_swiglu, bool):
            raise ValueError("model.use_fused_clipped_swiglu must be a boolean.")
        if not isinstance(self.model.use_triton_inplace_rope, bool):
            raise ValueError("model.use_triton_inplace_rope must be a boolean.")
        if not isinstance(self.model.use_deepgemm_wo_a, bool):
            raise ValueError("model.use_deepgemm_wo_a must be a boolean.")
        if not isinstance(self.model.use_fused_partial_rotary, bool):
            raise ValueError("model.use_fused_partial_rotary must be a boolean.")
        if self.model.head_compute_dtype not in ("fp32", "bf16", "hf32"):
            raise ValueError("model.head_compute_dtype must be 'fp32', 'bf16' or 'hf32'.")
        if not isinstance(self.model.attn_implementation, str) or not self.model.attn_implementation:
            raise ValueError("model.attn_implementation must be a non-empty string.")
        if self.model.load_pretrained_weights and not self.model.model_name_or_path:
            raise ValueError("model.model_name_or_path is required when load_pretrained_weights is enabled.")
        if self.model.pretrained_engram_rows_per_chunk <= 0:
            raise ValueError("model.pretrained_engram_rows_per_chunk must be positive.")

    def validate_indexer_loss_config(self) -> None:
        """Validate the indexer backend selection and auxiliary gradient weight."""
        if not isinstance(self.model.use_fused_lightning_indexer_loss, bool):
            raise ValueError("model.use_fused_lightning_indexer_loss must be a boolean.")
        coefficient = self.model.indexer_loss_coeff
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, (int, float))
            or not math.isfinite(coefficient)
            or coefficient < 0
        ):
            raise ValueError("model.indexer_loss_coeff must be a finite non-negative number.")

        if not isinstance(self.model.track_indexer_kl_loss, bool):
            raise ValueError("model.track_indexer_kl_loss must be a boolean.")
        if not isinstance(self.model.use_fused_indexer_kl_loss, bool):
            raise ValueError("model.use_fused_indexer_kl_loss must be a boolean.")
        if self.model.track_indexer_kl_loss and coefficient == 0:
            raise ValueError("track_indexer_kl_loss requires a positive indexer_loss_coeff.")

    def validate_data_config(self):
        """Validate and normalize dataloader configuration."""
        if self.data.max_seq_length <= 0:
            raise ValueError("data.max_seq_length must be positive.")
        supported_media_keys = {"images", "medias"}
        if self.data.media_key not in supported_media_keys:
            raise ValueError(
                f"data.media_key must be one of {sorted(supported_media_keys)}, got {self.data.media_key!r}."
            )
        if self.data.sfa_layout not in (None, "BSND", "TND"):
            raise ValueError("data.sfa_layout must be BSND, TND, or null.")
        if self.data.is_multimodal:
            if not isinstance(self.data.image_column, str) or not self.data.image_column:
                raise ValueError("data.image_column must be a non-empty string for multimodal data.")
            if not isinstance(self.data.conversation_column, str) or not self.data.conversation_column:
                raise ValueError("data.conversation_column must be a non-empty string for multimodal data.")

    def validate_fsdp_config(self) -> None:
        """Initialize the FSDP plan and validate its prefetch pattern order.

        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                fsdp_plan=FSDPPlanConfig(
                    'ignored_modules':['*mlp.experts*'],
                    'apply_modules': {
                        'model.layers.*': {reshard_after_forward=None, shard_placement_fn=None}
                    }
                )
            )
        )
        """
        self.distributed.fsdp_plan = (
            FSDPPlanConfig() if self.distributed.fsdp_plan is None else self.distributed.fsdp_plan
        )
        fsdp_plan = self.distributed.fsdp_plan
        prefetch_modules = fsdp_plan.prefetch_modules
        if prefetch_modules is not None:
            if not isinstance(prefetch_modules, list) or any(
                not isinstance(pattern, str) for pattern in prefetch_modules
            ):
                raise ValueError("distributed.fsdp_plan.prefetch_modules must be a list of apply_modules patterns.")
            if sorted(prefetch_modules) != sorted(fsdp_plan.apply_modules or {}):
                raise ValueError(
                    "distributed.fsdp_plan.prefetch_modules must contain every apply_modules pattern exactly once."
                )
        if fsdp_plan.backward_prefetch_on_moe_input:
            if fsdp_plan.fsdp_implementation != "native":
                raise ValueError(
                    "distributed.fsdp_plan.backward_prefetch_on_moe_input requires fsdp_implementation='native'."
                )
            if self.distributed.expert_parallel_size <= 1:
                raise ValueError(
                    "distributed.fsdp_plan.backward_prefetch_on_moe_input requires expert_parallel_size > 1 "
                    "(no expert a2a to overlap with when experts are not EP-sharded)."
                )
            if fsdp_plan.num_to_backward_prefetch <= 0:
                raise ValueError(
                    "distributed.fsdp_plan.backward_prefetch_on_moe_input requires num_to_backward_prefetch > 0."
                )
            if not self.distributed.ep_plan.apply_efsdp_modules:
                raise ValueError(
                    "distributed.fsdp_plan.backward_prefetch_on_moe_input "
                    "requires ep_plan.apply_efsdp_modules to wrap experts."
                )
        if fsdp_plan.prefetch_wait_a2a:
            if fsdp_plan.num_to_forward_prefetch <= 0 and not fsdp_plan.backward_prefetch_on_moe_input:
                raise ValueError(
                    "distributed.fsdp_plan.prefetch_wait_a2a requires num_to_forward_prefetch > 0 "
                    "or backward_prefetch_on_moe_input."
                )
            if self.distributed.expert_parallel_size <= 1:
                raise ValueError(
                    "distributed.fsdp_plan.prefetch_wait_a2a requires expert_parallel_size > 1 "
                    "(no expert a2a is issued with EP=1)."
                )
            if not self.distributed.ep_plan.apply_efsdp_modules:
                raise ValueError(
                    "distributed.fsdp_plan.prefetch_wait_a2a requires ep_plan.apply_efsdp_modules for a2a event sources."
                )
        if self.distributed.fully_shard_parallel_size > 1:
            if self.distributed.expert_parallel_size > 1:
                self.distributed.fsdp_plan.ignored_modules.extend(self.distributed.ep_plan.apply_modules)

            self.distributed.fsdp_plan.ignored_modules = list(
                set(self.distributed.fsdp_plan.ignored_modules)
            )  # remove duplicates

    def validate_profile_config(self):
        """Validate the run.profile collection section."""
        ranks = self.run.profile.ranks
        if ranks is not None and (
            not isinstance(ranks, (list, tuple))
            or any(not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 for rank in ranks)
        ):
            raise ValueError("run.profile.ranks must be a list of non-negative integers or null.")

    def validate_memory_snapshot_config(self):
        """Validate the run.memory_snapshot collection section."""
        profile = self.run.memory_snapshot
        if profile.start_step < 0:
            raise ValueError(f"run.memory_snapshot.start_step must be >= 0; got {profile.start_step}.")
        if profile.end_step is not None:
            if profile.end_step < profile.start_step:
                raise ValueError(
                    "run.memory_snapshot.end_step must be >= start_step "
                    f"(the window is inclusive on both ends); got start_step={profile.start_step}, "
                    f"end_step={profile.end_step}."
                )
        if profile.save_path is not None:
            if not isinstance(profile.save_path, str) or not profile.save_path.strip():
                raise ValueError("run.memory_snapshot.save_path must be a non-empty string or null.")
        if profile.dump_ranks is not None:
            if not isinstance(profile.dump_ranks, (list, tuple)) or any(
                not isinstance(rank, int) or rank < 0 for rank in profile.dump_ranks
            ):
                raise ValueError("run.memory_snapshot.dump_ranks must be a list of non-negative integers or null.")
        stacks = str(profile.stacks).strip().lower()
        if stacks not in ("all", "python"):
            raise ValueError(f"run.memory_snapshot.stacks must be 'all' or 'python'; got {profile.stacks!r}.")
        profile.stacks = stacks
        if profile.max_entries is not None and profile.max_entries < 1:
            raise ValueError(f"run.memory_snapshot.max_entries must be >= 1 or null; got {profile.max_entries}.")

    def validate_tp_config(self):
        """tensor parallelize plan

        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                tp_plan=TPPlanConfig(
                    colwise_parallel=['*.q_proj', '*.k_proj', '*.v_proj'],
                    rowwise_parallel=['*.o_proj']
                )
            )
        )
        """
        self.distributed.tp_plan = TPPlanConfig() if self.distributed.tp_plan is None else self.distributed.tp_plan
        self.distributed.tp_plan.colwise_parallel = (
            [] if self.distributed.tp_plan.colwise_parallel is None else self.distributed.tp_plan.colwise_parallel
        )
        self.distributed.tp_plan.rowwise_parallel = (
            [] if self.distributed.tp_plan.rowwise_parallel is None else self.distributed.tp_plan.rowwise_parallel
        )
        self.distributed.tp_plan.sequence_parallel = (
            [] if self.distributed.tp_plan.sequence_parallel is None else self.distributed.tp_plan.sequence_parallel
        )

    def _validate_tnd_cp_config(self) -> None:
        """Derive and validate packed-TND mode from the input layout.

        Packed TND is a property of the batch, not of a patch list: one packed
        stream is cut across the KV-AllGather ranks and each rank assembles the
        prefix of every document it touches. It therefore follows
        ``data.sfa_layout`` and never combines with the native BSND path.
        """
        plan = self.distributed.cp_plan
        if plan.tnd_exchange_style not in ("full_prefix", "halo_prefix"):
            raise ValueError(
                "distributed.cp_plan.tnd_exchange_style must be full_prefix or halo_prefix; "
                f"got {plan.tnd_exchange_style!r}."
            )
        plan.tnd_enabled = self.data.sfa_layout == "TND"
        # The layout alone selects packed TND, including the zero-split arm.
        if not plan.tnd_enabled:
            return
        if self.data.batch_size != 1:
            raise ValueError(
                f"packed-TND CP requires micro batch size 1 (data.batch_size == 1); got {self.data.batch_size}."
            )
        if self.distributed.ulysses_parallel_size != 1 or self.distributed.kcp_parallel_size != 1:
            raise ValueError(
                "packed-TND CP owns the whole context-parallel domain; set "
                "distributed.ulysses_parallel_size=1 and kcp_parallel_size=1."
            )
        if plan.kv_allgather_function_patches:
            raise ValueError("packed-TND CP is native; it cannot be combined with KV-AllGather function patches.")
        if self.memory.chunk_loss:
            # Chunked head normalises locally; packed-TND normalises over the whole stream.
            raise ValueError("packed-TND CP normalises the loss over the whole stream; set memory.chunk_loss=false.")
        if self.memory.recompute:
            # Replay would read a state slot that later source layers already overwrote.
            raise ValueError(
                "packed-TND CP does not support memory.recompute: a replay reads the KV source "
                "state that later source layers have already overwritten. Set memory.recompute=false."
            )

    def validate_cp_config(self) -> None:
        """Validate native CP or normalize function patches and parallel constraints."""
        kcp_parallel_size = self.distributed.kcp_parallel_size
        kda_p2p_parallel_size = self.distributed.kda_p2p_parallel_size
        ulysses_parallel_size = self.distributed.ulysses_parallel_size
        for field_name, size in (
            ("kv_allgather_parallel_size", self.distributed.kv_allgather_parallel_size),
            ("kcp_parallel_size", kcp_parallel_size),
            ("kda_p2p_parallel_size", kda_p2p_parallel_size),
            ("ulysses_parallel_size", ulysses_parallel_size),
        ):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"distributed.{field_name} must be a positive integer; got {size!r}.")

        self.distributed.cp_plan = CPPlanConfig() if self.distributed.cp_plan is None else self.distributed.cp_plan
        # Snapshot before normalization erases the unset/explicit-empty distinction:
        # an explicit empty ulysses_function_patches opts out for models whose
        # Ulysses forwards are built into the modeling classes.
        ulysses_patches_specified = self.distributed.cp_plan.ulysses_function_patches is not None
        if self.distributed.cp_plan.qsa_cp_query_chunk_size is not None and (
            isinstance(self.distributed.cp_plan.qsa_cp_query_chunk_size, bool)
            or not isinstance(self.distributed.cp_plan.qsa_cp_query_chunk_size, int)
            or self.distributed.cp_plan.qsa_cp_query_chunk_size <= 0
        ):
            raise ValueError("distributed.cp_plan.qsa_cp_query_chunk_size must be a positive integer when provided.")
        if self.distributed.cp_plan.qsa_sparse_gather is not None and not isinstance(
            self.distributed.cp_plan.qsa_sparse_gather, bool
        ):
            raise ValueError("distributed.cp_plan.qsa_sparse_gather must be a boolean when provided.")
        for field_name in (
            "ulysses_function_patches",
            "kcp_function_patches",
            "kv_allgather_function_patches",
            "loss_function_patches",
        ):
            patches = getattr(self.distributed.cp_plan, field_name)
            if patches is None:
                patches = []
            elif isinstance(patches, dict):
                patches = [patches]
            elif not isinstance(patches, list):
                raise ValueError(f"distributed.cp_plan.{field_name} must be a mapping or list of mappings.")

            normalized = []
            for patch in patches:
                if not isinstance(patch, dict):
                    raise ValueError(f"distributed.cp_plan.{field_name} entries must be mappings.")
                targets = patch.get("target_functions")
                if isinstance(targets, str):
                    targets = [targets]
                if not isinstance(targets, list) or not targets:
                    raise ValueError(f"distributed.cp_plan.{field_name} entries must set target_functions.")
                if any(not isinstance(target, str) or ":" in target or "." not in target for target in targets):
                    raise ValueError(
                        f"distributed.cp_plan.{field_name} target_functions must use dotted callable paths."
                    )
                patch_type = patch.get("type")
                if not isinstance(patch_type, str) or not patch_type:
                    raise ValueError(f"distributed.cp_plan.{field_name} entries must set type.")
                normalized_patch = {**patch, "target_functions": targets}
                if patch_type == "kda" and field_name not in (
                    "ulysses_function_patches",
                    "kcp_function_patches",
                ):
                    raise ValueError(
                        "KDA context-parallel patches must use ulysses_function_patches or kcp_function_patches."
                    )
                if patch_type == "compressor_sparse_attention":
                    raise ValueError(
                        "DeepSeek-V4.1 CP uses the model's native interface; remove compressor_sparse_attention patches."
                    )
                if patch_type == "kv_allgather_tnd":
                    raise ValueError(
                        "kv_allgather_tnd no longer uses a function patch: set "
                        "distributed.kv_allgather_parallel_size > 1 with data.sfa_layout='TND'."
                    )
                normalized.append(normalized_patch)
            setattr(self.distributed.cp_plan, field_name, normalized)

        deepseek_compressor_specs = [
            spec
            for axis in ("kv_allgather", "ulysses", "kcp")
            for spec in getattr(self.distributed.cp_plan, f"{axis}_function_patches")
            if spec["type"] in {"compressor_full_attention", "compressor_full_attention_transposed"}
        ]
        if deepseek_compressor_specs and (kcp_parallel_size > 1 or kda_p2p_parallel_size > 1):
            raise ValueError(
                "DeepSeek-V4 compressor attention CP is not supported on the KCP axis or the KDA P2P axis."
            )
        if deepseek_compressor_specs and self.distributed.kv_allgather_parallel_size > 1 and ulysses_parallel_size > 1:
            raise ValueError(
                "DeepSeek-V4 compressor CP supports only one sequence-parallel axis at a time: "
                "kv_allgather_parallel_size and ulysses_parallel_size cannot both be greater than 1."
            )

        # Qwen QSA/GDN forwards carry their own KV-AllGather implementation;
        # they are installed through the Ulysses class-forward patches below,
        # so they are not the native-model API path handled by this block.
        kv_embedded_types = {"qsa", "gated_delta_net"}
        ulysses_types = {patch["type"] for patch in self.distributed.cp_plan.ulysses_function_patches}
        qwen_embedded_cp = bool(ulysses_types) and ulysses_types <= kv_embedded_types

        self._validate_tnd_cp_config()

        # An enabled KV axis without wrappers requires the model's native API.
        # The CP entry verifies that API when the model is available.
        if (
            self.distributed.kv_allgather_parallel_size > 1
            and not self.distributed.cp_plan.kv_allgather_function_patches
            and not qwen_embedded_cp
            and not self.distributed.cp_plan.tnd_enabled
        ):
            if any(
                (
                    self.distributed.cp_plan.ulysses_function_patches,
                    self.distributed.cp_plan.kcp_function_patches,
                    self.distributed.cp_plan.loss_function_patches,
                    self.module_patches,
                )
            ):
                raise ValueError(
                    "distributed.kv_allgather_parallel_size > 1: "
                    "Native KV-AllGather CP without kv_allgather_function_patches "
                    "cannot be combined with function or module patches."
                )
            if any(
                size != 1
                for size in (
                    ulysses_parallel_size,
                    kcp_parallel_size,
                    kda_p2p_parallel_size,
                    self.distributed.tensor_parallel_size,
                )
            ):
                raise ValueError("DeepSeek-V4.1 KV-AllGather requires Ulysses=KCP=TP=1 and kda_p2p_parallel_size=1.")
            if self.memory.chunk_batch:
                raise ValueError("DeepSeek-V4.1 CP does not support chunk_batch.")
            if self.data.sfa_layout == "TND":
                raise ValueError("DeepSeek-V4.1 CP currently supports BSND input only.")
            if any(
                (
                    self.model.use_sparse_flash_attn,
                    self.model.use_fused_lightning_indexer,
                    self.model.use_fused_compressor,
                    self.model.use_fused_mhc,
                    self.model.tilekernels.get("tilekernels_mhc_post", False),
                    self.model.use_fused_clipped_swiglu,
                    self.model.use_triton_inplace_rope,
                    self.model.use_deepgemm_wo_a,
                    self.model.use_fused_vision_flash_attn,
                    self.model.use_fused_partial_rotary,
                )
            ):
                raise ValueError("DeepSeek-V4.1 CP requires all fused flags and Triton RoPE disabled.")

        self._validate_kda_cp_config()
        plan = self.distributed.cp_plan
        hybrid_cp_enabled = plan.kda_backend == "hybrid_p2p"
        if (
            not hybrid_cp_enabled
            and kcp_parallel_size == 1
            and ulysses_parallel_size > 1
            and not plan.ulysses_function_patches
            and not ulysses_patches_specified
        ):
            raise ValueError(
                "distributed.cp_plan.ulysses_function_patches must be set when Ulysses is enabled; "
                "set it to [] only when the model's Ulysses forwards are built into the modeling classes."
            )
        # Qwen QSA and GDN forwards embed their own KV-AllGather handling.
        # They must not be combined with a generic KV patch. Patch-free native
        # KV-AllGather and incompatible patch combinations are validated above.
        if self.distributed.kv_allgather_parallel_size > 1:
            generic_kv_patches = self.distributed.cp_plan.kv_allgather_function_patches
            if generic_kv_patches and ulysses_types & kv_embedded_types:
                raise ValueError(
                    "Generic kv_allgather_function_patches cannot be combined with the "
                    "KV-aware Ulysses patch types (qsa, gated_delta_net)."
                )

        context_parallel_active = (
            kcp_parallel_size > 1
            or kda_p2p_parallel_size > 1
            or ulysses_parallel_size > 1
            or self.distributed.kv_allgather_parallel_size > 1
        )
        if self.memory.chunk_batch and (context_parallel_active or hybrid_cp_enabled):
            if hybrid_cp_enabled:
                raise ValueError("Hybrid P2P does not support memory.chunk_batch.")
            raise ValueError(
                "memory.chunk_batch cannot be combined with context parallelism "
                "(kcp_parallel_size/kda_p2p_parallel_size/ulysses_parallel_size/kv_allgather_parallel_size > 1): "
                "both features take over module forwards on the same sequence and the "
                "combination is undefined."
            )

    def _validate_kda_cp_config(self) -> None:
        """Normalize KDA/MLA backends and validate state-CP combinations.

        Called after common CP axes and function patches have been normalized.
        Generic Ulysses patch and chunk-batch checks stay in validate_cp_config.
        """
        kcp_parallel_size = self.distributed.kcp_parallel_size
        kda_p2p_parallel_size = self.distributed.kda_p2p_parallel_size
        ulysses_parallel_size = self.distributed.ulysses_parallel_size
        plan = self.distributed.cp_plan
        plan.mla_backend = "ulysses" if plan.mla_backend is None else plan.mla_backend
        if plan.mla_backend not in ("ulysses", "kv_allgather"):
            raise ValueError(
                "distributed.cp_plan.mla_backend must be ulysses or kv_allgather; ring is not implemented."
            )

        hybrid_cp_enabled = plan.kda_backend == "hybrid_p2p"
        if hybrid_cp_enabled:
            if kcp_parallel_size != 1:
                raise ValueError(
                    "Hybrid P2P requires distributed.kcp_parallel_size == 1; "
                    "set distributed.kda_p2p_parallel_size for its P2P axis."
                )
            # Only the supported Hybrid P2P combinations differ from legacy CP.
            if self.memory.chunk_loss:
                raise ValueError("Hybrid P2P does not support memory.chunk_loss; use the legacy Kimi K3 CP path.")
            if self.module_patches:
                raise ValueError("Hybrid P2P uses model subclasses; remove legacy module_patches.")
            if any(size & (size - 1) for size in (ulysses_parallel_size, kda_p2p_parallel_size)):
                raise ValueError("Hybrid P2P axis sizes must be powers of two (including 1).")
        else:
            if kda_p2p_parallel_size != 1:
                raise ValueError("distributed.kda_p2p_parallel_size > 1 requires kda_backend=hybrid_p2p.")
            if plan.mla_backend != "ulysses":
                raise ValueError("mla_backend=kv_allgather requires kda_backend=hybrid_p2p.")
            if ulysses_parallel_size > 1 and kcp_parallel_size > 1:
                raise ValueError("2D Ulysses-KCP AllGather is retired; select kda_backend=hybrid_p2p.")
            if kcp_parallel_size == 1 and plan.kda_backend != "kcp":
                raise ValueError("distributed.cp_plan.kda_backend requires kcp_parallel_size > 1.")
            if kcp_parallel_size > 1:
                from fsdp_turbo.ops.kcp import get_kcp_kda_backend

                get_kcp_kda_backend(plan.kda_backend)
                if not plan.kcp_function_patches:
                    raise ValueError("distributed.cp_plan.kcp_function_patches must be set when KCP is enabled.")

        # Reuse the existing KDA state-CP limits for both execution paths.
        if kcp_parallel_size > 1 or hybrid_cp_enabled:
            for requirement, size in (
                ("distributed.tensor_parallel_size == 1", self.distributed.tensor_parallel_size),
                ("distributed.kv_allgather_parallel_size == 1", self.distributed.kv_allgather_parallel_size),
                ("micro batch size 1 (data.batch_size == 1)", self.data.batch_size),
            ):
                if size != 1:
                    raise ValueError(f"KDA state context parallelism requires {requirement}; got {size}.")

    def validate_ep_config(self) -> None:
        """expert parallelize plan

        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                ep_plan=EPPlanConfig(
                    apply_modules: ['*mlp.experts*'],
                    dispatcher: 'all2all', 'mc2', 'domino', 'allgather', 'chunkmoe', 'moonep',
                                'custom_ep_forward_eager', 'custom_ep_forward_fused'
                )
            )
        )
        """
        self.distributed.ep_plan = (
            EPPlanConfig(apply_modules=[], dispatcher="all2all")
            if self.distributed.ep_plan is None
            else self.distributed.ep_plan
        )
        if self.distributed.ep_plan.gradient_divide_factor is None:
            # Divide across all expert mesh dims (edp * efsdp * ep). torchrun
            # sets WORLD_SIZE; fall back to 1 for non-distributed runs.
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            edp = world_size // (
                self.distributed.expert_fully_shard_parallel_size * self.distributed.expert_parallel_size
            )
            self.distributed.ep_plan.gradient_divide_factor = (
                self.distributed.expert_parallel_size * self.distributed.expert_fully_shard_parallel_size * edp
            )
        if not self.distributed.ep_plan.apply_efsdp_modules:
            self.distributed.ep_plan.apply_efsdp_modules = []
            # EP=1 never calls expert_fully_shard_modules, so auto-filled
            # selections would put unwrapped units into prefetch lists.
            if self.distributed.ep_plan.dispatcher != "moonep" and self.distributed.expert_parallel_size > 1:
                self.distributed.ep_plan.apply_efsdp_modules = [
                    ep_module for ep_module in self.distributed.ep_plan.apply_modules if ep_module.endswith(".experts")
                ]
        elif self.distributed.expert_parallel_size <= 1:
            # Expert FSDP only wraps experts into a separate expert mesh; with
            # EP=1 experts stay inside their layer's FSDP unit, so the explicit
            # selection never takes effect. Keep the value for compatibility
            # and warn instead of raising.
            print_rank(
                logger.warning,
                "distributed.ep_plan.apply_efsdp_modules has no effect with "
                "expert_parallel_size <= 1 (experts stay inside their layer's "
                "FSDP unit and are never wrapped).",
            )

    def validate_moonep_config(self) -> None:
        ep_plan = self.distributed.ep_plan
        if ep_plan is None or ep_plan.dispatcher != "moonep":
            return
        if self.distributed.expert_parallel_size <= 1:
            raise ValueError(
                "MoonEP requires expert_parallel_size > 1; use the model's local expert path when EP is disabled."
            )
        if self.distributed.expert_fully_shard_parallel_size != 1:
            raise ValueError(
                "MoonEP does not support expert FSDP; set expert_fully_shard_parallel_size=1 or use another dispatcher."
            )
        if ep_plan.apply_efsdp_modules:
            raise ValueError(
                "MoonEP owns the expert weights, so they cannot also be wrapped "
                "by expert FSDP. Leave ep_plan.apply_efsdp_modules empty or use "
                f"another dispatcher; got {ep_plan.apply_efsdp_modules}."
            )
        moon_cfg = ep_plan.moonep_config
        if moon_cfg is None:
            moon_cfg = MoonEPConfig()
            ep_plan.moonep_config = moon_cfg
        if moon_cfg.num_sms <= 0:
            raise ValueError("MoonEP num_sms must be positive.")
        if moon_cfg.token_padding <= 0:
            raise ValueError("MoonEP token_padding must be positive.")
        if moon_cfg.tokens_per_rank is not None and moon_cfg.tokens_per_rank <= 0:
            raise ValueError("MoonEP tokens_per_rank must be positive when set.")
        if moon_cfg.top_k is not None and moon_cfg.top_k <= 0:
            raise ValueError("MoonEP top_k must be positive when set.")

        quant_plan = self.quantization.quantization_plan
        if quant_plan is not None and quant_plan.quant_recipe:
            moe_converters = [name for name in (quant_plan.converters or []) if name.startswith("quantize.moe.")]
            if moe_converters or quant_plan.quant_gmm:
                offending = ", ".join(moe_converters) or "quant_gmm"
                raise ValueError(
                    "MoonEP requires BF16 expert weights and is incompatible with MoE "
                    f"quantization. Remove {offending} or use another dispatcher."
                )

    def validate_vp_config(self):
        """vocab parallelize plan

        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                vp_plan=VPPlanConfig(
                    apply_modules: ['model.language_model.layers.{*}.ple.ple_embedding.ngram_embedding']
                )
            )
        )
        """
        vp_plan = self.distributed.vp_plan
        if vp_plan is None:
            return
        if self.distributed.vocab_parallel_size <= 1:
            raise ValueError(
                "distributed.vp_plan requires distributed.vocab_parallel_size > 1; "
                "remove vp_plan to use the frozen CPU-offload path."
            )
        if not vp_plan.apply_modules:
            raise ValueError("distributed.vp_plan.apply_modules must not be empty when vp_plan is set.")
        if vp_plan.apply_vfsdp_modules is None:
            vp_plan.apply_vfsdp_modules = vp_plan.apply_modules
        if vp_plan.gradient_divide_factor is None:
            # Match the FSDP gradient averaging applied to dense parameters:
            # the raw VP table gradient is a sum over all vp ranks' tokens.
            # torchrun sets WORLD_SIZE; fall back to 1 for non-distributed runs.
            vp_plan.gradient_divide_factor = float(int(os.environ.get("WORLD_SIZE", "1")))
        if vp_plan.gradient_divide_factor <= 0:
            raise ValueError("distributed.vp_plan.gradient_divide_factor must be positive.")
        if self.distributed.vocab_fully_shard_parallel_size is None:
            # torchrun sets WORLD_SIZE; fall back to 1 for non-distributed runs.
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.distributed.vocab_fully_shard_parallel_size = world_size // (
                self.distributed.vocab_data_parallel_size * self.distributed.vocab_parallel_size
            )

        fsdp_plan = self.distributed.fsdp_plan
        # V-FSDP patterns are managed by VP / V-FSDP instead of the FSDP plan
        # (apply_modules is a pattern -> plan dict).
        fsdp_plan.apply_modules = {
            pattern: module_plan
            for pattern, module_plan in (fsdp_plan.apply_modules or {}).items()
            if pattern not in vp_plan.apply_vfsdp_modules
        }
        # With singleton V-FSDP (vdp * vfsdp == 1) the pure-VP parameters must
        # stay outside ancestor FSDP units; otherwise fully_shard takes
        # ownership of them.
        if self.distributed.vocab_data_parallel_size * self.distributed.vocab_fully_shard_parallel_size == 1:
            if fsdp_plan.ignored_modules is None:
                fsdp_plan.ignored_modules = []
            for pattern in vp_plan.apply_vfsdp_modules:
                if pattern not in fsdp_plan.ignored_modules:
                    fsdp_plan.ignored_modules.append(pattern)

    def validate_engram_config(self) -> None:
        """Validate parallel topology required by Host-offload Engram."""
        backend = self.model.engram_storage_backend
        supported_backends = (
            "host_offload",
            "elastic_buffer",
        )
        if backend is None:
            return
        if backend not in supported_backends:
            raise ValueError(f"model.engram_storage_backend must be one of {supported_backends}, got {backend!r}.")
        if self.distributed.expert_parallel_size <= 1:
            raise ValueError("Host-offload Engram requires expert_parallel_size > 1.")
        if self.distributed.kv_allgather_parallel_size != 1 or self.distributed.ulysses_parallel_size != 1:
            if (
                self.distributed.kv_allgather_parallel_size == 1
                or self.distributed.cp_plan.kv_allgather_function_patches
            ):
                raise ValueError("Host-offload Engram CP requires native KV-AllGather CP without function patches.")

    def validate_module_patches(self):
        """Validate the namespace-neutral module patch configuration.

        Each entry describes exactly one dotted ``target`` and one callable
        ``replacement``. Callable objects are accepted for direct Python API
        use; YAML normally supplies dotted replacement paths.
        """
        if self.module_patches is None:
            self.module_patches = []
        elif isinstance(self.module_patches, dict):
            self.module_patches = [self.module_patches]
        elif not isinstance(self.module_patches, list):
            raise ValueError("module_patches must be a mapping or a list of mappings.")
        for patch in self.module_patches:
            if not isinstance(patch, dict):
                raise ValueError("module_patches entries must be mappings.")
            target = patch.get("target")
            replacement = patch.get("replacement")
            if not isinstance(target, str) or not target:
                raise ValueError("module_patches entries must set target.")
            if replacement is None:
                raise ValueError("module_patches entries must set replacement.")
            target_parts = str(target).split(".")
            if len(target_parts) < 2 or any(not part for part in target_parts):
                raise ValueError("module_patches target must use a dotted module.member or module.Class.member path.")
            if isinstance(replacement, str):
                replacement_parts = replacement.split(".")
                if ":" in replacement:
                    raise ValueError("module_patches replacement must use a dotted callable path, not a colon path.")
                if len(replacement_parts) < 2 or any(not part for part in replacement_parts):
                    raise ValueError("module_patches replacement must use a dotted callable path.")
            elif not callable(replacement):
                raise ValueError("module_patches replacement must be callable or a dotted callable path.")

    def validate_swap_config(self) -> None:
        """Normalize the swap plan and validate its CPU arena settings."""
        if isinstance(self.memory.swap_plan, dict):
            # Direct callers can pass a dict despite the normalized field type.
            self.memory.swap_plan = SwapPlanConfig(**self.memory.swap_plan)  # pylint: disable=not-a-mapping
        swap_plan = self.memory.swap_plan
        if swap_plan.cpu_arena_slab_size_mb is not None and swap_plan.cpu_arena_slab_size_mb <= 0:
            raise ValueError("swap_plan.cpu_arena_slab_size_mb must be positive or null")
        if swap_plan.cpu_arena_pool_policy not in ("all", "standard", "none"):
            raise ValueError("swap_plan.cpu_arena_pool_policy must be all, standard or none")

    def validate_recompute_config(self) -> None:
        """Normalize legacy lists and structured non-reentrant replay plans."""
        plan = self.memory.recompute_plan
        if isinstance(plan, dict):
            plan = dict(plan)
            if plan.pop("use_reentrant", False) is not False:
                raise ValueError("recompute requires use_reentrant=false")
            if "op_replay_scopes" in plan:
                if self.memory.op_replay_scopes:
                    raise ValueError("Define op_replay_scopes in memory or recompute_plan, not both")
                self.memory.op_replay_scopes = plan.pop("op_replay_scopes")
            plan = RecomputePlanConfig(**plan)
        if isinstance(plan, RecomputePlanConfig):
            if not isinstance(plan.swap_inputs, bool):
                raise ValueError("recompute_plan.swap_inputs must be a boolean.")
            if plan.swap_inputs and (not self.memory.recompute or not plan.apply_modules):
                raise ValueError("recompute_plan.swap_inputs requires recompute=true and a non-empty apply_modules.")
        self.memory.recompute_plan = [] if plan is None else plan
        self.memory.op_replay_scopes = [
            OpReplayScopeConfig(**scope) if isinstance(scope, dict) else scope for scope in self.memory.op_replay_scopes
        ]
        for scope in self.memory.op_replay_scopes:
            if not scope.apply_modules or not isinstance(scope.save_rng, bool):
                raise ValueError("Replay scopes require apply_modules and boolean save_rng")
            from fsdp_turbo.memory.op_replay.op_replay import _parse_op_list

            _parse_op_list(scope.cache_ops)
            if scope.quantizer is not None and (
                not isinstance(scope.quantizer, str) or scope.quantizer not in QUANTIZERS
            ):
                raise ValueError(f"Unknown payload quantizer {scope.quantizer!r}; expected one of {sorted(QUANTIZERS)}")
            if (
                not isinstance(scope.fp8_block_size, int)
                or scope.fp8_block_size not in Fp8BlockwiseQuantizer.SUPPORTED_BLOCK_SIZES
            ):
                raise ValueError("fp8_block_size must be 64, 128, 192 or 256")
            if scope.quantizer == Fp8BlockwiseQuantizer.name:
                if not fp8_payload_quant_supported():
                    raise ValueError("fp8_blockwise is only supported on Ascend 950PR&950DT products")
            elif scope.fp8_block_size != Fp8BlockwiseQuantizer.DEFAULT_BLOCK_SIZE:
                raise ValueError("fp8_block_size requires quantizer='fp8_blockwise'")
        if self.memory.op_replay_scopes and not self.memory.recompute:
            raise ValueError("op_replay_scopes requires recompute=true")

    def validate_chunk_batch_config(self):
        """Normalize and validate chunk-batch configuration.

        The selected modules and at least one sliced input are required. Without
        ``apply_modules`` no module would be patched; without arg/kwarg selectors
        the wrapper cannot infer batch size or know which tensors must be sliced.
        """
        if not self.memory.chunk_batch:
            return
        self.memory.chunk_batch_plan = (
            ChunkBatchPlanConfig() if self.memory.chunk_batch_plan is None else self.memory.chunk_batch_plan
        )
        if self.memory.chunk_batch_plan.chunk_mbs <= 0:
            raise ValueError("chunk_mbs must be positive.")
        self.memory.chunk_batch_plan.apply_modules = (
            [] if self.memory.chunk_batch_plan.apply_modules is None else self.memory.chunk_batch_plan.apply_modules
        )
        self.memory.chunk_batch_plan.chunk_arg_indexes = (
            []
            if self.memory.chunk_batch_plan.chunk_arg_indexes is None
            else self.memory.chunk_batch_plan.chunk_arg_indexes
        )
        self.memory.chunk_batch_plan.chunk_kwarg_names = (
            []
            if self.memory.chunk_batch_plan.chunk_kwarg_names is None
            else self.memory.chunk_batch_plan.chunk_kwarg_names
        )
        if not self.memory.chunk_batch_plan.apply_modules:
            raise ValueError("chunk_batch_plan.apply_modules must not be empty when chunk_batch is enabled.")
        if not self.memory.chunk_batch_plan.chunk_arg_indexes and not self.memory.chunk_batch_plan.chunk_kwarg_names:
            raise ValueError(
                "chunk_batch_plan must specify chunk_arg_indexes or chunk_kwarg_names when chunk_batch is enabled."
            )

    def validate_swap_activation_config(self):
        """Normalize and validate swap-activation configuration.

        When the feature is off, ensure the plan is None so downstream code can
        treat a non-None plan as an enabled signal. When on, require a non-empty
        apply_modules list — matching no module would silently disable swap.
        """
        if not self.memory.swap_activation:
            self.memory.swap_activation_plan = None
            return
        self.memory.swap_activation_plan = (
            SwapActivationPlanConfig() if self.memory.swap_activation_plan is None else self.memory.swap_activation_plan
        )
        self.memory.swap_activation_plan.apply_modules = (
            []
            if self.memory.swap_activation_plan.apply_modules is None
            else self.memory.swap_activation_plan.apply_modules
        )
        if not self.memory.swap_activation_plan.apply_modules:
            raise ValueError("swap_activation_plan.apply_modules must not be empty when swap_activation is enabled.")

    def validate_quantization_config(self):
        self.quantization.quantization_plan = (
            QuantizeConfig() if self.quantization.quantization_plan is None else self.quantization.quantization_plan
        )
        self.quantization.quantization_plan.fsdp_world_size = self.distributed.fully_shard_parallel_size
        if not (self.quantization.quantization_plan.converters or self.quantization.quantization_plan.quant_recipe):
            # LP all-gather only applies when quantization is actually enabled.
            self.quantization.quantization_plan.enable_fsdp_low_precision_all_gather = False

    def __str__(self):
        import dataclasses as dc

        def _format_value(v):
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    return "[]"
                if len(v) <= 3:
                    return str(v)
                return f"[{v[0]}, {v[1]}, ... ] (len={len(v)})"
            if isinstance(v, dict):
                if len(v) == 0:
                    return "{}"
                return str(v)
            if isinstance(v, torch.dtype):
                return str(v)
            return repr(v)

        # First pass: collect all leaf fields to determine max name width.
        def _collect_leaves(obj, prefix=""):
            leaves = []
            for f in dc.fields(obj):
                val = getattr(obj, f.name)
                display = f.name if not prefix else f"{prefix}.{f.name}"
                if val is None:
                    leaves.append((display, "None"))
                elif dc.is_dataclass(val) and not isinstance(val, torch.dtype):
                    leaves.extend(_collect_leaves(val, display))
                else:
                    leaves.append((display, _format_value(val)))
            return leaves

        sep = "=" * 60
        # Auto-discover sections from dataclass fields that are themselves dataclasses.
        sections = []
        for f in dc.fields(self):
            val = getattr(self, f.name)
            if dc.is_dataclass(val) and not isinstance(val, torch.dtype):
                sections.append((f.name.capitalize(), val))

        # Collect all leaves across sections to find max name width.
        all_leaves = []
        for section_name, section_obj in sections:
            all_leaves.extend(_collect_leaves(section_obj))
        max_name_len = max(len(name) for name, _ in all_leaves) if all_leaves else 20

        # Second pass: render with aligned values.
        def _format_dataclass(obj, indent=2, prefix=""):
            lines = []
            for f in dc.fields(obj):
                val = getattr(obj, f.name)
                display = f.name if not prefix else f"{prefix}.{f.name}"
                if val is None:
                    val_str = "None"
                elif dc.is_dataclass(val) and not isinstance(val, torch.dtype):
                    lines.extend(_format_dataclass(val, indent, display))
                    continue
                else:
                    val_str = _format_value(val)
                # Pad name + dashes so that value starts at value_col.
                padded_name = " " * indent + display
                total_name_width = indent + max_name_len
                dash_count = max(1, total_name_width - len(padded_name) + 2)
                lines.append(f"{padded_name} {'-' * dash_count} {val_str}")
            return lines

        lines = [sep, f"{'FSDPTurboConfig':^60}", sep]
        for section_name, section_obj in sections:
            lines.append(f"  [{section_name}]")
            lines.extend(_format_dataclass(section_obj, indent=4))
            lines.append("")
        if self.module_patches:
            lines.append("  [Module_patches]")
            lines.append(f"    module_patches ---- {self.module_patches!r}")
            lines.append("")
        lines.append(sep)
        return "\n".join(lines)


def _coerce_value(value, field_type):
    if not isinstance(value, str):
        return value

    origin = getattr(field_type, "__origin__", None)
    if origin is Union:
        args = [a for a in field_type.__args__ if a is not type(None)]  # pylint: disable=unidiomatic-typecheck
        if args:
            field_type = args[0]

    if field_type is float:
        try:
            return float(value)
        except (ValueError, TypeError):
            print_rank(logger.warning, f"Cannot convert '{value}' to float, keeping as str")
            return value

    if field_type is int:
        try:
            return int(value)
        except (ValueError, TypeError):
            print_rank(logger.warning, f"Cannot convert '{value}' to int, keeping as str")
            return value

    if field_type is torch.dtype:
        try:
            return get_dtype(value)
        except (ValueError, TypeError) as error:
            print_rank(logger.warning, f"Cannot convert '{value}' to torch.dtype: {error}, keeping as str")
            return value

    return value


def _dict_to_dataclass(cls, data):
    if data is None:
        return None
    if not isinstance(data, dict):
        return data
    import dataclasses as dc

    field_types = {field.name: field.type for field in dc.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            print_rank(logger.warning, f"Ignoring unknown config key '{key}' for {cls.__name__}")
            continue
        field_type = field_types[key]
        origin = getattr(field_type, "__origin__", None)
        optional_dataclass = None
        if origin is Union:
            for arg in field_type.__args__:
                if arg is not type(None) and dc.is_dataclass(arg):  # pylint: disable=unidiomatic-typecheck
                    optional_dataclass = arg
                    break
        # Structured recompute plans include replay scopes and use_reentrant,
        # consumed by validate_recompute_config rather than RecomputePlanConfig.
        if cls is MemoryConfig and key == "recompute_plan":
            kwargs[key] = value
        elif dc.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        elif optional_dataclass is not None and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(optional_dataclass, value)
        else:
            kwargs[key] = _coerce_value(value, field_type)
    return cls(**kwargs)


def load_config_from_yaml(yaml_path):
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # Extract top-level sub-configs (promoted from old TrainConfig)
    model_raw = raw.pop("model", None)
    optimizer_raw = raw.pop("optimizer", None)
    data_raw = raw.pop("data", None)
    run_raw = raw.pop("run", None)
    checkpoint_raw = raw.pop("checkpoint", None)

    model_cfg = _dict_to_dataclass(ModelConfig, model_raw) if model_raw is not None else ModelConfig()
    optimizer_cfg = (
        _dict_to_dataclass(OptimizerConfig, optimizer_raw) if optimizer_raw is not None else OptimizerConfig()
    )
    data_cfg = _dict_to_dataclass(DataConfig, data_raw) if data_raw is not None else DataConfig()
    run_cfg = _dict_to_dataclass(TrainRunConfig, run_raw) if run_raw is not None else TrainRunConfig()
    checkpoint_cfg = (
        _dict_to_dataclass(CheckpointConfig, checkpoint_raw) if checkpoint_raw is not None else CheckpointConfig()
    )
    # Extract distributed sub-config
    distributed_raw = raw.pop("distributed", None)
    if distributed_raw is not None:
        # Extract plan configs from distributed dict
        fsdp_plan_raw = distributed_raw.pop("fsdp_plan", None)
        tp_plan_raw = distributed_raw.pop("tp_plan", None)
        ep_plan_raw = distributed_raw.pop("ep_plan", None)
        cp_plan_raw = distributed_raw.pop("cp_plan", None)
        vp_plan_raw = distributed_raw.pop("vp_plan", None)

        fsdp_plan = _dict_to_dataclass(FSDPPlanConfig, fsdp_plan_raw) if fsdp_plan_raw is not None else None
        tp_plan = _dict_to_dataclass(TPPlanConfig, tp_plan_raw) if tp_plan_raw is not None else None
        ep_plan = _dict_to_dataclass(EPPlanConfig, ep_plan_raw) if ep_plan_raw is not None else None
        cp_plan = _dict_to_dataclass(CPPlanConfig, cp_plan_raw) if cp_plan_raw is not None else None
        vp_plan = _dict_to_dataclass(VPPlanConfig, vp_plan_raw) if vp_plan_raw is not None else None

        distributed_valid_fields = {f.name for f in fields(DistributedConfig)}
        extra_keys = set(distributed_raw.keys()) - distributed_valid_fields
        if extra_keys:
            print_rank(logger.warning, f"Ignoring unknown distributed config keys: {extra_keys}")
        filtered_distributed_raw = {k: v for k, v in distributed_raw.items() if k in distributed_valid_fields}

        distributed_cfg = DistributedConfig(
            fsdp_plan=fsdp_plan,
            tp_plan=tp_plan,
            ep_plan=ep_plan,
            cp_plan=cp_plan,
            vp_plan=vp_plan,
            **filtered_distributed_raw,
        )
    else:
        distributed_cfg = DistributedConfig()

    # Extract memory sub-config
    memory_raw = raw.pop("memory", None)
    if memory_raw is not None:
        memory_cfg = _dict_to_dataclass(MemoryConfig, memory_raw)
    else:
        memory_cfg = MemoryConfig()

    # Extract quantization sub-config
    quantization_raw = raw.pop("quantization", None)
    if quantization_raw is not None:
        quantization_plan_raw = quantization_raw.pop("quantization_plan", None)
        quantization_plan = (
            _dict_to_dataclass(QuantizeConfig, quantization_plan_raw) if quantization_plan_raw is not None else None
        )
        quantization_cfg = QuantizationConfig(quantization_plan=quantization_plan)
    else:
        quantization_cfg = QuantizationConfig()

    valid_fields = {f.name for f in fields(FSDPTurboConfig)}
    extra_keys = set(raw.keys()) - valid_fields
    if extra_keys:
        print_rank(logger.warning, f"Ignoring unknown top-level config keys: {extra_keys}")
    filtered_raw = {k: v for k, v in raw.items() if k in valid_fields}

    return FSDPTurboConfig(
        model=model_cfg,
        optimizer=optimizer_cfg,
        data=data_cfg,
        run=run_cfg,
        checkpoint=checkpoint_cfg,
        distributed=distributed_cfg,
        memory=memory_cfg,
        quantization=quantization_cfg,
        **filtered_raw,
    )

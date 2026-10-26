# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Keep the configuration schema and its __post_init__ validation together.
# pylint: disable=too-many-lines
import logging
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Callable, Literal, Union, Optional
from pathlib import Path

import torch
import yaml

from fsdp_turbo.utils.dtype import get_dtype
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match

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
    use_fused_vision_flash_attn: bool = False
    # DeepSeek-V4.1 head projection precision; cross entropy remains FP32.
    head_compute_dtype: Literal["fp32", "bf16", "hf32"] = "fp32"
    attn_implementation: str = 'eager'
    # MTP (Multi-Token Prediction) layers to train. 0 disables MTP.
    mtp_num_layers: int = 0
    # Scaling factor applied to the mean MTP loss before adding to the main loss.
    mtp_loss_scaling_factor: float = 0.1


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


@dataclass
class ProfileConfig:
    enabled: bool = False
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
    ulysses_function_patches: List[Dict[str, Any]] = field(default_factory=list)
    kcp_function_patches: List[Dict[str, Any]] = field(default_factory=list)
    kv_allgather_function_patches: List[Dict[str, Any]] = field(default_factory=list)
    loss_function_patches: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EPPlanConfig:
    """Configuration for Expert Parallelism (EP) plan for MoE models."""

    apply_modules: List[str] = field(default_factory=list)
    dispatcher: Union[Literal["all2all", "allgather", "mc2", "domino", "chunkmoe"], Callable, str] = "all2all"
    fixed_router: bool = False
    apply_efsdp_modules: List[str] = field(default_factory=list)
    # Gradient divide factor for expert HSDP. When unset (None) it is
    # computed in the config validation as ep * edp * efsdp.
    gradient_divide_factor: Optional[float] = None
    use_fused_ops: bool = True
    use_swiglu: bool = True
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
    recompute_plan: List[str] = None
    chunk_batch: bool = False
    chunk_batch_plan: Optional[ChunkBatchPlanConfig] = None
    chunk_loss: bool = False
    chunk_loss_size: int = 512
    swap_activation: bool = False
    swap_activation_plan: Optional[SwapActivationPlanConfig] = None


@dataclass
class QuantizationConfig:
    quantization_plan: Optional[QuantizeConfig] = None


@dataclass
class FSDPTurboConfig:
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
        self.validate_model_config()
        self.validate_data_config()
        self.validate_tp_config()
        self.validate_chunk_loss_config()
        self.validate_cp_config()
        self.validate_module_patches()
        self.validate_ep_config()
        self.validate_engram_config()
        self.validate_recompute_config()
        self.validate_chunk_batch_config()
        self.validate_swap_activation_config()
        self.validate_quantization_config()
        self.validate_fsdp_config()
        self.validate_vp_config()
        self.validate_memory_snapshot_config()

    def validate_chunk_loss_config(self):
        if not isinstance(self.memory.chunk_loss, bool):
            raise ValueError("memory.chunk_loss must be a boolean.")
        chunk_size = self.memory.chunk_loss_size
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("memory.chunk_loss_size must be a positive integer.")
        if not self.memory.chunk_loss:
            return
        if any(
            size > 1
            for size in (
                self.distributed.ulysses_parallel_size,
                self.distributed.kv_allgather_parallel_size,
                self.distributed.kcp_parallel_size,
            )
        ):
            raise ValueError("memory.chunk_loss does not support context parallelism.")
        recompute_patterns = (self.memory.recompute_plan or []) if self.memory.recompute else []
        swap_plan = self.memory.swap_activation_plan
        swap_patterns = (swap_plan.apply_modules or []) if self.memory.swap_activation and swap_plan else []
        if any(
            module_name_match(pattern, name)
            for pattern in [*recompute_patterns, *swap_patterns]
            for name in ("", "model.head")
        ):
            raise ValueError("memory.chunk_loss cannot wrap the head or root in recompute/swap_activation hooks.")

    def validate_model_config(self):
        """Validate model loading configuration."""
        if self.model.config_path is not None and (
            not isinstance(self.model.config_path, str) or not self.model.config_path.strip()
        ):
            raise ValueError("model.config_path must be a non-empty string when provided.")
        if not isinstance(self.model.use_fused_clipped_swiglu, bool):
            raise ValueError("model.use_fused_clipped_swiglu must be a boolean.")
        if not isinstance(self.model.use_triton_inplace_rope, bool):
            raise ValueError("model.use_triton_inplace_rope must be a boolean.")
        if self.model.head_compute_dtype not in ("fp32", "bf16", "hf32"):
            raise ValueError("model.head_compute_dtype must be 'fp32', 'bf16' or 'hf32'.")
        if not isinstance(self.model.attn_implementation, str) or not self.model.attn_implementation:
            raise ValueError("model.attn_implementation must be a non-empty string.")
        if self.model.load_pretrained_weights and not self.model.model_name_or_path:
            raise ValueError("model.model_name_or_path is required when load_pretrained_weights is enabled.")
        if self.model.pretrained_engram_rows_per_chunk <= 0:
            raise ValueError("model.pretrained_engram_rows_per_chunk must be positive.")

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
        if self.distributed.fully_shard_parallel_size > 1:
            if self.distributed.expert_parallel_size > 1:
                self.distributed.fsdp_plan.ignored_modules.extend(self.distributed.ep_plan.apply_modules)

            self.distributed.fsdp_plan.ignored_modules = list(
                set(self.distributed.fsdp_plan.ignored_modules)
            )  # remove duplicates

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

    def validate_cp_config(self) -> None:
        """Normalize CP patch specs and reject unsupported parallel configurations."""
        kcp_parallel_size = self.distributed.kcp_parallel_size
        ulysses_parallel_size = self.distributed.ulysses_parallel_size
        for field_name, size in (
            ("kv_allgather_parallel_size", self.distributed.kv_allgather_parallel_size),
            ("kcp_parallel_size", kcp_parallel_size),
            ("ulysses_parallel_size", ulysses_parallel_size),
        ):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"distributed.{field_name} must be a positive integer; got {size!r}.")

        self.distributed.cp_plan = CPPlanConfig() if self.distributed.cp_plan is None else self.distributed.cp_plan
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
                if patch_type == "compressor_sparse_attention" and field_name != "kv_allgather_function_patches":
                    raise ValueError("Compressor sparse attention CP must use kv_allgather_function_patches.")
                normalized.append(normalized_patch)
            setattr(self.distributed.cp_plan, field_name, normalized)

        compressor_sparse_patches = [
            spec
            for spec in self.distributed.cp_plan.kv_allgather_function_patches
            if spec["type"] == "compressor_sparse_attention"
        ]
        if compressor_sparse_patches:
            if len(self.distributed.cp_plan.kv_allgather_function_patches) != 1:
                raise ValueError("DeepSeek-V4.1 CP requires exactly one compressor sparse attention patch.")
            if ulysses_parallel_size != 1 or kcp_parallel_size != 1 or self.distributed.tensor_parallel_size != 1:
                raise ValueError("DeepSeek-V4.1 KV-AllGather requires Ulysses=KCP=TP=1.")
            if self.memory.recompute or self.memory.chunk_batch or self.distributed.cp_plan.loss_function_patches:
                raise ValueError("DeepSeek-V4.1 CP does not support recompute/chunk_batch or external loss patches.")
            if self.data.is_multimodal:
                raise ValueError("DeepSeek-V4.1 CP currently supports text batches only.")
            if self.data.sfa_layout == "TND":
                raise ValueError("DeepSeek-V4.1 CP currently supports BSND input only.")
            if any(
                (
                    self.model.use_sparse_flash_attn,
                    self.model.use_fused_lightning_indexer,
                    self.model.use_fused_compressor,
                    self.model.use_fused_mhc,
                    self.model.use_fused_clipped_swiglu,
                    self.model.use_triton_inplace_rope,
                    self.model.use_fused_vision_flash_attn,
                )
            ):
                raise ValueError("DeepSeek-V4.1 CP requires all fused flags and Triton RoPE disabled.")

        if kcp_parallel_size > 1 and ulysses_parallel_size > 1:
            raise ValueError(
                "distributed.kcp_parallel_size and distributed.ulysses_parallel_size "
                "cannot be enabled together; KCP already applies Ulysses to MLA on "
                "the KCP sequence group."
            )
        if kcp_parallel_size > 1:
            if not self.distributed.cp_plan.kcp_function_patches:
                raise ValueError("distributed.cp_plan.kcp_function_patches must be set when KCP is enabled.")
            if self.distributed.tensor_parallel_size != 1:
                raise ValueError(
                    "KDA state context parallelism requires distributed.tensor_parallel_size == 1; "
                    f"got {self.distributed.tensor_parallel_size}."
                )
            if self.distributed.kv_allgather_parallel_size != 1:
                raise ValueError(
                    "KDA state context parallelism requires distributed.kv_allgather_parallel_size == 1; "
                    f"got {self.distributed.kv_allgather_parallel_size}."
                )
            if self.data.batch_size != 1:
                raise ValueError(
                    "KDA state context parallelism requires micro batch size 1 "
                    f"(data.batch_size == 1); got {self.data.batch_size}."
                )
        if ulysses_parallel_size > 1 and not self.distributed.cp_plan.ulysses_function_patches:
            raise ValueError("distributed.cp_plan.ulysses_function_patches must be set when Ulysses is enabled.")
        if (
            self.distributed.kv_allgather_parallel_size > 1
            and not self.distributed.cp_plan.kv_allgather_function_patches
        ):
            raise ValueError(
                "distributed.cp_plan.kv_allgather_function_patches must be set when KV-AllGather CP is enabled."
            )

    def validate_ep_config(self) -> None:
        """expert parallelize plan

        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                ep_plan=EPPlanConfig(
                    apply_modules: ['*mlp.experts*'],
                    dispatcher: 'all2all', 'mc2', 'domino', 'allgather', 'chunkmoe',
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
        if (
            self.model.use_fused_clipped_swiglu
            and self.distributed.ep_plan.apply_modules
            and self.distributed.ep_plan.dispatcher != "custom_native_eager_forward"
        ):
            raise ValueError(
                "model.use_fused_clipped_swiglu requires DeepSeek-V4.1's custom_native_eager_forward EP dispatcher."
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
            for ep_module in self.distributed.ep_plan.apply_modules:
                if ep_module.endswith(".experts"):
                    self.distributed.ep_plan.apply_efsdp_modules.append(ep_module)

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
            compressor_sparse_cp = any(
                spec["type"] == "compressor_sparse_attention"
                for spec in self.distributed.cp_plan.kv_allgather_function_patches
            )
            if not compressor_sparse_cp:
                raise ValueError("Host-offload Engram CP requires the compressor_sparse_attention KV-AllGather patch.")

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

    def validate_recompute_config(self):
        self.memory.recompute_plan = [] if self.memory.recompute_plan is None else self.memory.recompute_plan

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
        if dc.is_dataclass(field_type) and isinstance(value, dict):
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

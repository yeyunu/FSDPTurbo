# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Trainable causal-LM adapter for the vendored DeepSeek-V4.1 architecture."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from fsdp_turbo.memory.chunk_loss.chunkloss_lm_head import apply_chunkloss_module
from fsdp_turbo.memory.chunk_loss.chunk_loss import (
    calculate_lm_loss,
    chunk_loss as compute_chunk_loss,
    fixed_cross_entropy,
)
from fsdp_turbo.memory.recompute.recompute import register_replay_context
from fsdp_turbo.ops.hf32_linear import hf32_linear

from .engram.host_offload_engram import HostOffloadEngram

from . import model as reference_model

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .image_processor import ImageInput


def _load_model_args(*, config_path: str | Path | None = None, model_name_or_path: str | Path | None = None) -> dict:
    if config_path is not None:
        selected_path = Path(config_path)
    elif model_name_or_path:
        selected_path = Path(model_name_or_path) / "config.json"
    else:
        raise FileNotFoundError("Provide config_path or a model_name_or_path directory containing config.json.")
    with selected_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


@register_replay_context
def _shared_attn_replay_isolation(module):
    """Give a checkpoint replay its own copy of the ``shared_attn`` slots.

    The factory runs when the checkpoint is wired, i.e. at the start of ``module``'s forward, so
    the snapshot holds exactly the slots the preceding source layers published for this module's
    consumers. On every recompute the replay reads and writes that copy; the live state is put
    back on exit so earlier layers' backward keeps consuming the tensors it saved. Capturing only
    at that point is also why the state cannot come from the end of the forward: with several
    ``kv_source_layers`` groups the slots then belong to the last group.
    """
    runtime = reference_model.shared_attn
    return runtime.replay_window(runtime.snapshot())


@dataclass
class DeepseekV41CausalLMOutput:
    """Minimal Transformers-like output consumed by :class:`BaseTrainer`."""

    loss: torch.Tensor | None
    logits: torch.Tensor | None
    hidden_states: tuple[torch.Tensor, ...] | None = None


def _validate_training_args(model_args) -> None:
    if model_args.dtype != "bf16" or model_args.expert_dtype is not None:
        raise ValueError("DeepSeek-V4.1 training requires dtype='bf16' and expert_dtype=None.")
    if model_args.n_mtp_layers or model_args.dspark_block_size:
        raise ValueError("DSpark/MTP is an inference feature and must be disabled for causal-LM training.")
    if len(model_args.compress_ratios) < model_args.n_layers:
        raise ValueError("compress_ratios must contain at least one entry per backbone layer.")


class DeepseekV41ForCausalLMAdapter(nn.Module):
    """Trainable causal-LM facade over DeepSeek's reference ``Transformer``."""

    def __init__(
        self,
        backbone: nn.Module,
        model_args: reference_model.ModelArgs,
        *,
        chunk_loss: bool = False,
        chunk_loss_size: int = 512,
    ) -> None:
        """Assemble an optional loss-aware head before parallel wrapping.

        The head calls this adapter's loss_function with already shifted targets.
        """
        super().__init__()
        self.model = backbone
        self.model_args = model_args
        self.enable_chunk_loss = chunk_loss
        self.chunk_size = chunk_loss_size
        self._kv_cp_group: torch.distributed.ProcessGroup | None = None
        if chunk_loss:
            apply_chunkloss_module(backbone.head)

    def _calculate_head_loss(self, hidden_states, head_weight, head_bias, *, shift_labels, alpha, reduction):
        if not self.model.head.use_hf32:
            return calculate_lm_loss(
                hidden_states, head_weight, head_bias, shift_labels=shift_labels, alpha=alpha, reduction=reduction
            )
        logits = hf32_linear(hidden_states, head_weight).float()
        loss = fixed_cross_entropy(
            logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1), alpha=alpha, reduction=reduction
        )
        return loss, logits

    def loss_function(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        shift_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute chunked mean loss for aligned labels, or full logits without labels.

        Called by the public head interface inside its FSDP forward boundary.
        Targets equal to -100 are ignored; entirely ignored targets yield zero.
        Projection follows the head's compute_dtype/use_hf32; loss uses FP32.
        """
        hidden_states = hidden_states.to(self.model.head.compute_dtype)
        head_weight = head_weight.to(self.model.head.compute_dtype)
        if head_bias is not None:
            head_bias = head_bias.to(self.model.head.compute_dtype)
        if shift_labels is None:
            if self.model.head.use_hf32:
                logits = hf32_linear(hidden_states, head_weight)
                return logits if head_bias is None else logits + head_bias
            return F.linear(hidden_states, head_weight, head_bias)
        num_items = (shift_labels != -100).sum().clamp_min(1)
        loss_kwargs_chunks = [
            dict(shift_labels=labels, alpha=num_items, reduction="sum")
            for labels in torch.split(shift_labels, self.chunk_size, dim=1)
        ]
        return compute_chunk_loss(
            hidden_states, head_weight, head_bias, self._calculate_head_loss, loss_kwargs_chunks, self.chunk_size
        )

    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        *,
        strict: bool = True,
        engram_rows_per_chunk: int = 65_536,
    ):
        """Load dequantized HF weights after FSDPTurbo has sharded this model."""
        from .state_dict_adapter import load_deepseek_v41_from_pretrained

        return load_deepseek_v41_from_pretrained(
            self,
            pretrained_model_name_or_path,
            strict=strict,
            engram_rows_per_chunk=engram_rows_per_chunk,
        )

    def set_context_parallel_group(self, group: torch.distributed.ProcessGroup) -> None:
        """Configure CP before training without modifying any module's methods.

        The adapter passes this group to every attention layer and to CP loss.
        A single-rank group retains the ordinary training/inference path.
        """
        if group is None:
            raise ValueError("Native KV-AllGather CP requires an explicit process group.")
        self._kv_cp_group = group if torch.distributed.get_world_size(group) > 1 else None

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        images: list[list[ImageInput]] | None = None,
        token_types: torch.LongTensor | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        sfa_layout: str | list[str] | tuple[str, ...] = "BSND",
        actual_seq_len: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **model_inputs: object,
    ) -> DeepseekV41CausalLMOutput:
        """Compute training logits and optional causal loss.

        CP consumes text shards or complete image batches fused before sharding,
        with complete, unshifted labels. Unsupported cache/layout combinations raise
        ValueError before the corresponding model computation.
        """
        if use_cache:
            raise ValueError("DeepSeek-V4.1 FSDPTurbo training does not support KV cache inputs.")
        if model_inputs:
            unsupported_inputs = ", ".join(sorted(model_inputs))
            raise ValueError(f"Unsupported model inputs: {unsupported_inputs}.")
        if images is not None:
            has_image = any(
                img is not None
                for sample in images
                for img in (sample if isinstance(sample, (list, tuple)) else (sample,))
            )
            if not has_image and not self.model_args.vision_enabled:
                # Text-only batch from the multimodal collate path: the collate always
                # emits an images entry (all-None placeholders). Drop it only while the
                # vision tower is off: with the tower enabled the reference forward runs
                # the vision/aligner dummy forward for this entry, which keeps the
                # collective order identical on ranks whose batch does contain images.
                # token_types (all TEXT=-1) is still accepted either way.
                images = None
        if images is not None and token_types is None:
            raise ValueError("images requires token_types to locate image spans.")
        if images is not None and not self.model_args.vision_enabled:
            raise ValueError("Image inputs require an enabled DeepSeek vision tower.")

        cp_group = self._kv_cp_group
        if isinstance(sfa_layout, (list, tuple)):
            if len(sfa_layout) != 1:
                raise ValueError("A batch must use one common sfa_layout.")
            sfa_layout = sfa_layout[0]
        sfa_layout = sfa_layout.upper()
        if sfa_layout not in ("BSND", "TND"):
            raise ValueError(f"sfa_layout must be BSND or TND, got {sfa_layout!r}.")
        if cp_group is not None and sfa_layout != "BSND":
            raise ValueError("DeepSeek-V4.1 CP currently supports BSND input only.")
        cu_seqlens = None
        if sfa_layout == "TND":
            if not self.model_args.use_sparse_flash_attn:
                raise ValueError("sfa_layout='TND' requires use_sparse_flash_attn=True.")
            if input_ids.ndim != 2 or input_ids.size(0) != 1:
                raise ValueError("TND requires input_ids with shape [1, total_tokens].")
            if actual_seq_len is None:
                raise ValueError("actual_seq_len is required when sfa_layout='TND'.")
            actual_seq_len = actual_seq_len.reshape(-1).to(device=input_ids.device, dtype=torch.int32)
            if actual_seq_len.numel() == 0:
                raise ValueError("actual_seq_len must contain at least one sequence end.")
            if actual_seq_len[0].item() <= 0 or not torch.all(actual_seq_len[1:] > actual_seq_len[:-1]).item():
                raise ValueError("actual_seq_len must be strictly increasing and positive.")
            if actual_seq_len[-1].item() != input_ids.numel():
                raise ValueError(f"actual_seq_len must finish at the flattened token count {input_ids.numel()}.")
            cu_seqlens = F.pad(actual_seq_len, (1, 0), value=0).contiguous()
            sequence_lengths = torch.diff(cu_seqlens)
            for ratio in set(self.model_args.compress_ratios[: self.model_args.n_layers]):
                if ratio and torch.any(sequence_lengths % ratio != 0).item():
                    raise ValueError(
                        "TND sequence lengths must be divisible by every active "
                        f"compression ratio; ratio={ratio}, lengths={sequence_lengths.tolist()}."
                    )
            if cu_seqlens.numel() > 2:
                if self.model_args.use_fused_compressor:
                    raise NotImplementedError("Packed TND does not yet support the fused compressor.")
                if self.model_args.use_fused_lightning_indexer:
                    raise NotImplementedError("Packed TND does not yet support fused lightning indexers.")
            if position_ids is not None:
                expected_position_ids = torch.cat(
                    [torch.arange(length, device=input_ids.device) for length in sequence_lengths.tolist()]
                ).reshape_as(input_ids)
                if not torch.equal(position_ids.to(device=input_ids.device), expected_position_ids):
                    raise ValueError("Packed TND position_ids must restart at zero for every document.")
        elif actual_seq_len is not None:
            raise ValueError("actual_seq_len is only valid when sfa_layout='TND'.")
        elif position_ids is not None and cp_group is None:
            raise ValueError("position_ids is currently only valid for TND or CP input.")

        engram_hashes = None
        if cp_group is not None:
            from .context_parallel import prepare_cp_inputs

            hash_input_ids, hash_slice, full_mask, hash_mask = prepare_cp_inputs(
                self, input_ids, position_ids, attention_mask, labels, images, token_types
            )
            if self.model.engram_hash is not None:
                # Hash complete token history before slicing so CP shard edges
                # do not become artificial n-gram sequence boundaries.
                engram_hashes = self.model.engram_hash(hash_input_ids, 0, hash_mask)[:, hash_slice]
            if labels is not None:
                labels = labels.masked_fill(~full_mask.to(device=labels.device), -100)

        hidden_states, _, collected_hidden_states = self.model.forward_backbone(
            input_ids,
            images=images,
            token_types=token_types,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            sfa_layout=sfa_layout,
            cu_seqlens=cu_seqlens,
            engram_hashes=engram_hashes,
            cp_group=cp_group,
        )
        shift_labels = None
        if labels is not None and cp_group is None:
            if cu_seqlens is not None and cu_seqlens.numel() > 2:
                labels = labels.clone()
                boundary_starts = cu_seqlens[1:-1].to(device=labels.device, dtype=torch.long)
                labels.view(-1)[boundary_starts] = -100
            shift_labels = labels[:, 1:].contiguous().to(hidden_states.device)
            if attention_mask is not None:
                shift_labels = shift_labels.masked_fill(~attention_mask[:, 1:].bool(), -100)

        if self.enable_chunk_loss and shift_labels is not None:
            logits = None
            loss = self.model.head(hidden_states[:, :-1], self.loss_function, shift_labels)
        else:
            logits = (
                self.model.head(hidden_states, self.loss_function)
                if self.enable_chunk_loss
                else self.model.head(hidden_states, full_logits=True)
            )
            loss = None
            if labels is not None and cp_group is not None:
                from fsdp_turbo.distributed.context_parallel.loss import cp_compute_lm_loss

                loss = cp_compute_lm_loss(logits, labels, cp_group)
            elif shift_labels is not None:
                loss = F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                valid_target_count = (shift_labels != -100).sum().clamp_min(1)
                loss = loss / valid_target_count

        return DeepseekV41CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=collected_hidden_states,
        )


def build_deepseek_v41_model_args(
    *, config_path: str | Path | None = None, model_name_or_path: str | Path | None = None
):
    """Build training arguments from an explicit JSON or model directory's config.json."""
    model_args = reference_model.ModelArgs(
        **_load_model_args(config_path=config_path, model_name_or_path=model_name_or_path)
    )
    _validate_training_args(model_args)
    return model_args


def _build_deepseek_v41_model(
    model_args,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    head_compute_dtype: Literal["fp32", "bf16", "hf32"] = "fp32",
    chunk_loss: bool = False,
    chunk_loss_size: int = 512,
) -> DeepseekV41ForCausalLMAdapter:
    if head_compute_dtype not in ("fp32", "bf16", "hf32"):
        raise ValueError("head_compute_dtype must be 'fp32', 'bf16' or 'hf32'.")
    with reference_model.set_dtype(torch.bfloat16):
        backbone = reference_model.Transformer(model_args, tokenizer=tokenizer)
    backbone.head.compute_dtype = torch.bfloat16 if head_compute_dtype == "bf16" else torch.float32
    backbone.head.use_hf32 = head_compute_dtype == "hf32"
    return DeepseekV41ForCausalLMAdapter(
        backbone,
        model_args,
        chunk_loss=chunk_loss,
        chunk_loss_size=chunk_loss_size,
    )


def build_deepseek_v41_model(
    *,
    model_args: reference_model.ModelArgs | None = None,
    config_path: str | Path | None = None,
    model_name_or_path: str | Path | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    engram_storage_backend: Literal["host_offload", "elastic_buffer"] = "host_offload",
    use_sparse_flash_attn: bool = False,
    use_fused_lightning_indexer: bool = False,
    use_fused_compressor: bool = False,
    use_fused_mhc: bool = False,
    use_fused_clipped_swiglu: bool = False,
    use_triton_inplace_rope: bool = False,
    use_deepgemm_wo_a: bool = False,
    use_fused_vision_flash_attn: bool = False,
    use_fused_partial_rotary: bool = False,
    use_fused_engram: bool = False,
    head_compute_dtype: Literal["fp32", "bf16", "hf32"] = "fp32",
    chunk_loss: bool = False,
    chunk_loss_size: int = 512,
) -> DeepseekV41ForCausalLMAdapter:
    """Build an uninitialized BF16 model from model_args or a selected config JSON.

    Host Engram tables start on meta storage. FSDPTurbo later shards and
    initializes them without allocating their full storage. chunk_loss enables
    the public chunk-loss head, using this adapter's loss_function.
    ``use_deepgemm_wo_a`` selects the optional BF16 NPU output projection;
    its DeepGEMM dependency is loaded on the first projection call.
    """
    if model_args is None:
        model_args = build_deepseek_v41_model_args(config_path=config_path, model_name_or_path=model_name_or_path)
    else:
        _validate_training_args(model_args)
    model_args = replace(
        model_args,
        engram_storage_backend=engram_storage_backend,
        use_sparse_flash_attn=use_sparse_flash_attn,
        use_fused_lightning_indexer=use_fused_lightning_indexer,
        use_fused_compressor=use_fused_compressor,
        use_fused_mhc=use_fused_mhc,
        use_fused_clipped_swiglu=use_fused_clipped_swiglu,
        use_triton_inplace_rope=use_triton_inplace_rope,
        use_deepgemm_wo_a=use_deepgemm_wo_a,
        use_fused_vision_flash_attn=use_fused_vision_flash_attn,
        use_fused_partial_rotary=use_fused_partial_rotary,
        use_fused_engram=use_fused_engram,
    )
    return _build_deepseek_v41_model(
        model_args,
        tokenizer=tokenizer,
        head_compute_dtype=head_compute_dtype,
        chunk_loss=chunk_loss,
        chunk_loss_size=chunk_loss_size,
    )


_IMAGE_MARKERS = {"image_start", "image_newline", "image_end"}
_MHC_PARAMETERS = {
    "hc_attn_fn",
    "hc_ffn_fn",
    "hc_attn_base",
    "hc_ffn_base",
    "hc_attn_scale",
    "hc_ffn_scale",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
}


def _depth_std(layer_id: int, base_std: float) -> float:
    return base_std / math.sqrt(2.0 * (layer_id + 1))


def _materialize_parameters(model: nn.Module, device: torch.device) -> None:
    """Allocate only meta parameter shards, preserving aliases and all buffers."""
    materialized_parameters: dict[int, nn.Parameter] = {}
    for module in model.modules():
        if isinstance(module, HostOffloadEngram):
            continue
        for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
            if not parameter.is_meta:
                continue
            parameter_id = id(parameter)
            if parameter_id not in materialized_parameters:
                materialized_parameters[parameter_id] = nn.Parameter(
                    torch.empty_like(parameter, device=device), requires_grad=parameter.requires_grad
                )
            module.register_parameter(name, materialized_parameters[parameter_id])


@torch.no_grad()
def _init_weights(
    module: nn.Module,
    *,
    module_name: str,
    layer_ids: dict[int, int],
    standard_deviation: float,
) -> None:
    """Initialize one module's direct parameters using DeepSeek's sampling rules.

    Module path and owning Block depths select vision/shared-expert rules.
    Meta parameters and Host Engram storage are left to their allocation owners.
    """
    if isinstance(module, HostOffloadEngram):
        # This table owns its row-wise initialization. After parallelism
        # it may already be materialized on the host, so never overwrite
        # it through the generic rules below.
        return

    is_vision_linear = isinstance(module, nn.Linear) and (
        ".vision." in f".{module_name}." or ".aligner." in f".{module_name}."
    )
    is_norm = "norm" in type(module).__name__.lower()
    for local_name, parameter in module.named_parameters(recurse=False):
        if parameter.is_meta:
            continue

        if is_vision_linear:
            if local_name == "weight":
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
            else:
                bound = 1.0 / math.sqrt(module.in_features)
                nn.init.uniform_(parameter, -bound, bound)
        elif local_name in _IMAGE_MARKERS:
            nn.init.normal_(parameter, mean=0.0, std=1.0)
        elif is_norm:
            if local_name == "weight":
                nn.init.ones_(parameter)
            else:
                nn.init.zeros_(parameter)
        elif local_name in {"q_weight", "k_weight"}:
            nn.init.ones_(parameter)
        elif local_name in _MHC_PARAMETERS or local_name == "attn_sink":
            nn.init.normal_(parameter, mean=0.0, std=standard_deviation)
        elif isinstance(module, reference_model.ParallelEmbedding):
            nn.init.normal_(parameter, mean=0.0, std=1.0)
        elif isinstance(module, reference_model.ParallelHead):
            output_std = module.dim**-0.5
            nn.init.trunc_normal_(
                parameter,
                mean=0.0,
                std=output_std,
                a=-3.0 * output_std,
                b=3.0 * output_std,
            )
        elif isinstance(module, reference_model.DeepSeekV41Experts):
            depth_std = _depth_std(layer_ids[id(module)], standard_deviation)
            if local_name == "gate_up_proj":
                nn.init.normal_(parameter, mean=0.0, std=standard_deviation)
                if standard_deviation != 0:
                    local_parameter = parameter
                    global_row_offset = 0
                    if isinstance(parameter, DTensor):
                        local_parameter = parameter.to_local()
                        _, global_offset = compute_local_shape_and_global_offset(
                            parameter.shape, parameter.device_mesh, parameter.placements
                        )
                        global_row_offset = global_offset[1]
                    # Scale only the local intersection with the global up half.
                    global_up_start = parameter.shape[1] // 2
                    local_up_start = max(global_up_start - global_row_offset, 0)
                    local_parameter[:, local_up_start:].mul_(depth_std / standard_deviation)
            else:
                nn.init.normal_(parameter, mean=0.0, std=depth_std)
        elif (f"{module_name}.{local_name}" if module_name else local_name).endswith(
            (".ffn.gate.weight", ".ffn.shared_experts.w2.weight", ".ffn.shared_experts.w3.weight")
        ):
            nn.init.normal_(parameter, mean=0.0, std=_depth_std(layer_ids[id(module)], standard_deviation))
        elif local_name == "weight" or parameter.ndim > 1:
            nn.init.normal_(parameter, mean=0.0, std=standard_deviation)
        else:
            nn.init.zeros_(parameter)


def initialize_deepseek_v41_model(
    model: nn.Module,
    standard_deviation: float = 0.02,
    *,
    device: torch.device | None = None,
) -> None:
    """Materialize local parameter shards, then initialize each module's weights.

    After EP/FSDP sharding, pass ``device`` to allocate meta parameters on the
    compute device, or CPU with CPU Offload. Omit it for an already materialized
    model. Real buffers and shared parameter aliases are preserved; Host Engram
    tables own their initialization. Parent-before-child traversal preserves
    the existing random sampling order for a fixed seed and parallel topology.
    """
    if device is not None:
        _materialize_parameters(model, device)

    layer_ids = {
        id(module): layer.layer_id
        for layer in model.modules()
        if isinstance(layer, reference_model.Block)
        for module in layer.modules()
    }
    for module_name, module in model.named_modules():
        _init_weights(
            module,
            module_name=module_name,
            layer_ids=layer_ids,
            standard_deviation=standard_deviation,
        )


def prepare_deepseek_v41_model_for_fsdp(
    model: nn.Module, *, device: torch.device, parameter_dtype: torch.dtype
) -> nn.Module:
    """Defer parameter storage until EP/FSDP has selected each rank's shards.

    Call before parallelism and before loading or initializing weights: existing
    parameter values are discarded. Buffers retain their values and dtypes on
    ``device``; deferred Host Engram tables keep their own allocation lifecycle.
    """
    meta_parameters: dict[int, nn.Parameter] = {}
    for module in model.modules():
        if isinstance(module, HostOffloadEngram):
            continue
        for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
            parameter_id = id(parameter)
            if parameter_id not in meta_parameters:
                meta_parameters[parameter_id] = nn.Parameter(
                    torch.empty_like(parameter, device="meta", dtype=parameter_dtype),
                    requires_grad=parameter.requires_grad,
                )
            module.register_parameter(name, meta_parameters[parameter_id])
        # Only buffers move to the accelerator. In particular, do not cast
        # complex RoPE frequencies or discard integer Engram hash constants.
        module._apply(lambda tensor: tensor if tensor.is_meta else tensor.to(device=device), recurse=False)
    return model


__all__ = [
    "DeepseekV41CausalLMOutput",
    "DeepseekV41ForCausalLMAdapter",
    "build_deepseek_v41_model",
    "build_deepseek_v41_model_args",
    "initialize_deepseek_v41_model",
    "prepare_deepseek_v41_model_for_fsdp",
]

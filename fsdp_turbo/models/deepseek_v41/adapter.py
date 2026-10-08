# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Trainable causal-LM adapter for the vendored DeepSeek-V4.1 architecture."""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .host_offload_embedding import HostOffloadEmbedding

from . import model as reference_model

_REFERENCE_DIRECTORY = Path(__file__).parent


def _load_model_args() -> dict:
    with (_REFERENCE_DIRECTORY / "config_8p.json").open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


@dataclass
class DeepseekV41CausalLMOutput:
    """Minimal Transformers-like output consumed by :class:`BaseTrainer`."""

    loss: torch.Tensor | None
    logits: torch.Tensor
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

    def __init__(self, backbone: nn.Module, model_args: reference_model.ModelArgs):
        super().__init__()
        self.model = backbone
        self.model_args = model_args

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        images=None,
        token_types: torch.LongTensor | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        **model_inputs,
    ) -> DeepseekV41CausalLMOutput:
        if use_cache:
            raise ValueError("DeepSeek-V4.1 FSDPTurbo training does not support KV cache inputs.")
        if model_inputs:
            unsupported_inputs = ", ".join(sorted(model_inputs))
            raise ValueError(f"Unsupported model inputs: {unsupported_inputs}.")
        if (images is None) != (token_types is None):
            raise ValueError("images and token_types must be supplied together.")
        if images is not None and not self.model_args.vision_enabled:
            raise ValueError("Image inputs require an enabled DeepSeek vision tower.")

        hidden_states, _, collected_hidden_states = self.model.forward_backbone(
            input_ids,
            images=images,
            token_types=token_types,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        logits = self.model.head(hidden_states, full_logits=True)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous().float()
            shift_labels = labels[:, 1:].contiguous().to(shift_logits.device)
            if attention_mask is not None:
                shift_labels = shift_labels.masked_fill(~attention_mask[:, 1:].bool(), -100)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return DeepseekV41CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=collected_hidden_states,
        )


def build_deepseek_v41_model_args():
    """Build model arguments directly from the vendored training config."""
    model_args = reference_model.ModelArgs(**_load_model_args())
    _validate_training_args(model_args)
    return model_args


def _build_deepseek_v41_model(
    model_args,
    *,
    tokenizer=None,
) -> DeepseekV41ForCausalLMAdapter:
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        backbone = reference_model.Transformer(model_args, tokenizer=tokenizer)
    finally:
        torch.set_default_dtype(previous_dtype)
    return DeepseekV41ForCausalLMAdapter(backbone, model_args)


def build_deepseek_v41_model(
    *,
    model_args: reference_model.ModelArgs | None = None,
    tokenizer=None,
    engram_storage_backend: Literal["host_offload", "elastic_buffer"] = "host_offload",
    use_sparse_flash_attn: bool = False,
    use_fused_lightning_indexer: bool = False,
    use_fused_compressor: bool = False,
    use_fused_mhc: bool = False,
    use_fused_vision_flash_attn: bool = False,
) -> DeepseekV41ForCausalLMAdapter:
    """Build an uninitialized BF16 model from the vendored config JSON.

    Host Engram tables start on meta storage. FSDPTurbo later shards and
    initializes them without allocating their full storage.
    """
    if model_args is None:
        model_args = build_deepseek_v41_model_args()
    else:
        _validate_training_args(model_args)
    model_args = replace(
        model_args,
        engram_storage_backend=engram_storage_backend,
        use_sparse_flash_attn=use_sparse_flash_attn,
        use_fused_lightning_indexer=use_fused_lightning_indexer,
        use_fused_compressor=use_fused_compressor,
        use_fused_mhc=use_fused_mhc,
        use_fused_vision_flash_attn=use_fused_vision_flash_attn,
    )
    return _build_deepseek_v41_model(
        model_args,
        tokenizer=tokenizer,
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


def _depth_std(name: str, base_std: float) -> float:
    layer_id = int(name.split(".layers.", 1)[1].split(".", 1)[0])
    return base_std / math.sqrt(2.0 * (layer_id + 1))


def initialize_deepseek_v41_model(model: nn.Module, standard_deviation: float = 0.02) -> None:
    """Initialize common modules, then apply the few DeepSeek-V4.1-specific rules."""
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if isinstance(module, HostOffloadEmbedding):
                module.init_std = standard_deviation
                module.init_seed = (torch.initial_seed() + zlib.crc32(module_name.encode())) % (2**64)

            for local_name, parameter in module.named_parameters(recurse=False):
                if parameter.is_meta:
                    continue

                name = f"{module_name}.{local_name}" if module_name else local_name
                is_vision_linear = isinstance(module, nn.Linear) and (
                    ".vision." in f".{module_name}." or ".aligner." in f".{module_name}."
                )

                if is_vision_linear:
                    if local_name == "weight":
                        nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
                    else:
                        bound = 1.0 / math.sqrt(module.in_features)
                        nn.init.uniform_(parameter, -bound, bound)
                elif local_name in _IMAGE_MARKERS:
                    nn.init.normal_(parameter, mean=0.0, std=1.0)
                elif "norm" in type(module).__name__.lower():
                    if local_name == "weight":
                        nn.init.ones_(parameter)
                    else:
                        nn.init.zeros_(parameter)
                elif local_name in {"q_weight", "k_weight"}:
                    nn.init.ones_(parameter)
                elif local_name in _MHC_PARAMETERS or local_name == "attn_sink":
                    nn.init.trunc_normal_(parameter, mean=0.0, std=standard_deviation)
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
                    depth_std = _depth_std(name, standard_deviation)
                    if local_name == "gate_up_proj":
                        gate, up = parameter.chunk(2, dim=1)
                        nn.init.trunc_normal_(gate, mean=0.0, std=standard_deviation)
                        nn.init.trunc_normal_(up, mean=0.0, std=depth_std)
                    else:
                        nn.init.trunc_normal_(parameter, mean=0.0, std=depth_std)
                elif name.endswith(".ffn.gate.weight") or name.endswith(
                    (".ffn.shared_experts.w2.weight", ".ffn.shared_experts.w3.weight")
                ):
                    nn.init.trunc_normal_(parameter, mean=0.0, std=_depth_std(name, standard_deviation))
                elif local_name == "weight" or parameter.ndim > 1:
                    nn.init.trunc_normal_(parameter, mean=0.0, std=standard_deviation)
                else:
                    nn.init.zeros_(parameter)


def prepare_deepseek_v41_model_for_fsdp(model: nn.Module, *, device, parameter_dtype: torch.dtype) -> nn.Module:
    """Materialize a uniform parameter dtype before applying native FSDP.

    The reference architecture intentionally creates a mixture of BF16 and FP32
    parameters. Native FSDP requires every managed parameter group to have one
    original dtype, so normalize floating-point parameters here. Buffers are only
    moved to the accelerator: casting them would corrupt complex RoPE frequencies.
    FP32-sensitive forward paths cast their operands locally when computing.
    """
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.to(dtype=parameter_dtype)
    # Keep deferred Engram tables on meta until FSDPTurbo has its data mesh.
    # Move each module's own tensors so parent .to() cannot materialize them.
    for module in model.modules():
        if isinstance(module, HostOffloadEmbedding) and module.weight.is_meta:
            continue
        module._apply(lambda tensor: tensor.to(device=device), recurse=False)
    return model


__all__ = [
    "DeepseekV41CausalLMOutput",
    "DeepseekV41ForCausalLMAdapter",
    "build_deepseek_v41_model",
    "build_deepseek_v41_model_args",
    "initialize_deepseek_v41_model",
    "prepare_deepseek_v41_model_for_fsdp",
]

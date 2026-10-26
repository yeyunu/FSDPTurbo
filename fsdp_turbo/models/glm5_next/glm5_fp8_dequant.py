# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""fp8 dequantization helpers for GLM5Next checkpoints.

``transformers`` may load fine-grained fp8 checkpoints either as the plain
``Glm5NextTextExperts`` (bf16) or as a drop-in fp8 module that keeps fp8
weights + block scales and dequantizes on the fly.  The latter assumes CUDA
kernels and does not fit the fsdp_turbo EP dispatcher, so the expert weights
are dequantized to bf16 **before** FSDP/EP wrapping (data conversion, not a
behaviour patch: the EP forward itself lives on ``Glm5NextTextExperts`` as
regular ``ep_forward_eager`` / ``ep_forward_fused`` methods).
"""

import logging

import torch

logger = logging.getLogger(__name__)

_FP8_DTYPES = ()
for _dt in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"):
    if hasattr(torch, _dt):
        _FP8_DTYPES += (getattr(torch, _dt),)


def _dump_module(module) -> str:
    """Describe a module's parameters/buffers for diagnostics."""
    lines = []
    for name, tensor in {
        **dict(module.named_parameters(recurse=False)),
        **dict(module.named_buffers(recurse=False)),
    }.items():
        if isinstance(tensor, torch.Tensor):
            lines.append(f"    {name}: dtype={tensor.dtype}, shape={tuple(tensor.shape)}")
    return "\n".join(lines) or "    (no direct params/buffers)"


def _dequant_fp8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a fp8 weight with its scale tensor to bf16.

    Supports both scale layouts seen in GLM5.3 checkpoints:
    - fine-grained 128x128 blockwise: weight [E, out, in], scale [E, out/128, in/128]
    - broadcast: scale with ndim smaller than weight, e.g. per-expert
      [E, 1, 1] or per-tensor scalar; broadcast against the fp8 view.
    """
    block = 128
    out_dim, in_dim = weight.shape[-2], weight.shape[-1]
    rows, cols = out_dim // block, in_dim // block
    # Fine-grained 128x128 blockwise scales: a [rows, cols] grid per weight row
    # (expert). 3D weights [E, out, in] carry [E, rows, cols] grids; plain 2D
    # weights [out, in] carry [rows, cols] or [1, rows, cols] grids.
    if scale.ndim in (2, 3) and scale.shape[-1] == cols and scale.shape[-2] == rows:
        if scale.ndim == 2:
            scale = scale.unsqueeze(0)
        scale = scale.repeat_interleave(block, dim=1).repeat_interleave(block, dim=2)
        scale = scale[:, :out_dim, :in_dim]
        return (weight.to(scale.dtype) * scale).reshape(weight.shape).to(torch.bfloat16)
    if scale.numel() == weight.shape[0] and scale.shape[0] == weight.shape[0]:
        # per-expert / per-row scalar scales: [E] / [E,1,1]
        scale = scale.reshape(weight.shape[0], *([1] * (weight.ndim - 1)))
    elif scale.numel() == 1:
        # per-tensor scalar scale
        scale = scale.reshape(*([1] * weight.ndim))
    return (weight.to(scale.dtype) * scale).to(torch.bfloat16)


def _find_scale_name(weight_name: str, tensors: dict) -> str | None:
    """Locate the block-scale tensor matching a given fp8 weight."""
    candidates = [
        f"{weight_name}_scale_inv",
        f"{weight_name}_scale",
        "weight_scale_inv",
        "weight_scale",
    ]
    for candidate in candidates:
        if candidate in tensors:
            return candidate
    # Fallback: a single scale-looking tensor when there is only one weight.
    scale_names = [
        n
        for n, t in tensors.items()
        if "scale" in n and isinstance(t, torch.Tensor) and t.ndim >= 1 and t.dtype.is_floating_point
    ]
    if len(tensors) == 2 and len(scale_names) == 1:
        return scale_names[0]
    return None


def _dequant_module_fp8(module: torch.nn.Module) -> bool:
    """Dequantize any fp8 parameter on this module in place. True if changed."""
    tensors = {**dict(module.named_parameters(recurse=False)), **dict(module.named_buffers(recurse=False))}
    fp8_names = [n for n, t in tensors.items() if isinstance(t, torch.Tensor) and t.dtype in _FP8_DTYPES]
    if not fp8_names:
        return False

    new_params, remove_names = {}, []
    for weight_name in fp8_names:
        weight = tensors[weight_name]
        scale_name = _find_scale_name(weight_name, tensors)
        if scale_name is None:
            raise RuntimeError(
                f"[GLM5 fp8 dequant] Found fp8 weight '{weight_name}' on "
                f"{type(module).__name__} but no matching scale tensor. "
                f"Module attributes:\n{_dump_module(module)}"
            )
        bf16_weight = _dequant_fp8_weight(weight, tensors[scale_name])
        new_params[weight_name] = bf16_weight
        remove_names.extend([weight_name, scale_name])

    for name in remove_names:
        module._parameters.pop(name, None)
        module._buffers.pop(name, None)
    for name, value in new_params.items():
        module.register_parameter(name, torch.nn.Parameter(value, requires_grad=True))
    return True


def is_glm5_experts(module: torch.nn.Module) -> bool:
    """Structural check: GLM5Next-style fused experts module."""
    from torch.distributed.tensor import DTensor

    return (
        isinstance(getattr(module, "gate_up_proj", None), torch.Tensor)
        and isinstance(getattr(module, "down_proj", None), torch.Tensor)
        and not isinstance(module, DTensor)
    )


def dequantize_glm5_experts_fp8(model: torch.nn.Module) -> int:
    """Dequantize fp8 expert weights (and any other fp8 params) to bf16.

    Must run BEFORE FSDPTurbo wrapping and BEFORE convert_model_dtype (a
    blind ``.to(bfloat16)`` on a fp8 param drops its scale and silently
    corrupts the weights). Returns the number of dequantized modules.
    """
    from fsdp_turbo.utils.log import print_rank

    dequantized = 0
    for module in model.modules():
        if _dequant_module_fp8(module):
            dequantized += 1
            if is_glm5_experts(module):
                print_rank(
                    logger.info,
                    f"> Dequantized fp8 experts on {type(module).__name__} to bf16",
                )
    return dequantized

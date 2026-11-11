# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Torch reference implementations for swap payload quantization."""

import torch

from fsdp_turbo.ops.registry import register_op


@register_op('dynamic_quant', 'cpu')
def dynamic_quant_cpu(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a matrix to symmetric int8 with one fp32 scale per row."""
    values = tensor.float()
    scale = values.abs().amax(dim=-1) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    payload = (values / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return payload, scale


@register_op('dynamic_block_quant', 'cpu')
def dynamic_block_quant_cpu(tensor: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each 1xN block to e4m3fn using its fp32 amax / 448 scale."""
    blocks = tensor.float().unflatten(-1, (-1, block_size))
    scale = blocks.abs().amax(dim=-1) / 448.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    payload = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.reshape(tensor.shape), scale


@register_op('dequant_int8_per_row', 'cpu')
def dequant_int8_per_row_cpu(payload: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Reconstruct rows in fp32 and cast to the original activation dtype."""
    return (payload.to(dtype) * scale.reshape(-1, 1)).to(dtype)


@register_op('dequant_fp8_blockwise', 'cpu')
def dequant_fp8_blockwise_cpu(
    payload: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, block_size: int = 128
) -> torch.Tensor:
    """Reconstruct e4m3fn blocks with their fp32 scales."""
    blocks = payload.float().unflatten(-1, (-1, block_size))
    return (blocks * scale[..., None]).to(dtype).reshape(payload.shape)


@register_op('fp8_payload_quant_supported', 'cpu')
def fp8_payload_quant_supported_cpu() -> bool:
    """FP8 swap payloads require Ascend 950PR&950DT products."""
    return False

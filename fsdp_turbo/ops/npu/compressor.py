# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Validated adapter for the training-capable CANN Compressor operator."""

from __future__ import annotations

import torch

from fsdp_turbo.ops.registry import register_op


def _get_compressor_op():
    try:
        from cann_ops_transformer.ops.attention.compressor import compressor
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "use_fused_compressor=True requires cann_ops_transformer with the "
            "training-capable compressor/compressor-grad operators."
        ) from error
    return compressor


@register_op("fused_compressor", "npu")
def fused_compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    ape: torch.Tensor,
    *,
    compress_ratio: int,
    start_pos: torch.Tensor,
) -> torch.Tensor:
    """Compress BSH activations while retaining the packaged autograd path."""
    if x.device.type != "npu":
        raise ValueError("Fused compressor requires NPU tensors")
    if x.ndim != 3:
        raise ValueError(f"Fused compressor x must be BSH, got shape {tuple(x.shape)}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"Fused compressor x must be FP16/BF16, got {x.dtype}")
    if wkv.dtype != x.dtype or wgate.dtype != x.dtype:
        raise TypeError("Fused compressor weights must have the same dtype as x")
    if wkv.ndim != 2 or wgate.shape != wkv.shape:
        raise ValueError("Fused compressor wkv/wgate must have the same 2-D shape")
    output_dim = wkv.shape[0]
    if x.shape[-1] != wkv.shape[1]:
        raise ValueError("Fused compressor x and weight hidden dimensions do not match")
    if compress_ratio < 2 or compress_ratio > 128:
        raise ValueError("Fused compressor compress_ratio must be in [2, 128]")
    if state_cache.dtype != torch.float32 or state_cache.ndim != 3:
        raise TypeError("Fused compressor state_cache must be a 3-D FP32 tensor")
    if state_cache.shape[-1] != 2 * output_dim:
        raise ValueError("Fused compressor state_cache last dimension must be 2 * D")
    if not 1 <= state_cache.shape[1] <= 1024:
        raise ValueError("Fused compressor state_cache block size must be in [1, 1024]")
    if (
        state_block_table.dtype != torch.int32
        or state_block_table.ndim != 2
        or state_block_table.shape[0] != x.shape[0]
    ):
        raise ValueError("Fused compressor state_block_table must be int32 [B, N]")
    if ape.dtype != torch.float32 or ape.shape != (compress_ratio, output_dim):
        raise ValueError("Fused compressor ape must be FP32 with shape [ratio, D]")
    if start_pos.dtype != torch.int32 or start_pos.shape != (x.shape[0],):
        raise ValueError("Fused compressor start_pos must be int32 with shape [B]")

    compressor = _get_compressor_op()
    return compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        compress_ratio,
        state_block_table=state_block_table,
        start_pos=start_pos,
        coff=1,
        cache_mode=1,
    )


__all__ = ["fused_compressor"]

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Adapter for CANN FlashAttentionScore used by the DeepSeek vision tower."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from math import sqrt

import torch

from fsdp_turbo.ops.registry import register_op


_LAYOUT = "BSND"
_FULL_SEQUENCE_TOKENS = 2**31 - 1


@lru_cache(maxsize=1)
def _get_fusion_attention():
    try:
        torch_npu = import_module("torch_npu")
        return torch_npu.npu_fusion_attention
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "use_fused_vision_flash_attn=True requires "
            "torch_npu.npu_fusion_attention backed by "
            "FlashAttentionScore/FlashAttentionScoreGrad"
        ) from error


def _validate_shape_and_dtype(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "npu":
        raise ValueError("Fused vision FlashAttentionScore requires NPU tensors")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Fused vision FlashAttentionScore requires BF16 {name}, got {tensor.dtype}")
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(
            f"Fused vision FlashAttentionScore expects {name} in [1,S,N,D] BSND layout, got {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] <= 0 or tensor.shape[-1] > 512:
        raise ValueError(f"FlashAttentionScore requires head_dim in [1, 512], got {tensor.shape[-1]}")


@register_op("fused_vision_flash_attention", "npu")
def fused_vision_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Compute full bidirectional MHA for one image in BSND layout."""
    for tensor, name in ((query, "query"), (key, "key"), (value, "value")):
        _validate_shape_and_dtype(tensor, name)
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Fused vision FlashAttentionScore requires identical Q/K/V shapes")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    outputs = _get_fusion_attention()(
        query,
        key,
        value,
        query.shape[2],
        input_layout=_LAYOUT,
        pse=None,
        padding_mask=None,
        atten_mask=None,
        scale=1.0 / sqrt(query.shape[-1]),
        keep_prob=1.0,
        pre_tockens=_FULL_SEQUENCE_TOKENS,
        next_tockens=_FULL_SEQUENCE_TOKENS,
        inner_precise=0,
        prefix=None,
        sparse_mode=0,
    )
    return outputs[0]


__all__ = ["fused_vision_flash_attention"]

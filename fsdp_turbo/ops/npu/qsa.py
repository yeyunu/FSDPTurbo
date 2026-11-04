# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""NPU backend of the Qwen4-Exp QSA token-selection operator.

Registers the fused Triton-Ascend implementation from
``fsdp_turbo.ops.triton.qsa_indexer``; that implementation falls back to the
exact eager reference on its own when a layout is unsupported.
"""

import torch

from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.ops.triton.qsa_indexer import qsa_compress_block_keys_triton, qsa_selected_token_mask_triton


@register_op("qsa_selected_token_mask", "npu")
def qsa_selected_token_mask_npu(*args, **kwargs):
    """Run QSA token selection with the Triton-Ascend fast path."""
    return qsa_selected_token_mask_triton(*args, **kwargs)


@register_op("qsa_compress_block_keys", "npu")
def qsa_compress_block_keys_npu(
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    k_norm_weight: torch.Tensor,
    *,
    rms_norm_eps: float,
    compress_ratio: int,
) -> torch.Tensor:
    """Compress local QSA keys with the Triton backend before CP gathering."""
    return qsa_compress_block_keys_triton(
        raw_keys,
        full_cos,
        full_sin,
        k_norm_weight,
        rms_norm_eps=rms_norm_eps,
        compress_ratio=compress_ratio,
    )


__all__ = ["qsa_selected_token_mask_npu", "qsa_compress_block_keys_npu"]

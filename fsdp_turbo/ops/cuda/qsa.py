# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CUDA backend of the Qwen4-Exp QSA token-selection operator.

Registers the fused Triton implementation from
``fsdp_turbo.ops.triton.qsa_indexer``; that implementation falls back to the
exact eager reference on its own when a layout is unsupported.
"""

from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.ops.triton.qsa_indexer import qsa_selected_token_mask_triton


@register_op("qsa_selected_token_mask", "cuda")
def qsa_selected_token_mask_cuda(*args, **kwargs):
    """Run QSA token selection with the Triton fast path."""
    return qsa_selected_token_mask_triton(*args, **kwargs)


__all__ = ["qsa_selected_token_mask_cuda"]

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Semantic dispatch wrapper for indexed sparse attention."""

from fsdp_turbo.ops import dispatch_op


def indexed_sparse_attention(
    query,
    key_value,
    attention_sink,
    topk_indices,
    softmax_scale,
    attention_mask=None,
    use_eager=False,
    use_sparse_flash_attn=False,
    compressed_key_value=None,
    compressed_topk_indices=None,
    compress_ratio=1,
    window_size=128,
):
    """Apply differentiable attention over explicitly selected KV positions."""
    device_type = "cpu" if use_eager else None
    return dispatch_op(
        "indexed_sparse_attention",
        query,
        key_value,
        attention_sink,
        topk_indices,
        softmax_scale,
        attention_mask,
        use_sparse_flash_attn,
        compressed_key_value,
        compressed_topk_indices,
        compress_ratio,
        window_size,
        device_type=device_type,
    )


__all__ = ["indexed_sparse_attention"]

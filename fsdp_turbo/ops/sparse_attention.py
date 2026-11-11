# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Semantic dispatch wrapper for indexed sparse attention."""

import torch

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
    layout="BSND",
    cu_seqlens=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
):
    """Apply differentiable attention over explicitly selected KV positions.

    For packed TND the optional ``cu_seqlens_ori_kv`` / ``cu_seqlens_cmp_kv``
    carry a KV prefix longer than the local q segment; omitting them keeps the
    original behaviour where both follow ``cu_seqlens``.
    """
    if isinstance(query, torch.Tensor) and query.numel() == 0:
        # Free guard: an empty query is a caller bug the kernels cannot report.
        raise ValueError("indexed_sparse_attention: empty query is not supported")
    device_type = "cpu" if use_eager else None
    # Optional: keep the 13-argument positional contract for existing callers.
    extra = {}
    if cu_seqlens_ori_kv is not None:
        extra["cu_seqlens_ori_kv"] = cu_seqlens_ori_kv
    if cu_seqlens_cmp_kv is not None:
        extra["cu_seqlens_cmp_kv"] = cu_seqlens_cmp_kv
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
        layout,
        cu_seqlens,
        device_type=device_type,
        **extra,
    )


__all__ = ["indexed_sparse_attention", "sparse_flash_attention"]


# rightDownCausal: the causal triangle anchored at the bottom right, matching
# the NPU operator's sparse_mode=3.
SFA_SPARSE_MODE_CAUSAL = 3


def sparse_flash_attention(
    query,
    key,
    value,
    topk_indices,
    softmax_scale,
    rope_head_dim=0,
    query_rope=None,
    key_rope=None,
    sparse_mode=SFA_SPARSE_MODE_CAUSAL,
):
    """Attend over the indexer's selected keys without building a dense mask.

    query is the absorbed MLA query [B, S, N, kv_lora_rank]; key and value are
    the shared compressed latent [B, KV, 1, kv_lora_rank]; topk_indices is the
    indexer's [B, S, topk] with -1 marking unused slots.
    """
    return dispatch_op(
        "sparse_flash_attention",
        query,
        key,
        value,
        topk_indices,
        softmax_scale,
        rope_head_dim=rope_head_dim,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_mode=sparse_mode,
    )

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""PyTorch fallback for indexed sparse attention."""

import torch

from fsdp_turbo.ops.registry import register_op


@register_op("indexed_sparse_attention", "cpu")
def indexed_sparse_attention_torch(
    query: torch.Tensor,
    key_value: torch.Tensor,
    attention_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    attention_mask: torch.Tensor | None = None,
    use_sparse_flash_attn: bool = False,
    compressed_key_value: torch.Tensor | None = None,
    compressed_topk_indices: torch.Tensor | None = None,
    compress_ratio: int = 1,
    window_size: int = 128,
    layout: str = "BSND",
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute indexed attention with an additional normalization sink."""
    del use_sparse_flash_attn, compress_ratio, window_size, cu_seqlens
    if layout.upper() != "BSND":
        raise NotImplementedError("The PyTorch indexed-attention fallback only supports BSND.")
    if compressed_key_value is not None:
        if compressed_topk_indices is None:
            raise ValueError("compressed_topk_indices is required with compressed_key_value")
        compressed_topk_indices = torch.where(
            compressed_topk_indices >= 0,
            compressed_topk_indices + key_value.size(1),
            compressed_topk_indices,
        )
        topk_indices = torch.cat((topk_indices, compressed_topk_indices), dim=-1)
        key_value = torch.cat((key_value, compressed_key_value), dim=1)
    batch_size, sequence_length, _, head_dim = query.shape
    topk_size = topk_indices.shape[-1]
    topk_indices = topk_indices.to(query.device)
    safe_indices = topk_indices.clamp_min(0).long()
    expanded_key_value = key_value.unsqueeze(1).expand(-1, sequence_length, -1, -1)
    selected_key_value = torch.gather(
        expanded_key_value,
        2,
        safe_indices.unsqueeze(-1).expand(batch_size, sequence_length, topk_size, head_dim),
    )
    attention_scores = torch.einsum("bshd,bskd->bshk", query, selected_key_value).float() * softmax_scale
    attention_scores = attention_scores.masked_fill(topk_indices.unsqueeze(2) < 0, -torch.inf)
    if attention_mask is not None and key_value.size(1) == attention_mask.size(1):
        selected_key_mask = torch.gather(attention_mask.bool(), 1, safe_indices.flatten(1)).view(
            batch_size, sequence_length, topk_size
        )
        attention_scores = attention_scores.masked_fill(~selected_key_mask.unsqueeze(2), -torch.inf)

    # NPU cat may read an expanded zero-stride view as an invalid GM address for
    # long sequences. Materialize both inputs before concatenation so the eager
    # fallback presents dense, contiguous storage to the device kernel.
    sink_scores = attention_sink.float().view(1, 1, -1, 1).expand(batch_size, sequence_length, -1, -1).contiguous()
    attention_probabilities = torch.softmax(torch.cat((attention_scores.contiguous(), sink_scores), dim=-1), dim=-1)[
        ..., :-1
    ]
    attention_output = torch.einsum(
        "bshk,bskd->bshd",
        attention_probabilities.to(selected_key_value.dtype),
        selected_key_value,
    )
    return attention_output.to(query.dtype)


@register_op("sparse_flash_attention", "cpu")
def sparse_flash_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    rope_head_dim: int = 0,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    sparse_mode: int = 3,
) -> torch.Tensor:
    """Gather-based reference for ``npu_sparse_flash_attention`` (MLA-absorb DSA).

    query is the absorbed query [B, S, N, D]; key and value are the shared latent
    [B, KV, 1, D]; topk_indices is [B, S, topk] with -1 marking unused slots.
    The indices already encode causality and padding, so ``sparse_mode`` needs
    no separate handling here.
    """
    del sparse_mode
    batch_size, sequence_length, _, head_dim = query.shape
    kv_length = key.shape[1]
    topk_size = topk_indices.shape[-1]

    safe_indices = topk_indices.clamp(0, kv_length - 1).long()
    selected_key = torch.gather(
        key.squeeze(2).unsqueeze(1).expand(-1, sequence_length, -1, -1),
        2,
        safe_indices.unsqueeze(-1).expand(batch_size, sequence_length, topk_size, head_dim),
    )
    selected_value = torch.gather(
        value.squeeze(2).unsqueeze(1).expand(-1, sequence_length, -1, -1),
        2,
        safe_indices.unsqueeze(-1).expand(batch_size, sequence_length, topk_size, head_dim),
    )

    attention_scores = torch.einsum("bsnd,bstd->bsnt", query.float(), selected_key.float()) * softmax_scale
    if rope_head_dim != 0:
        if query_rope is None or key_rope is None:
            raise ValueError(f"rope_head_dim={rope_head_dim} but no rope tensors were given.")
        selected_key_rope = torch.gather(
            key_rope.squeeze(2).unsqueeze(1).expand(-1, sequence_length, -1, -1),
            2,
            safe_indices.unsqueeze(-1).expand(batch_size, sequence_length, topk_size, rope_head_dim),
        )
        attention_scores += (
            torch.einsum("bsnr,bstr->bsnt", query_rope.float(), selected_key_rope.float()) * softmax_scale
        )
    attention_scores = attention_scores.masked_fill(topk_indices.unsqueeze(2) < 0, -torch.inf)

    attention_probabilities = torch.softmax(attention_scores, dim=-1)
    attention_output = torch.einsum(
        "bsnt,bstd->bsnd",
        attention_probabilities.to(selected_value.dtype),
        selected_value,
    )
    return attention_output.to(query.dtype)


__all__ = ["indexed_sparse_attention_torch", "sparse_flash_attention_torch"]

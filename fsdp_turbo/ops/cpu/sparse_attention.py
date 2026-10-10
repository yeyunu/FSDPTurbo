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

    sink_scores = attention_sink.float().view(1, 1, -1, 1).expand(batch_size, sequence_length, -1, -1)
    attention_probabilities = torch.softmax(torch.cat((attention_scores, sink_scores), dim=-1), dim=-1)[..., :-1]
    attention_output = torch.einsum(
        "bshk,bskd->bshd",
        attention_probabilities.to(selected_key_value.dtype),
        selected_key_value,
    )
    return attention_output.to(query.dtype)


__all__ = ["indexed_sparse_attention_torch"]

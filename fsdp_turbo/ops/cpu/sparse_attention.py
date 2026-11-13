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
    cu_seqlens_ori_kv: torch.Tensor | None = None,
    cu_seqlens_cmp_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute indexed attention with a sink and accumulate KV gradients by row.

    Index the original ``[batch, kv_length, head_dim]`` storage directly so
    backward never materializes a ``[batch, query_length, kv_length, head_dim]``
    gradient. Repeated selected positions accumulate into the same KV row.
    """
    del use_sparse_flash_attn, window_size, cu_seqlens
    del cu_seqlens_ori_kv, cu_seqlens_cmp_kv
    if layout.upper() != "BSND":
        raise NotImplementedError("The PyTorch indexed-attention fallback only supports BSND.")
    if compressed_key_value is not None:
        if compressed_topk_indices is None:
            compressed_length = compressed_key_value.size(1)
            compressed_topk_indices = torch.arange(compressed_length, device=query.device)
            compressed_topk_indices = compressed_topk_indices.view(1, 1, -1).expand(query.size(0), query.size(1), -1)
            causal_threshold = (torch.arange(query.size(1), device=query.device) + 1) // compress_ratio
            compressed_topk_indices = compressed_topk_indices.masked_fill(
                compressed_topk_indices >= causal_threshold.view(1, -1, 1),
                -1,
            )
        compressed_topk_indices = torch.where(
            compressed_topk_indices >= 0,
            compressed_topk_indices + key_value.size(1),
            compressed_topk_indices,
        )
        topk_indices = torch.cat((topk_indices, compressed_topk_indices), dim=-1)
        key_value = torch.cat((key_value, compressed_key_value), dim=1)
    batch_size, sequence_length = query.shape[:2]
    topk_size = topk_indices.shape[-1]
    topk_indices = topk_indices.to(query.device)
    safe_indices = topk_indices.clamp_min(0).long()
    batch_indices = torch.arange(batch_size, device=key_value.device).view(batch_size, 1, 1)
    selected_key_value = key_value[batch_indices, safe_indices]
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


def sparse_attention_selected_mask(indices: torch.Tensor, key_length: int) -> torch.Tensor:
    """Decode -1-padded key indices into a shared boolean mask."""
    # Use an extra column so padding cannot overwrite a valid key.
    selected = torch.zeros((*indices.shape[:-1], key_length + 1), device=indices.device, dtype=torch.bool)
    selected.scatter_(-1, indices.long().masked_fill(indices < 0, key_length), True)
    return selected[..., :key_length]


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
    *,
    cu_seqlens: torch.Tensor | None = None,
    return_softmax_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather-based reference for ``npu_sparse_flash_attention`` (MLA-absorb DSA).

    query is the absorbed query [B, S, N, D]; key and value are the shared latent
    [B, KV, 1, D]; topk_indices is [B, S, topk] with -1 marking unused slots.
    The indices already encode causality and padding, so ``sparse_mode`` needs
    no separate handling here. Packed inputs use document-local indices.
    """
    if cu_seqlens is not None:
        boundaries = cu_seqlens.tolist()
        inputs = (query, key, value, topk_indices)
        parts = [
            sparse_flash_attention_torch(
                *(tensor[start:end].unsqueeze(0) for tensor in inputs),
                softmax_scale,
                rope_head_dim,
                None if query_rope is None else query_rope[start:end].unsqueeze(0),
                None if key_rope is None else key_rope[start:end].unsqueeze(0),
                sparse_mode,
                return_softmax_stats=return_softmax_stats,
            )
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        if return_softmax_stats:
            return tuple(torch.cat(tensors, dim=1).squeeze(0) for tensors in zip(*parts))
        return torch.cat(parts, dim=1).squeeze(0)

    if return_softmax_stats:
        # The dense reference avoids materializing an [B,S,topk,D] gather
        # for each latent and retains FP32 weights for the KL reference.
        selected = sparse_attention_selected_mask(topk_indices, key.shape[1])
        scores = torch.einsum("bshd,bnd->bshn", query.float(), key.squeeze(2).float())
        if rope_head_dim:
            if query_rope is None or key_rope is None:
                raise ValueError(f"rope_head_dim={rope_head_dim} but no rope tensors were given.")
            scores = scores + torch.einsum("bshr,bnr->bshn", query_rope.float(), key_rope.squeeze(2).float())
        scores = (scores * softmax_scale).masked_fill(~selected.unsqueeze(2), -torch.inf)
        maximum = scores.amax(-1)
        denominator = (scores - maximum.unsqueeze(-1)).exp().sum(-1)
        output = torch.einsum("bshn,bnd->bshd", scores.softmax(-1), value.squeeze(2).float())
        return output.to(query.dtype), maximum.detach(), denominator.detach()

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

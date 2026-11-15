# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""PyTorch reference for shared-KV sparse MLA and its selected-key mask."""

import torch

from fsdp_turbo.ops.registry import register_op


def sparse_mla_selected_mask(indices: torch.Tensor, key_length: int) -> torch.Tensor:
    """Decode packed -1-padded key indices into a shared boolean mask."""
    batch, sequence, _, _ = indices.shape
    # An extra sentinel column prevents padding from overwriting a valid key.
    selected = torch.zeros((batch, sequence, key_length + 1), device=indices.device, dtype=torch.bool)
    keys = indices.squeeze(2).long()
    selected.scatter_(-1, keys.masked_fill(keys < 0, key_length), True)
    return selected[..., :key_length]


@register_op("sparse_mla", "cpu")
def sparse_mla_cpu(query: torch.Tensor, key: torch.Tensor, query_rope: torch.Tensor,
                   key_rope: torch.Tensor, indices: torch.Tensor, scale: float,
                   cu_seqlens: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference BSND/TND shared K=V MLA with separate RoPE and softmax stats."""
    if cu_seqlens is not None:
        boundaries = cu_seqlens.tolist()
        outputs = [
            sparse_mla_cpu(*(tensor[start:end].unsqueeze(0) for tensor in (query, key, query_rope, key_rope, indices)), scale)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        return tuple(torch.cat([part.squeeze(0) for part in tensors]) for tensors in zip(*outputs))
    selected = sparse_mla_selected_mask(indices, key.shape[1])
    logits = (torch.einsum("bshd,bnd->bshn", query.float(), key.squeeze(2).float())
               + torch.einsum("bshr,bnr->bshn", query_rope.float(), key_rope.squeeze(2).float())) * scale
    logits = logits.masked_fill(~selected.unsqueeze(2), -torch.inf)
    maximum = logits.amax(-1)
    denominator = (logits - maximum.unsqueeze(-1)).exp().sum(-1)
    probabilities = logits.softmax(-1)
    output = torch.einsum("bshn,bnd->bshd", probabilities, key.squeeze(2).float()).to(query.dtype)
    return output, maximum.detach(), denominator.detach()


__all__ = ["sparse_mla_selected_mask", "sparse_mla_cpu"]

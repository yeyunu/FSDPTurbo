# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Portable selected-key gradient for the lightning indexer."""

import torch

from fsdp_turbo.ops.registry import register_op


@register_op("sparse_indexer_gradient", "cpu")
def sparse_indexer_gradient_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    target: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    """Return a zero-valued scalar with the gradient of the summed, detached-teacher KL."""
    batch = torch.arange(query.size(0), device=query.device)[:, None, None]
    total = query.new_zeros((), dtype=torch.float32)
    for q, w, idx, teacher in zip(
        query.split(32, 1),
        weights.split(32, 1),
        indices.split(32, 1),
        target.split(32, 1),
    ):
        valid = idx >= 0
        keys = key[batch, idx.clamp_min(0).long()]
        logits = torch.einsum("bshd,bskd->bshk", q.float(), keys.float()).relu()
        logits = (logits * w.float().unsqueeze(-1)).sum(2).masked_fill(~valid, -torch.inf)
        logits = torch.where(valid.any(-1, keepdim=True), logits, 0.0)
        log_probs = logits.log_softmax(-1).masked_fill(~valid, 0)
        total = total - (torch.nn.functional.normalize(teacher, p=1, dim=-1) * log_probs).sum()
    return total - total.detach()

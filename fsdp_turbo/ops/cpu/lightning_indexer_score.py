# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Eager reference for the DSA (lightning) indexer scoring operator.

Mirrors the fused Triton kernel ``fsdp_turbo.ops.triton.lightning_indexer_score``:
``out[b, s, p] = sum_h w[b, s, h] * relu(scale * q[b, s, h, :] . k[b, p, :])``
with masked-out pool candidates driven to float32 ``min`` so the result feeds
straight into ``topk``.
"""

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.registry import register_op


@register_op("lightning_indexer_score", "cpu")
def lightning_indexer_score_torch(
    q: torch.Tensor,
    pool_keys: torch.Tensor,
    weights: torch.Tensor,
    valid_candidates: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute the pooled-key indexer scores: q [B,S,NH,D], pool_keys [B,P,D]."""
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # [B, S, NH, P] -> relu -> head-weighted sum -> [B, S, P]
    scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores * scale)
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

    if valid_candidates is not None:
        index_scores = torch.where(valid_candidates, index_scores, torch.finfo(index_scores.dtype).min)
    return index_scores


__all__ = ["lightning_indexer_score_torch"]

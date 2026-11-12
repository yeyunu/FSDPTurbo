# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Portable reference for GLM53's equal-length selected-support Indexer objective."""

import torch

from fsdp_turbo.ops.registry import register_op


@register_op("glm_lightning_indexer", "cpu")
def lightning_indexer_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score ReLU query/key products, then sum weighted heads and apply causal top-k."""
    scores = torch.einsum("bshd,btd->bsht", query.float(), key[:, :, 0].float()).relu()
    scores = (scores * weights.float()[..., None]).sum(2)
    positions = torch.arange(scores.shape[1], device=query.device)
    keys = torch.arange(scores.shape[-1], device=query.device)
    scores = scores.masked_fill(keys[None, None, :] > positions[None, :, None], float("-inf"))
    selected_scores, indices = scores.topk(min(topk, scores.shape[-1]), dim=-1)
    indices = indices.masked_fill(~torch.isfinite(selected_scores), -1)
    return indices.to(torch.int32), selected_scores


@register_op("glm_indexer_kl_loss", "cpu")
def indexer_kl_loss_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    maximum: torch.Tensor,
    total: torch.Tensor,
    scale: float,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
) -> torch.Tensor:
    """Recompute the detached teacher on selected support and differentiate Indexer KL.

    This small-tensor reference follows the LLM reference's epsilon convention;
    it is a functional fallback, not NPU numerical-equivalence certification.
    Queries with no valid selected keys contribute zero to the batch-token mean.
    """
    batch, length, heads, _ = query.shape
    safe = indices.long().clamp(0, key.shape[1] - 1)
    positions = torch.arange(length, device=query.device)
    valid = (indices >= 0) & (indices < key.shape[1]) & (indices <= positions[None, :, None])
    row_valid = valid.any(-1, keepdim=True)
    with torch.no_grad():
        teacher = torch.einsum("bshd,btd->bsht", query.float(), key[:, :, 0].float())
        teacher += torch.einsum("bshd,btd->bsht", query_rope.float(), key_rope[:, :, 0].float())
        teacher = teacher.gather(-1, safe[:, :, None, :].expand(-1, -1, heads, -1)) * scale
        teacher = teacher.masked_fill(~valid[:, :, None, :], float("-inf"))
        teacher = torch.where(row_valid.unsqueeze(2), teacher, 0.0)
        target = teacher.softmax(-1).mean(2).masked_fill(~valid, 0.0)
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-9)
    index_scores = torch.einsum("bshd,btd->bsht", query_index.float(), key_index[:, :, 0].float()).relu()
    selected = (index_scores * weights.float()[..., None]).sum(2).gather(-1, safe)
    selected = selected.masked_fill(~valid, float("-inf"))
    selected = torch.where(row_valid, selected, 0.0)
    prediction = selected.softmax(-1)
    epsilon = 1e-9
    normal_target = target / (target.sum(-1, keepdim=True) + epsilon)
    return ((normal_target.clamp_min(epsilon).log() - (prediction + epsilon).log()) * target).sum() / (batch * length)

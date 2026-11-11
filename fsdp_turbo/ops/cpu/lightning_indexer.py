# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""PyTorch fallback for the fused lightning indexer."""

import torch

from fsdp_turbo.ops.registry import register_op


def _candidate_blocks(score, visible, topk_blocks, block_size):
    """Top candidate blocks per row, mirroring `select_candidate_blocks`."""
    batch_size, sequence_length, key_length = score.shape
    padded = torch.nn.functional.pad(score, (0, -key_length % block_size), value=-torch.inf)
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = block_scores.shape[-1]
    last = (visible - 1) // block_size
    block_scores = block_scores.masked_fill(torch.arange(num_blocks) == last.unsqueeze(-1), torch.inf)
    top = block_scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    lengths = keep.sum(dim=-1).reshape(batch_size * sequence_length, 1).int()
    indices = torch.full((batch_size * sequence_length, 1, num_blocks), -1, dtype=torch.int32)
    flat_keep = keep.reshape(batch_size * sequence_length, num_blocks)
    for row in range(batch_size * sequence_length):
        chosen = flat_keep[row].nonzero().flatten()
        indices[row, 0, : chosen.numel()] = chosen.int()
    return indices, lengths


def _candidate_mask(score, candidate_block_indices, candidate_block_length, block_size):
    """Boolean key mask for the blocks a row selected, as the consumer expects."""
    batch_size, sequence_length, key_length = score.shape
    flat_indices = candidate_block_indices.reshape(batch_size * sequence_length, -1)
    lengths = candidate_block_length.reshape(batch_size * sequence_length).long()
    num_blocks = (key_length + block_size - 1) // block_size
    block_ids = torch.arange(num_blocks)
    mask = torch.zeros(batch_size * sequence_length, key_length, dtype=torch.bool)
    for row in range(batch_size * sequence_length):
        chosen = flat_indices[row, : lengths[row]]
        chosen = chosen[chosen >= 0]
        if chosen.numel():
            mask[row] = torch.isin(block_ids, chosen).repeat_interleave(block_size)[:key_length]
    return mask.reshape(batch_size, sequence_length, key_length)


@register_op("fused_lightning_indexer", "cpu")
def fused_lightning_indexer_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weights: torch.Tensor,
    *,
    topk: int,
    compress_ratio: int,
    index_offset: int,
    attention_mask: torch.Tensor | None,
    page_size: int,
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
    candidate_block_indices: torch.Tensor | None = None,
    candidate_block_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Causal compressed top-k, the eager semantics of QuantLightningIndexer.

    ``query`` is ``[B, S, H, D]``, ``key`` is ``[B, K, D]``, ``head_weights`` is
    ``[B, S, H]``. Returns ``[B, S, topk]`` with valid indices before ``-1``;
    with candidate blocks it also returns block ids and lengths.
    """
    del page_size
    batch_size, sequence_length, _, _ = query.shape
    score = torch.einsum("bshd,btd->bsht", query.float(), key.float())
    score = (score.relu_() * head_weights.float().unsqueeze(-1)).sum(dim=2)
    visible = (torch.arange(1, sequence_length + 1) // compress_ratio).unsqueeze(0).unsqueeze(-1)
    score = score.masked_fill(torch.arange(key.shape[1]) >= visible, -torch.inf)
    valid_rows = (
        torch.ones(batch_size, sequence_length, dtype=torch.bool)
        if attention_mask is None
        else attention_mask.to(torch.bool)
    )
    score = score.masked_fill(~valid_rows.unsqueeze(-1), -torch.inf)
    candidates = candidate_lengths = None
    if candidate_block_indices is not None:
        block_mask = _candidate_mask(score, candidate_block_indices, candidate_block_length, candidate_block_size)
        score = score.masked_fill(~block_mask, -torch.inf)
    elif candidate_topk_blocks > 0:
        candidates, candidate_lengths = _candidate_blocks(
            score,
            visible.squeeze(-1).expand(batch_size, sequence_length),
            candidate_topk_blocks,
            candidate_block_size,
        )
    width = min(topk, key.shape[1])
    selected = score.topk(width, dim=-1, sorted=False).indices.sort(dim=-1).values
    selected = torch.where(selected < visible, selected + index_offset, -1)
    selected = selected.masked_fill(~valid_rows.unsqueeze(-1), -1)
    order = torch.argsort(selected < 0, dim=-1, stable=True)
    selected = selected.gather(-1, order)
    indices = torch.nn.functional.pad(selected, (0, topk - width), value=-1)
    return indices.int(), candidates, candidate_lengths

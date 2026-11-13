# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""PyTorch references for lightning index selection and training KL loss."""

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


@register_op("lightning_indexer_topk", "cpu")
def lightning_indexer_topk_cpu(
    query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, topk: int, cu_seqlens: torch.Tensor | None = None
) -> torch.Tensor:
    """Return document-local causal top-k indices for BSND or unpadded TND."""
    if cu_seqlens is not None:
        boundaries = cu_seqlens.tolist()
        return torch.cat(
            [
                lightning_indexer_topk_cpu(
                    query[start:end].unsqueeze(0), key[start:end].unsqueeze(0), weights[start:end].unsqueeze(0), topk
                ).squeeze(0)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
        )
    scores = torch.einsum("bshd,bnd->bshn", query.float(), key.float()).relu()
    scores = torch.einsum("bsh,bshn->bsn", weights.float(), scores)
    sequence, key_length = query.shape[1], key.shape[1]
    visible = (
        torch.arange(sequence, device=query.device)[:, None] >= torch.arange(key_length, device=query.device)[None, :]
    )
    scores = scores.masked_fill(~visible, -torch.inf)
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., : min(topk, key_length)]
    selected = visible[None, :, :].expand_as(scores).gather(-1, indices)
    indices = indices.masked_fill(~selected, -1).to(torch.int32)
    return torch.nn.functional.pad(indices, (0, topk - indices.shape[-1]), value=-1).unsqueeze(2)


@register_op("lightning_indexer_kl_loss", "cpu")
def lightning_indexer_kl_loss_cpu(
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    indices: torch.Tensor,
    maximum: torch.Tensor,
    denominator: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean sparse teacher KL for BSND/TND; only indexer features receive gradients."""
    from fsdp_turbo.ops.cpu.sparse_attention import sparse_attention_selected_mask

    if cu_seqlens is not None:
        boundaries = cu_seqlens.tolist()
        tensors = (query_index, key_index, weights, query, key, query_rope, key_rope, indices, maximum, denominator)
        return (
            sum(
                lightning_indexer_kl_loss_cpu(*(tensor[start:end].unsqueeze(0) for tensor in tensors), scale)
                * (end - start)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            )
            / query_index.shape[0]
        )
    selected = sparse_attention_selected_mask(indices.squeeze(-2), key.shape[1])
    with torch.no_grad():
        teacher_scores = (
            torch.einsum("bshd,bnd->bshn", query.float(), key.squeeze(2).float())
            + torch.einsum("bshr,bnr->bshn", query_rope.float(), key_rope.squeeze(2).float())
        ) * scale
        teacher = teacher_scores.masked_fill(~selected.unsqueeze(2), -torch.inf).softmax(-1).mean(2)
    scores = torch.einsum("bshd,bnd->bshn", query_index.float(), key_index.float()).relu()
    scores = torch.einsum("bsh,bshn->bsn", weights.float(), scores)
    log_probability = scores.masked_fill(~selected, -torch.inf).log_softmax(-1).masked_fill(~selected, 0)
    return (teacher * (teacher.clamp_min(1e-30).log() - log_probability)).sum(-1).mean()

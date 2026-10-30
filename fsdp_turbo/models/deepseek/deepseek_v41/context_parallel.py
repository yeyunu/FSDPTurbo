# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Eager, contiguous KV-AllGather training for DeepSeek-V4.1.

Q and hidden states stay local. Original KV, compressed KV and index K use
global sequence coordinates. No decode cache or fused operator is used here.
Fused flags are checked by FSDPTurboConfig; the model builder rejects MTP/DSpark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fsdp_turbo.distributed.context_parallel.kv_allgather.utils import kv_allgather
from fsdp_turbo.ops.sparse_attention import indexed_sparse_attention
from .indexer import _compute_index_score
from .layers import apply_rotary_emb
from .runtime import ContextParallelAttentionState

if TYPE_CHECKING:
    from .adapter import DeepseekV41ForCausalLMAdapter
    from .attention import Attention
    from .image_processor import ImageInput
    from .model_args import ModelArgs


def validate_sequence_length(length: int, cp_size: int, args: ModelArgs) -> None:
    """Require equal contiguous shards aligned with each layer's compression ratio."""
    if cp_size < 1 or length <= 0 or length > args.max_seq_len or length % cp_size:
        raise ValueError("DeepSeek-V4.1 CP requires a non-empty sequence <= max_seq_len divisible by CP size.")
    if any(ratio > 1 and (length // cp_size) % ratio for ratio in args.compress_ratios[: args.n_layers]):
        raise ValueError("DeepSeek-V4.1 CP local sequence length must be divisible by every compression ratio.")


def prepare_cp_inputs(
    adapter: DeepseekV41ForCausalLMAdapter,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor | None,
    images: list[list[ImageInput]] | None,
    token_types: torch.Tensor | None,
) -> tuple[torch.Tensor, slice, torch.Tensor, torch.Tensor | None]:
    """Validate all ranks before data collectives and return Engram hash inputs.

    Text inputs are contiguous shards. Image batches retain complete token IDs,
    image coordinates and token types until fusion in the backbone. Labels are
    always complete and unshifted. Returns the hash input ids, the local row
    slice into their hashed output, the complete padding mask for labels and a
    hash mask aligned one-to-one with the hash inputs (``None`` when Engram
    hashing is disabled). Invalid rank metadata raises before collectives that
    require matching sequence shapes.

    For text shards the hash inputs carry only the left neighbour's n-gram halo
    instead of the fully gathered sequence, so the hash mask prepends a halo
    column rather than spanning the complete sequence.
    """
    group = adapter._kv_cp_group
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    # A fixed-size collective also detects unequal shards before all_gather(KV).
    shape = tuple(input_ids.shape)
    batch, length = shape if len(shape) == 2 else (-1, -1)
    complete = images is not None
    total = length if complete else length * size
    local_length = total // size
    valid = batch > 0 and length > 0 and total % size == 0
    valid = valid and (token_types is None if not complete else token_types.shape == input_ids.shape)
    if complete:
        valid = valid and len(images) == batch
    if attention_mask is not None:
        valid = valid and attention_mask.shape == input_ids.shape
        valid = valid and bool(((attention_mask == 0) | (attention_mask == 1)).all())
        if complete:
            valid = valid and not bool((~attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()).any())
    if position_ids is not None:
        start = 0 if complete else rank * max(length, 0)
        expected = torch.arange(start, start + max(length, 0), device=input_ids.device)
        valid = valid and position_ids.ndim == 2 and position_ids.shape[0] in (1, batch)
        valid = valid and position_ids.shape[-1] == length and bool((position_ids == expected).all())
    valid = valid and (labels is None or tuple(labels.shape) == (batch, total))
    status = torch.tensor(
        [batch, length, int(valid), int(labels is not None), int(attention_mask is not None), int(complete)],
        device=input_ids.device,
    )
    statuses = [torch.empty_like(status) for _ in range(size)]
    dist.all_gather(statuses, status, group=group)
    if any(not torch.equal(item, status) for item in statuses) or not valid:
        raise ValueError(
            "DeepSeek-V4.1 CP requires equal text shards or complete image batches, global contiguous "
            "position_ids and complete unshifted labels on every rank."
        )
    validate_sequence_length(total, size, adapter.model_args)
    if not adapter.training:
        raise ValueError("DeepSeek-V4.1 KV-AllGather currently supports training mode only.")
    local_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    if complete:
        full_mask = local_mask
    else:
        masks = [torch.empty_like(local_mask) for _ in range(size)]
        dist.all_gather(masks, local_mask.contiguous(), group=group)
        full_mask = torch.cat(masks, dim=1)
    if not complete and bool((~full_mask[:, :-1] & full_mask[:, 1:]).any()):
        raise ValueError("DeepSeek-V4.1 CP supports right padding only; left padding and mask holes are unsupported.")
    hash_state = adapter.model.engram_hash
    if hash_state is None:
        return input_ids, slice(None), full_mask, None
    if complete:
        # Image batches carry the complete token sequence on every rank, so no
        # boundary exchange is needed; token_types masks out image spans.
        hash_mask = local_mask if token_types is None else local_mask & (token_types < 0)
        return input_ids, slice(rank * local_length, (rank + 1) * local_length), full_mask, hash_mask
    # An n-gram ending in this shard looks back at most max_ngram_size-1 tokens,
    # so exchanging the left neighbour's halo replaces the full sequence
    # all-gather. Right-only padding was validated above, which makes every
    # halo token from the left neighbour valid; rank 0 fills its halo with the
    # Engram pad id and keeps it unmasked, hashing like a regular pad token.
    halo_size = hash_state.layout.max_ngram_size - 1
    hash_input_ids, hash_slice = exchange_cp_hash_inputs_halo(
        input_ids, halo_size, adapter.model_args.engram_pad_id, group
    )
    halo_mask = input_ids.new_ones((batch, halo_size), dtype=torch.bool)
    hash_mask = torch.cat((halo_mask, local_mask), dim=1)
    return hash_input_ids, hash_slice, full_mask, hash_mask


def exchange_cp_hash_inputs_halo(
    input_ids: torch.Tensor, halo_size: int, pad_id: int, group: dist.ProcessGroup
) -> tuple[torch.Tensor, slice]:
    """Exchange only the left-neighbour boundary tokens before Engram n-gram hashing.

    Each rank sends the last ``halo_size`` ids of its shard to rank + 1 in one
    variable all-to-all and prepends the received halo to its local shard.
    Rank 0 has no left neighbour and fills its halo with ``pad_id``, the same
    token ``NgramHashState`` substitutes when a look-back crosses the start of
    the sequence. Hashing ``[halo | local]`` and slicing the local rows is
    therefore bitwise equal to hashing the fully gathered sequence, while the
    exchanged payload stays at ``halo_size`` ids regardless of CP size.

    Args:
        input_ids: Local contiguous shard, shape ``[batch, local_length]``.
        halo_size: Boundary token count, ``max_ngram_size - 1``.
        pad_id: Raw Engram pad token id; the hash state maps it to its compressed id.
        group: Contiguous CP group whose group rank equals the shard order.

    Returns:
        ``(hash_input_ids, local_slice)`` with ``hash_input_ids`` shaped
        ``[batch, halo_size + local_length]`` and the slice selecting local rows.
    """
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    batch, length = input_ids.shape
    send_sizes = [0] * size
    recv_sizes = [0] * size
    if rank + 1 < size:
        send_sizes[rank + 1] = batch * halo_size
    if rank > 0:
        recv_sizes[rank - 1] = batch * halo_size
    if rank + 1 < size:
        boundary = input_ids[:, -halo_size:].reshape(-1).contiguous()
    else:
        boundary = input_ids.new_empty(0)
    halo_flat = input_ids.new_empty(sum(recv_sizes))
    dist.all_to_all_single(
        halo_flat,
        boundary,
        output_split_sizes=recv_sizes,
        input_split_sizes=send_sizes,
        group=group,
    )
    if rank == 0:
        halo = input_ids.new_full((batch, halo_size), pad_id)
    else:
        halo = halo_flat.view(batch, halo_size)
    return torch.cat((halo, input_ids), dim=1), slice(halo_size, halo_size + length)


def _topk_by_position(scores: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select CP keys by score, breaking exact ties by ascending global position."""
    indices = scores.argsort(dim=-1, descending=True, stable=True)[..., :count]
    return scores.gather(-1, indices), indices


def _select_candidate_blocks(
    logits: torch.Tensor, visible: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Select CP candidate blocks with deterministic ties and discard masked blocks."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    newest = (visible - 1) // block_size
    scores = scores.masked_fill(torch.arange(scores.size(-1), device=logits.device) == newest, torch.inf)
    values, indices = _topk_by_position(scores, min(topk_blocks, scores.size(-1)))
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _indexer_forward(
    indexer, hidden_states, query_latent, latent, rotary_freqs, start, total, group, index_k, candidates
):
    ratio, rope_head_dim = indexer.compress_ratio, indexer.rope_head_dim
    length = hidden_states.shape[1]
    if indexer.owns_k:
        k = indexer.k_norm(indexer.wk(latent))
        apply_rotary_emb(k[..., -rope_head_dim:], rotary_freqs[start : start + length : ratio])
        index_k = kv_allgather(k, group, gather_dim=1)
    q = indexer.wq_b(query_latent).unflatten(-1, (indexer.n_local_heads, indexer.index_head_dim))
    apply_rotary_emb(q[..., -rope_head_dim:], rotary_freqs[start : start + length])
    weights = indexer.weights_proj(hidden_states) * (indexer.softmax_scale * indexer.n_heads**-0.5)
    scores = _compute_index_score(q, index_k, weights)
    visible = (torch.arange(start + 1, start + length + 1, device=hidden_states.device) // ratio).unsqueeze(-1)
    scores.masked_fill_(torch.arange(total // ratio, device=hidden_states.device) >= visible, -torch.inf)
    if indexer.is_candidate_source:
        candidates = _select_candidate_blocks(
            scores, visible, indexer.candidate_topk_blocks, indexer.candidate_block_size
        )
    elif indexer.uses_candidates:
        scores = scores.masked_fill(~candidates, -torch.inf)
    selected_scores, indices = _topk_by_position(scores, min(indexer.index_topk, total // ratio))
    indices = torch.where(selected_scores > -torch.inf, indices + total, -1)
    return indices.sort(dim=-1).values.int(), index_k, candidates


def compute_context_parallel_attention(
    attention: Attention,
    hidden_states: torch.Tensor,
    start_pos: int,
    group: dist.ProcessGroup,
    state: ContextParallelAttentionState | None = None,
) -> tuple[torch.Tensor, ContextParallelAttentionState]:
    """Compute local rows and return immutable cross-layer state for checkpoint replay."""
    if start_pos != 0 or not attention.training:
        raise ValueError("DeepSeek-V4.1 CP attention is training-only; decode/cache is unsupported.")
    batch, length, _ = hidden_states.shape
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    start, total = rank * length, size * length
    compress_kv, index_k, topk_idxs, candidates = state if state is not None else ContextParallelAttentionState()
    rope_head_dim = attention.rope_head_dim
    rotary_freqs = attention.freqs_cis[start : start + length]
    query_latent = attention.q_norm(attention.wq_a(hidden_states))
    q = attention.wq_b(query_latent).unflatten(-1, (attention.n_local_heads, attention.head_dim))
    apply_rotary_emb(q[..., -rope_head_dim:], rotary_freqs)
    kv = attention.kv_norm(attention.wkv(hidden_states))
    apply_rotary_emb(kv[..., -rope_head_dim:], rotary_freqs)
    kv = kv_allgather(kv, group, gather_dim=1)
    # Build only local rows, in the same order as the non-CP prefill implementation.
    end = torch.arange(start, start + length, device=hidden_states.device).unsqueeze(1)
    indices = (end - attention.window_size + 1).clamp_min(0) + torch.arange(
        min(total, attention.window_size), device=hidden_states.device
    )
    indices = torch.where(indices > end, -1, indices).int().unsqueeze(0).expand(batch, -1, -1)
    if attention.compress_ratio:
        latent = attention.compressor(hidden_states, 0) if attention.is_kv_source else None
        if attention.is_index_source:
            topk_idxs, index_k, candidates = _indexer_forward(
                attention.indexer,
                hidden_states,
                query_latent,
                latent,
                attention.freqs_cis,
                start,
                total,
                group,
                index_k,
                candidates,
            )
        if latent is not None:
            # Index K was computed from the unrotated latent above. Clone before
            # RoPE so autograd's saved indexer input is never modified in place.
            latent = latent.clone()
            apply_rotary_emb(
                latent[..., -rope_head_dim:], attention.freqs_cis[start : start + length : attention.compress_ratio]
            )
            compress_kv = kv_allgather(latent, group, gather_dim=1)
        kv = torch.cat((kv, compress_kv), dim=1)
        indices = torch.cat((indices, topk_idxs), dim=-1)
    output = indexed_sparse_attention(q, kv, attention.attn_sink, indices, attention.softmax_scale)
    output = output.clone()
    apply_rotary_emb(output[..., -rope_head_dim:], rotary_freqs, True)
    output = output.reshape(batch, length, attention.n_local_groups, -1)
    weight = attention.wo_a.weight.view(attention.n_local_groups, attention.o_lora_rank, -1)
    output = attention.wo_b(torch.einsum("bsgd,grd->bsgr", output, weight).flatten(2))
    return output, ContextParallelAttentionState(compress_kv, index_k, topk_idxs, candidates)

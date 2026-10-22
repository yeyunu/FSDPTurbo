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

from fsdp_turbo.distributed.context_parallel.kv_allgather.utils import kv_allgather
from fsdp_turbo.ops.sparse_attention import indexed_sparse_attention
from .indexer import _compute_index_score, select_candidate_blocks
from .layers import apply_rotary_emb
from .runtime import shared_attn

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
) -> tuple[torch.Tensor, slice]:
    """Validate all ranks before data collectives and return Engram IDs plus a local slice.

    Every CP rank must call this with equal contiguous shards and complete,
    unshifted labels. Invalid or inconsistent inputs raise ValueError. Only
    Engram needs gathered integer IDs; otherwise the input and full slice return.
    """
    group = adapter._kv_cp_group
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    # A fixed-size collective also detects unequal shards before all_gather(KV).
    shape = tuple(input_ids.shape)
    batch, length = shape if len(shape) == 2 else (-1, -1)
    valid = batch > 0 and length > 0 and images is None and token_types is None
    if attention_mask is not None:
        valid = valid and attention_mask.shape == input_ids.shape and bool(attention_mask.bool().all())
    if position_ids is not None:
        expected = torch.arange(rank * max(length, 0), (rank + 1) * max(length, 0), device=input_ids.device)
        valid = valid and position_ids.ndim == 2 and position_ids.shape[0] in (1, batch)
        valid = valid and position_ids.shape[-1] == length and bool((position_ids == expected).all())
    valid = valid and (labels is None or tuple(labels.shape) == (batch, length * size))
    status = torch.tensor([batch, length, int(valid), int(labels is not None)], device=input_ids.device)
    statuses = [torch.empty_like(status) for _ in range(size)]
    dist.all_gather(statuses, status, group=group)
    if any(not torch.equal(item, status) for item in statuses) or not valid:
        raise ValueError(
            "DeepSeek-V4.1 CP requires equal text shards, no padding/images, global contiguous "
            "position_ids and complete unshifted labels on every rank."
        )
    validate_sequence_length(length * size, size, adapter.model_args)
    if not adapter.training:
        raise ValueError("DeepSeek-V4.1 KV-AllGather currently supports training mode only.")
    if adapter.model.engram_hash is None:
        return input_ids, slice(None)
    ids = [torch.empty_like(input_ids) for _ in range(size)]
    dist.all_gather(ids, input_ids.contiguous(), group=group)
    return torch.cat(ids, dim=1), slice(rank * length, (rank + 1) * length)


def _indexer_forward(indexer, hidden_states, query_latent, latent, rotary_freqs, start, total, group):
    ratio, rope_head_dim = indexer.compress_ratio, indexer.rope_head_dim
    length = hidden_states.shape[1]
    if indexer.owns_k:
        k = indexer.k_norm(indexer.wk(latent))
        apply_rotary_emb(k[..., -rope_head_dim:], rotary_freqs[start : start + length : ratio])
        shared_attn.index_k = kv_allgather(k, group, gather_dim=1)
    q = indexer.wq_b(query_latent).unflatten(-1, (indexer.n_local_heads, indexer.index_head_dim))
    apply_rotary_emb(q[..., -rope_head_dim:], rotary_freqs[start : start + length])
    weights = indexer.weights_proj(hidden_states) * (indexer.softmax_scale * indexer.n_heads**-0.5)
    scores = _compute_index_score(q, shared_attn.index_k, weights)
    visible = (torch.arange(start + 1, start + length + 1, device=hidden_states.device) // ratio).unsqueeze(-1)
    scores.masked_fill_(torch.arange(total // ratio, device=hidden_states.device) >= visible, -torch.inf)
    if indexer.is_candidate_source:
        shared_attn.candidates = select_candidate_blocks(
            scores, visible, indexer.candidate_topk_blocks, indexer.candidate_block_size
        )
    elif indexer.uses_candidates:
        scores = scores.masked_fill(~shared_attn.candidates, -torch.inf)
    indices = scores.topk(min(indexer.index_topk, total // ratio), dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(indices < visible, indices + total, -1).int()


def attention_forward(
    attention: Attention, hidden_states: torch.Tensor, start_pos: int, group: dist.ProcessGroup
) -> torch.Tensor:
    """Compute local attention rows with differentiable global KV and index keys."""
    if start_pos != 0 or not attention.training:
        raise ValueError("DeepSeek-V4.1 CP attention is training-only; decode/cache is unsupported.")
    batch, length, _ = hidden_states.shape
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    start, total = rank * length, size * length
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
            shared_attn.topk_idxs = _indexer_forward(
                attention.indexer, hidden_states, query_latent, latent, attention.freqs_cis, start, total, group
            )
        if latent is not None:
            # Index K was computed from the unrotated latent above. Clone before
            # RoPE so autograd's saved indexer input is never modified in place.
            latent = latent.clone()
            apply_rotary_emb(
                latent[..., -rope_head_dim:], attention.freqs_cis[start : start + length : attention.compress_ratio]
            )
            shared_attn.compress_kv = kv_allgather(latent, group, gather_dim=1)
        kv = torch.cat((kv, shared_attn.compress_kv), dim=1)
        indices = torch.cat((indices, shared_attn.topk_idxs), dim=-1)
    output = indexed_sparse_attention(q, kv, attention.attn_sink, indices, attention.softmax_scale)
    output = output.clone()
    apply_rotary_emb(output[..., -rope_head_dim:], rotary_freqs, True)
    output = output.reshape(batch, length, attention.n_local_groups, -1)
    weight = attention.wo_a.weight.view(attention.n_local_groups, attention.o_lora_rank, -1)
    return attention.wo_b(torch.einsum("bsgd,grd->bsgr", output, weight).flatten(2))

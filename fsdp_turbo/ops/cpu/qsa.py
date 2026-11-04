# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Eager CPU implementation of the Qwen4-Exp QSA routing operator.

The implementation intentionally follows ``Qwen4ExpTextQSAIndexer.forward``
closely.  It serves as the CPU backend of ``qsa_selected_token_mask`` and as
the exact fallback / numerical oracle for the Triton fast path.
"""

from __future__ import annotations

import math

import torch

from fsdp_turbo.ops.registry import register_op


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (output * (1.0 + weight.float())).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    x_rope, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = x_rope * cos + _rotate_half(x_rope) * sin
    return torch.cat((rotated, x_pass), dim=-1)


@torch.no_grad()
def qsa_reference_mask(
    index_queries: torch.Tensor,
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    attention_mask: torch.Tensor,
    k_norm_weight: torch.Tensor,
    *,
    rms_norm_eps: float,
    token_budget: int,
    compress_ratio: int,
    return_token_ids: bool = False,
) -> torch.Tensor:
    """Return the exact eager QSA mask or its intermediate token IDs.

    Note on ``return_token_ids``: the mask form is the only consumer that matters
    for training and is bit-exact against the Triton fast path.  The token-ID form
    is diagnostic only — within a row the *set* of selected tokens is identical to
    the Triton path, but the *order* is not guaranteed to match: this reference
    emits score-topk permutation for every row whose visible complete blocks are
    non-empty, while the Triton path fills budget-fitting (dense) rows in
    positional order and only permutes rows that actually go through TopK.  Compare
    sorted IDs (or the mask) when testing.

    Args:
        index_queries: Normalized and RoPE-rotated index queries ``[B,Q,H,D]``.
        raw_keys: Uncompressed index keys, including cache, ``[B,KV,D]``.
        full_cos/full_sin: RoPE tensors ``[B,KV,R]`` where ``R <= D``.
        attention_mask: HF four-dimensional bool or additive mask
            ``[B,1,Q,KV]``.
        k_norm_weight: Qwen4-Exp residual RMSNorm parameter ``[D]``.  The
            effective multiplier is ``1 + weight``.
    """
    if attention_mask.dtype == torch.bool:
        visible = attention_mask
    else:
        visible = attention_mask == 0

    batch_size, query_length, num_heads, head_dim = index_queries.shape
    kv_length = raw_keys.shape[1]
    block_budget = token_budget // compress_ratio
    output_width = token_budget + compress_ratio - 1
    selected = torch.full(
        (batch_size, query_length, output_width),
        -1,
        dtype=torch.int64,
        device=index_queries.device,
    )

    for batch_idx in range(batch_size):
        rope_batch = min(batch_idx, full_cos.shape[0] - 1)
        for query_idx in range(query_length):
            local_visible = torch.nonzero(visible[batch_idx, 0, query_idx], as_tuple=False).flatten()
            num_complete = local_visible.numel() // compress_ratio
            if num_complete:
                block_tokens = local_visible[: num_complete * compress_ratio].view(num_complete, compress_ratio)
                grouped = raw_keys[batch_idx].index_select(0, block_tokens.flatten())
                pooled = grouped.view(num_complete, compress_ratio, head_dim).float().mean(1).to(raw_keys.dtype)
                pooled = _rms_norm(pooled, k_norm_weight, rms_norm_eps)
                starts = block_tokens[:, 0]
                keys = _apply_rope(
                    pooled,
                    full_cos[rope_batch].index_select(0, starts),
                    full_sin[rope_batch].index_select(0, starts),
                )
                scores = torch.matmul(index_queries[batch_idx, query_idx].float(), keys.float().T).T
                scores = torch.relu(scores).sum(-1) / math.sqrt(head_dim)
                top_blocks = scores.topk(min(block_budget, num_complete), dim=0).indices
                tokens = block_tokens.index_select(0, top_blocks).flatten()
            else:
                tokens = local_visible.new_empty((0,))
            tail = local_visible[num_complete * compress_ratio :]
            tokens = torch.cat((tokens, tail))
            selected[batch_idx, query_idx, : tokens.numel()] = tokens

    if return_token_ids:
        return selected.to(torch.int32)

    # Route every -1 to a sentinel column and remove that column afterwards.
    # This also avoids duplicate invalid writes racing with a valid token 0.
    bool_output = torch.zeros(
        (batch_size, 1, query_length, kv_length + 1), dtype=torch.bool, device=attention_mask.device
    )
    scatter_ids = torch.where(selected >= 0, selected, kv_length).unsqueeze(1)
    bool_output.scatter_(-1, scatter_ids, True)
    bool_output = bool_output[..., :kv_length]
    if attention_mask.dtype == torch.bool:
        return bool_output

    min_value = torch.finfo(attention_mask.dtype).min
    return torch.where(bool_output, attention_mask.new_zeros(()), min_value)


@register_op("qsa_selected_token_mask", "cpu")
def qsa_selected_token_mask_cpu(
    index_queries: torch.Tensor,
    raw_keys: torch.Tensor | None,
    full_cos: torch.Tensor | None,
    full_sin: torch.Tensor | None,
    attention_mask: torch.Tensor,
    k_norm_weight: torch.Tensor | None,
    *,
    rms_norm_eps: float,
    token_budget: int,
    compress_ratio: int,
    query_chunk_size: int | None = None,
    validate_prefix_mask: bool = True,
    return_token_ids: bool = False,
    block_keys: torch.Tensor | None = None,
) -> torch.Tensor:
    """CPU backend of ``qsa_selected_token_mask``: the exact eager reference.

    ``query_chunk_size`` and ``validate_prefix_mask`` only steer the Triton
    fast path and are accepted for signature compatibility, then ignored.
    """
    if (raw_keys is None) == (block_keys is None):
        raise ValueError("exactly one of raw_keys or block_keys must be provided")
    if block_keys is not None:
        return qsa_reference_block_mask(
            index_queries,
            block_keys,
            attention_mask,
            token_budget=token_budget,
            compress_ratio=compress_ratio,
            return_token_ids=return_token_ids,
        )
    return qsa_reference_mask(
        index_queries,
        raw_keys,
        full_cos,
        full_sin,
        attention_mask,
        k_norm_weight,
        rms_norm_eps=rms_norm_eps,
        token_budget=token_budget,
        compress_ratio=compress_ratio,
        return_token_ids=return_token_ids,
    )


@torch.no_grad()
def qsa_reference_block_mask(
    index_queries: torch.Tensor,
    block_keys: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    token_budget: int,
    compress_ratio: int,
    return_token_ids: bool = False,
) -> torch.Tensor:
    """Eager QSA selection over pre-pooled, pre-RoPE'd block keys.

    Companion of :func:`qsa_reference_mask` for the context-parallel path,
    where block keys are pooled locally per rank and gathered before scoring.
    Requires a causal-prefix attention mask (the CP forward guarantees this
    and validates it), so block ``b`` covers tokens
    ``[b * compress_ratio, (b + 1) * compress_ratio)``.
    """
    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    counts = visible[:, 0].sum(dim=-1)

    batch_size, query_length, num_heads, head_dim = index_queries.shape
    num_blocks = block_keys.shape[1]
    kv_length = num_blocks * compress_ratio
    block_budget = token_budget // compress_ratio
    output_width = token_budget + compress_ratio - 1
    selected = torch.full(
        (batch_size, query_length, output_width),
        -1,
        dtype=torch.int64,
        device=index_queries.device,
    )

    lanes = torch.arange(compress_ratio, device=index_queries.device)
    for batch_idx in range(batch_size):
        for query_idx in range(query_length):
            count = int(counts[batch_idx, query_idx])
            num_complete = count // compress_ratio
            if num_complete:
                keys = block_keys[batch_idx, :num_complete]
                scores = torch.matmul(index_queries[batch_idx, query_idx].float(), keys.float().T).T
                scores = torch.relu(scores).sum(-1) / math.sqrt(head_dim)
                top_blocks = scores.topk(min(block_budget, num_complete), dim=0).indices
                tokens = (top_blocks[:, None] * compress_ratio + lanes[None, :]).reshape(-1)
            else:
                tokens = selected.new_empty((0,))
            tail = torch.arange(num_complete * compress_ratio, count, device=index_queries.device)
            tokens = torch.cat((tokens, tail))
            selected[batch_idx, query_idx, : tokens.numel()] = tokens

    if return_token_ids:
        return selected.to(torch.int32)

    bool_output = torch.zeros(
        (batch_size, 1, query_length, kv_length + 1), dtype=torch.bool, device=attention_mask.device
    )
    scatter_ids = torch.where(selected >= 0, selected, kv_length).unsqueeze(1)
    bool_output.scatter_(-1, scatter_ids, True)
    bool_output = bool_output[..., :kv_length]
    if attention_mask.dtype == torch.bool:
        return bool_output

    min_value = torch.finfo(attention_mask.dtype).min
    return torch.where(bool_output, attention_mask.new_zeros(()), min_value)


@register_op("qsa_compress_block_keys", "cpu")
@torch.no_grad()
def qsa_compress_block_keys_cpu(
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    k_norm_weight: torch.Tensor,
    *,
    rms_norm_eps: float,
    compress_ratio: int,
) -> torch.Tensor:
    """Pool, normalize and rotate keys locally before CP communication."""
    if raw_keys.ndim != 3:
        raise ValueError("raw_keys must be [B, T, D]")
    batch_size, seq_length, head_dim = raw_keys.shape
    if seq_length % compress_ratio:
        raise ValueError(f"raw_keys length ({seq_length}) must be divisible by compress_ratio ({compress_ratio}).")
    num_blocks = seq_length // compress_ratio
    pooled = raw_keys.view(batch_size, num_blocks, compress_ratio, head_dim).float().mean(2)
    compressed = _rms_norm(pooled.to(raw_keys.dtype), k_norm_weight, rms_norm_eps)
    return _apply_rope(
        compressed,
        full_cos[:, : num_blocks * compress_ratio : compress_ratio],
        full_sin[:, : num_blocks * compress_ratio : compress_ratio],
    )


__all__ = [
    "qsa_reference_mask",
    "qsa_reference_block_mask",
    "qsa_selected_token_mask_cpu",
    "qsa_compress_block_keys_cpu",
]

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Fused Triton kernels for Qwen3.8 / Qwen4-Exp QSA routing.

The fast path targets the model's standard causal-prefix mask.  It removes the
per-query Python loop from the Hugging Face reference implementation while
retaining ``torch.topk``: Ascend's optimized TopK is substantially better than
an iterative scalar selection loop in Triton.
"""

# pylint: disable=possibly-used-before-assignment

from __future__ import annotations

import logging

import torch

from fsdp_turbo.ops.cpu.qsa import (
    _apply_rope,
    qsa_compress_block_keys_cpu,
    qsa_reference_block_mask,
    qsa_reference_mask,
)
from fsdp_turbo.utils.log import log_warning_once

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _compress_norm_rope_kernel(
        raw_key,
        norm_weight,
        cos,
        sin,
        compressed_key,
        stride_kb: tl.constexpr,
        stride_ks: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_cb: tl.constexpr,
        stride_cs: tl.constexpr,
        stride_cd: tl.constexpr,
        stride_cos_b: tl.constexpr,
        stride_cos_s: tl.constexpr,
        stride_cos_d: tl.constexpr,
        stride_sin_b: tl.constexpr,
        stride_sin_s: tl.constexpr,
        stride_sin_d: tl.constexpr,
        num_blocks: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        eps: tl.constexpr,
        compress_ratio: tl.constexpr,
        input_is_bf16: tl.constexpr,
        apply_rope: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        block_idx = tl.program_id(1)
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < head_dim
        token_start = block_idx * compress_ratio

        pooled = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for lane in tl.static_range(0, compress_ratio):
            ptr = raw_key + batch_idx * stride_kb + (token_start + lane) * stride_ks + dims * stride_kd
            pooled += tl.load(ptr, mask=dim_mask, other=0.0).to(tl.float32)
        pooled *= 1.0 / compress_ratio
        # Match the reference's ``mean(...).to(raw_keys.dtype)`` before norm.
        if input_is_bf16:
            pooled = pooled.to(tl.bfloat16).to(tl.float32)
        else:
            pooled = pooled.to(tl.float16).to(tl.float32)

        variance = tl.sum(pooled * pooled, axis=0) / head_dim
        inv_rms = tl.rsqrt(variance + eps)
        weight = tl.load(norm_weight + dims, mask=dim_mask, other=0.0).to(tl.float32)
        normalized = pooled * inv_rms * (1.0 + weight)
        # Qwen4-Exp RMSNorm casts back before RoPE.
        if input_is_bf16:
            normalized = normalized.to(tl.bfloat16)
        else:
            normalized = normalized.to(tl.float16)

        output = normalized
        if apply_rope:
            half_rotary = rotary_dim // 2
            rope_mask = dim_mask & (dims < rotary_dim)
            pair_dims = tl.where(dims < half_rotary, dims + half_rotary, dims - half_rotary)
            safe_pair_dims = tl.where(rope_mask, pair_dims, dims)
            paired = tl.gather(normalized, safe_pair_dims, 0)
            rotated_half = tl.where(dims < half_rotary, -paired, paired)
            cos_value = tl.load(
                cos + batch_idx * stride_cos_b + token_start * stride_cos_s + dims * stride_cos_d,
                mask=rope_mask,
                other=1.0,
            )
            sin_value = tl.load(
                sin + batch_idx * stride_sin_b + token_start * stride_sin_s + dims * stride_sin_d,
                mask=rope_mask,
                other=0.0,
            )
            output = tl.where(rope_mask, normalized * cos_value + rotated_half * sin_value, normalized)
        output_ptr = compressed_key + batch_idx * stride_cb + block_idx * stride_cs + dims * stride_cd
        tl.store(output_ptr, output, mask=dim_mask)

    @triton.jit
    def _qsa_score_kernel(
        query,
        compressed_key,
        visible_counts,
        scores,
        stride_qb: tl.constexpr,
        stride_qs: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_ks: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_os: tl.constexpr,
        stride_op: tl.constexpr,
        query_length: tl.constexpr,
        num_blocks: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        compress_ratio: tl.constexpr,
        score_scale: tl.constexpr,
        query_start,
        rows_in_chunk: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        local_q_tile = tl.program_id(1)
        key_tile = tl.program_id(2)

        local_q = local_q_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
        query_idx = query_start + local_q
        block_idx = key_tile * BLOCK_P + tl.arange(0, BLOCK_P)
        dims = tl.arange(0, BLOCK_D)
        q_mask = (local_q < rows_in_chunk) & (query_idx < query_length)
        p_mask = block_idx < num_blocks
        d_mask = dims < head_dim
        accumulated = tl.zeros((BLOCK_Q, BLOCK_P), dtype=tl.float32)

        k_ptr = compressed_key + batch_idx * stride_kb + block_idx[:, None] * stride_ks + dims[None, :] * stride_kd
        # Every index query head shares the same compressed key.  Keep one key
        # tile resident in UB across all head-wise dot products.
        k_values = tl.load(k_ptr, mask=p_mask[:, None] & d_mask[None, :], other=0.0)

        for head_idx in tl.static_range(0, num_heads):
            q_ptr = (
                query
                + batch_idx * stride_qb
                + query_idx[:, None] * stride_qs
                + head_idx * stride_qh
                + dims[None, :] * stride_qd
            )
            # Q/K storage is BF16/FP16 and tl.dot accumulates in FP32, matching
            # the useful precision of the gold path without the prohibitive
            # Ascend compile cost of an FP32 cube specialization at D=128.
            q_values = tl.load(q_ptr, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
            head_scores = tl.dot(q_values, tl.trans(k_values))
            accumulated += tl.maximum(head_scores, 0.0)

        counts = tl.load(
            visible_counts + batch_idx * stride_vb + query_idx * stride_vs,
            mask=q_mask,
            other=0,
        )
        visible_blocks = counts // compress_ratio
        valid = q_mask[:, None] & p_mask[None, :] & (block_idx[None, :] < visible_blocks[:, None])
        accumulated *= score_scale
        accumulated = tl.where(valid, accumulated, -float("inf"))
        out_ptr = scores + batch_idx * stride_ob + local_q[:, None] * stride_os + block_idx[None, :] * stride_op
        tl.store(out_ptr, accumulated, mask=(local_q[:, None] < rows_in_chunk) & p_mask[None, :])

    @triton.jit
    def _fill_dense_and_tail_kernel(
        visible_counts,
        selected,
        stride_vb: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_os: tl.constexpr,
        stride_ow: tl.constexpr,
        query_length: tl.constexpr,
        block_budget: tl.constexpr,
        compress_ratio: tl.constexpr,
        output_width: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        query_idx = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_W)
        count = tl.load(visible_counts + batch_idx * stride_vb + query_idx * stride_vs)
        complete_blocks = count // compress_ratio
        selected_blocks = tl.minimum(complete_blocks, block_budget)
        dense_count = selected_blocks * compress_ratio
        tail_count = count - complete_blocks * compress_ratio
        dense_valid = offsets < dense_count
        tail_offset = offsets - dense_count
        tail_valid = (tail_offset >= 0) & (tail_offset < tail_count)
        values = tl.where(dense_valid, offsets, complete_blocks * compress_ratio + tail_offset)
        out_ptr = selected + batch_idx * stride_ob + query_idx * stride_os + offsets * stride_ow
        tl.store(out_ptr, values, mask=(offsets < output_width) & (dense_valid | tail_valid))

    @triton.jit
    def _expand_sparse_topk_kernel(
        top_blocks,
        visible_counts,
        selected,
        stride_tb: tl.constexpr,
        stride_ts: tl.constexpr,
        stride_tk: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_os: tl.constexpr,
        stride_ow: tl.constexpr,
        query_start,
        rows_in_chunk: tl.constexpr,
        block_budget: tl.constexpr,
        compress_ratio: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        local_q = tl.program_id(1)
        query_idx = query_start + local_q
        offsets = tl.arange(0, BLOCK_T)
        block_slot = offsets // compress_ratio
        lane = offsets - block_slot * compress_ratio
        count = tl.load(visible_counts + batch_idx * stride_vb + query_idx * stride_vs)
        is_sparse = count // compress_ratio > block_budget
        block_id = tl.load(
            top_blocks + batch_idx * stride_tb + local_q * stride_ts + block_slot * stride_tk,
            mask=offsets < block_budget * compress_ratio,
            other=0,
        ).to(tl.int32)
        token_id = block_id * compress_ratio + lane
        out_ptr = selected + batch_idx * stride_ob + query_idx * stride_os + offsets * stride_ow
        tl.store(out_ptr, token_id, mask=is_sparse & (offsets < block_budget * compress_ratio))


def _visible_counts(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    return visible, visible[:, 0].sum(dim=-1).to(torch.int32)


def _is_prefix_mask(visible: torch.Tensor, counts: torch.Tensor) -> bool:
    kv_length = visible.shape[-1]
    expected = torch.arange(kv_length, device=visible.device).view(1, 1, 1, kv_length)
    expected = expected < counts[:, None, :, None]
    return bool(torch.equal(visible, expected))


@torch.no_grad()
def qsa_compress_block_keys_triton(
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    k_norm_weight: torch.Tensor,
    *,
    rms_norm_eps: float,
    compress_ratio: int,
) -> torch.Tensor:
    """Pool raw indexer keys into normalized, RoPE-rotated block keys.

    This is the locally-shardable half of ``qsa_selected_token_mask``: the
    context-parallel forward runs it on each rank's sequence shard (the shard
    length must be divisible by ``compress_ratio`` so blocks never straddle a
    rank boundary) and gathers the small ``[B, S/ratio, D]`` result instead of
    the full hidden states.  Bitwise it mirrors the fast path: pooling and
    RMSNorm in the Triton kernel, RoPE applied by the vectorized PyTorch op.
    """
    if raw_keys.ndim != 3:
        raise ValueError("raw_keys must be [B, T, D]")
    batch_size, seq_length, head_dim = raw_keys.shape
    if seq_length % compress_ratio:
        raise ValueError(f"raw_keys length ({seq_length}) must be divisible by compress_ratio ({compress_ratio}).")
    num_blocks = seq_length // compress_ratio
    block_cos = full_cos[:, : num_blocks * compress_ratio : compress_ratio]
    block_sin = full_sin[:, : num_blocks * compress_ratio : compress_ratio]

    use_triton = (
        _TRITON_AVAILABLE
        and raw_keys.device.type in ("npu", "cuda")
        and raw_keys.dtype in (torch.float16, torch.bfloat16)
        and head_dim >= 16
        and head_dim % 16 == 0
    )
    if use_triton:
        compressed = torch.empty((batch_size, num_blocks, head_dim), dtype=raw_keys.dtype, device=raw_keys.device)
        _compress_norm_rope_kernel[(batch_size, num_blocks)](
            raw_keys,
            k_norm_weight,
            full_cos,
            full_sin,
            compressed,
            *raw_keys.stride(),
            *compressed.stride(),
            *full_cos.stride(),
            *full_sin.stride(),
            num_blocks=num_blocks,
            head_dim=head_dim,
            rotary_dim=full_cos.shape[-1],
            eps=rms_norm_eps,
            compress_ratio=compress_ratio,
            input_is_bf16=raw_keys.dtype == torch.bfloat16,
            apply_rope=False,
            BLOCK_D=triton.next_power_of_2(head_dim),
        )
    else:
        return qsa_compress_block_keys_cpu(
            raw_keys,
            full_cos,
            full_sin,
            k_norm_weight,
            rms_norm_eps=rms_norm_eps,
            compress_ratio=compress_ratio,
        )
    return _apply_rope(compressed, block_cos, block_sin)


@torch.no_grad()
def qsa_selected_token_mask_triton(
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
    """Select QSA tokens using fused Triton kernels on NPU/CUDA.

    The accelerated route requires a causal prefix mask: every row must be
    ``True...True,False...False`` (or ``0...0,-inf...-inf`` for additive
    masks).  Set ``validate_prefix_mask=True`` while bringing up a new data
    pipeline; non-prefix masks then use the exact eager fallback.  Disable the
    check in production to avoid a device synchronization at every QSA layer.

    Pass ``block_keys`` (``[B, S/ratio, D]`` output of
    :func:`qsa_compress_block_keys`) instead of ``raw_keys`` to skip the
    compression stage — used by the context-parallel forward, which pools the
    keys locally per rank and gathers the compressed result.
    """
    if index_queries.ndim != 4:
        raise ValueError("index_queries must be [B,Q,H,D]")
    batch_size, query_length, num_heads, head_dim = index_queries.shape
    if not batch_size or not query_length or not num_heads or not head_dim:
        raise ValueError("QSA does not support empty batch/query/head dimensions")
    if (raw_keys is None) == (block_keys is None):
        raise ValueError("exactly one of raw_keys or block_keys must be provided")

    if block_keys is not None:
        if block_keys.ndim != 3 or block_keys.shape[0] not in (1, batch_size):
            raise ValueError("block_keys must be [B, S/ratio, D] with batch one or B")
        if block_keys.shape[2] != head_dim:
            raise ValueError("block_keys must match the index query head dimension")
        if full_cos is not None or full_sin is not None or k_norm_weight is not None:
            raise ValueError("full_cos/full_sin/k_norm_weight are unused with block_keys")
        kv_length = block_keys.shape[1] * compress_ratio
        keys_dtype = block_keys.dtype
    else:
        if raw_keys.ndim != 3:
            raise ValueError("raw_keys must be [B,KV,D]")
        kv_length = raw_keys.shape[1]
        if kv_length < query_length:
            raise ValueError("KV length must be greater than or equal to query length")
        if raw_keys.shape != (batch_size, kv_length, head_dim):
            raise ValueError("raw_keys must match the query batch and head dimension")
        if full_cos is None or full_sin is None or k_norm_weight is None:
            raise ValueError("full_cos/full_sin/k_norm_weight are required with raw_keys")
        if full_cos.ndim != 3 or full_sin.shape != full_cos.shape:
            raise ValueError("full_cos/full_sin must have the same [B,KV,R] shape")
        if full_cos.shape[1] != kv_length or full_cos.shape[0] not in (1, batch_size):
            raise ValueError("RoPE batch must be one or B and its sequence axis must equal KV")
        if k_norm_weight.shape != (head_dim,):
            raise ValueError("k_norm_weight must have shape [D]")
        keys_dtype = raw_keys.dtype

    if attention_mask.shape != (batch_size, 1, query_length, kv_length):
        raise ValueError("attention_mask must be [B,1,Q,KV]")
    rotary_dim = full_cos.shape[-1] if full_cos is not None else head_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("RoPE width must be positive, even, and no greater than the head dimension")
    input_tensors = (attention_mask, block_keys if block_keys is not None else raw_keys)
    if any(tensor.device != index_queries.device for tensor in input_tensors):
        raise ValueError("all QSA tensors must be on the same device")
    if full_cos is not None:
        if full_cos.device != index_queries.device or full_sin.device != index_queries.device:
            raise ValueError("all QSA tensors must be on the same device")
    if token_budget <= 0 or compress_ratio <= 1 or token_budget % compress_ratio:
        raise ValueError("token_budget must be positive and divisible by compress_ratio > 1")
    if query_chunk_size is not None and query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive or None for automatic selection")

    visible, counts = _visible_counts(attention_mask)
    device_type = index_queries.device.type
    supported_dtype = index_queries.dtype in (torch.float16, torch.bfloat16)
    use_triton = (
        _TRITON_AVAILABLE
        and device_type in ("npu", "cuda")
        and supported_dtype
        and keys_dtype == index_queries.dtype
        and head_dim >= 16
        and head_dim % 16 == 0
    )
    if validate_prefix_mask and not _is_prefix_mask(visible, counts):
        use_triton = False
    if not use_triton:
        # Loud once: the fallback is exact but runs a per-query Python loop, which is a
        # severe per-step slowdown when it happens silently inside a training run.
        log_warning_once(logger, "QSA Triton fast path unavailable for this input; using the exact eager reference.")
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

    if full_cos is not None and full_cos.shape[0] == 1 and batch_size != 1:
        full_cos = full_cos.expand(batch_size, -1, -1)
        full_sin = full_sin.expand(batch_size, -1, -1)
    if k_norm_weight is not None:
        k_norm_weight = k_norm_weight.contiguous()
    num_blocks = kv_length // compress_ratio
    block_budget = token_budget // compress_ratio
    output_width = token_budget + compress_ratio - 1
    block_w = triton.next_power_of_2(output_width)
    selected = torch.full((batch_size, query_length, block_w), -1, dtype=torch.int32, device=index_queries.device)

    _fill_dense_and_tail_kernel[(batch_size, query_length)](
        counts,
        selected,
        *counts.stride(),
        *selected.stride(),
        query_length=query_length,
        block_budget=block_budget,
        compress_ratio=compress_ratio,
        output_width=output_width,
        BLOCK_W=block_w,
    )

    # For a prefix-causal row, score-ordered TopK starts only after the visible
    # complete blocks exceed the budget (position token_budget + 1).
    query_position_offset = kv_length - query_length
    sparse_start = max(0, token_budget - query_position_offset)
    if num_blocks > block_budget and sparse_start < query_length:
        block_q, block_p, block_d = 64, 128, triton.next_power_of_2(head_dim)
        if block_keys is not None:
            # Already pooled, normalized and RoPE-rotated by the CP forward.
            compressed = block_keys.contiguous()
        else:
            # Compression and key RoPE are only consumed by sparse TopK rows.  For
            # short/prefill sequences whose visible prefix fits in the token
            # budget, skip this preprocessing entirely.
            compressed = torch.empty((batch_size, num_blocks, head_dim), dtype=raw_keys.dtype, device=raw_keys.device)
            _compress_norm_rope_kernel[(batch_size, num_blocks)](
                raw_keys,
                k_norm_weight,
                full_cos,
                full_sin,
                compressed,
                *raw_keys.stride(),
                *compressed.stride(),
                *full_cos.stride(),
                *full_sin.stride(),
                num_blocks=num_blocks,
                head_dim=head_dim,
                rotary_dim=full_cos.shape[-1],
                eps=rms_norm_eps,
                compress_ratio=compress_ratio,
                input_is_bf16=raw_keys.dtype == torch.bfloat16,
                apply_rope=False,
                BLOCK_D=block_d,
            )
            # Triton-Ascend fuses the BF16 RoPE mul/add into an FMA, while the gold
            # PyTorch path rounds each multiplication before addition.  The tiny
            # difference can reorder a TopK boundary, so keep this vectorized NPU
            # operation outside the compression kernel for exact routing parity.
            block_cos = full_cos[:, : num_blocks * compress_ratio : compress_ratio]
            block_sin = full_sin[:, : num_blocks * compress_ratio : compress_ratio]
            compressed = _apply_rope(compressed, block_cos, block_sin)
        padded_num_blocks = triton.cdiv(num_blocks, block_p) * block_p
        sparse_rows = query_length - sparse_start
        if query_chunk_size is None:
            # Cap launch count without allowing the FP32 score workspace to
            # grow beyond 32 MiB per invocation.  Rows stay aligned to the
            # score kernel's BLOCK_Q tile.
            workspace_rows = (32 * 2**20) // (batch_size * padded_num_blocks * 4)
            workspace_rows = max(block_q, workspace_rows // block_q * block_q)
            query_chunk_size = min(512, workspace_rows)
        workspace_rows = min(query_chunk_size, sparse_rows)
        score_workspace = torch.full(
            (batch_size, workspace_rows, padded_num_blocks),
            -torch.inf,
            dtype=torch.float32,
            device=index_queries.device,
        )
        top_values = torch.empty(
            (batch_size, workspace_rows, block_budget),
            dtype=torch.float32,
            device=index_queries.device,
        )
        top_blocks = torch.empty(
            (batch_size, workspace_rows, block_budget),
            dtype=torch.int64,
            device=index_queries.device,
        )
        for query_start in range(sparse_start, query_length, query_chunk_size):
            rows = min(query_chunk_size, query_length - query_start)
            grid = (batch_size, triton.cdiv(workspace_rows, block_q), triton.cdiv(num_blocks, block_p))
            _qsa_score_kernel[grid](
                index_queries,
                compressed,
                counts,
                score_workspace,
                *index_queries.stride(),
                *compressed.stride(),
                *counts.stride(),
                *score_workspace.stride(),
                query_length=query_length,
                num_blocks=num_blocks,
                num_heads=num_heads,
                head_dim=head_dim,
                compress_ratio=compress_ratio,
                score_scale=head_dim**-0.5,
                query_start=query_start,
                rows_in_chunk=workspace_rows,
                BLOCK_Q=block_q,
                BLOCK_P=block_p,
                BLOCK_D=block_d,
            )
            torch.topk(
                score_workspace,
                k=block_budget,
                dim=-1,
                sorted=True,
                out=(top_values, top_blocks),
            )
            block_t = triton.next_power_of_2(token_budget)
            _expand_sparse_topk_kernel[(batch_size, rows)](
                top_blocks,
                counts,
                selected,
                *top_blocks.stride(),
                *counts.stride(),
                *selected.stride(),
                query_start=query_start,
                rows_in_chunk=rows,
                block_budget=block_budget,
                compress_ratio=compress_ratio,
                BLOCK_T=block_t,
            )

    selected = selected[..., :output_width]
    if return_token_ids:
        return selected

    # Triton-Ascend 3.2 can compile the indirect tl.store used by a sparse
    # scatter, but on 910B it may silently lose non-contiguous writes.  Use the
    # optimized ACL scatter implementation for this final stage.  int64 is
    # intentional: torch.scatter does not accept the int32 routing IDs.
    bool_output = torch.zeros(
        (batch_size, 1, query_length, kv_length + 1), dtype=torch.bool, device=attention_mask.device
    )
    scatter_ids = torch.where(selected >= 0, selected, kv_length).to(torch.int64).unsqueeze(1)
    bool_output.scatter_(-1, scatter_ids, True)
    bool_output = bool_output[..., :kv_length]
    if attention_mask.dtype == torch.bool:
        return bool_output
    return torch.where(bool_output, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)

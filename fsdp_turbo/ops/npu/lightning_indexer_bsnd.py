# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Dense-BSND lightning indexer adapter for the cann_ops_transformer package.

Wraps the plain (non-quantized) ``ops.lightning_indexer`` entry point so the
DSA indexer can fetch its top-k compressed-KV selection in a single fused op.
This is the dense counterpart of ``fsdp_turbo.ops.npu.lightning_indexer``:
that module targets the DS41 QLI/QSLI adapters (MXFP4 weights, TND/PA_BBND
paged layouts), while everything here stays bf16 in the dense BSND layout.
"""

from functools import lru_cache

import torch
from cann_ops_transformer import ops


def _to_op_layout(compressed_kv, batch_size):
    """Materialize the compressed keys as a [B, T, 1, D] bf16 tensor the op accepts.

    Batch-1 prefix slices of a larger gathered buffer keep stride(0) of the
    full buffer: torch ignores strides of size-1 dimensions, so ``contiguous()``
    would be a no-op and the operator tiling rejects that layout. Flattening to
    2-D first forces a copy whenever the strides are non-standard, and stays a
    free view otherwise.
    """
    compressed_length, head_dim = compressed_kv.shape[1], compressed_kv.shape[2]
    flattened = compressed_kv.reshape(batch_size * compressed_length, head_dim)
    return flattened.view(batch_size, compressed_length, 1, head_dim).to(torch.bfloat16)


@lru_cache(maxsize=8)
def _cached_indexer_metadata(
    num_heads,
    head_dim,
    top_k,
    tail_length,
    batch_size,
    sequence_length,
    compressed_kv_length,
    device,
    *,
    mask_mode,
    compress_rate,
):
    """Describe the (B, T, D) key pool and per-batch uncompressed tail lengths.

    ``tail_length`` is the leftover of the uncompressed KV span that the
    compressed pool does not cover (``uncompressed_kv_length % compress_rate``);
    the operator uses it to align each query row to the end of the KV span.
    """
    tail_lengths = torch.full((batch_size,), tail_length, dtype=torch.int32, device=device)
    return ops.lightning_indexer_metadata(
        num_heads,
        1,
        head_dim,
        top_k,
        batch_size=batch_size,
        max_seqlen_q=sequence_length,
        max_seqlen_k=compressed_kv_length,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=mask_mode,
        cmp_ratio=compress_rate,
        cmp_residual_k=tail_lengths,
    )


def lightning_indexer_bsnd(
    query,  # [B, S, N, D], RoPE already applied outside
    compressed_kv,  # [B, T, D], indexer compressed key pool
    weights,  # [B, S, N] fp32; weights_scaling * softmax_scale folded in
    top_k,
    *,
    compress_rate,
    mask_mode=3,
    kv_len=None,
):
    """Pick the top-k compressed-KV entries for every query row.

    ``query`` / ``compressed_kv`` are handed to the operator in bf16; the
    score weights stay fp32 and must already include both scale factors since
    the operator applies none itself.

    ``kv_len`` is the uncompressed KV span covered by the compressed pool. The
    operator reconstructs it internally; when omitted it is taken as the query
    length, which is only correct for the non-CP call where the two coincide.
    Under CP prefix attention the prefix token count must be passed instead:
    both the query-length and pool-length derivations produce a wrong tail
    whenever the local span is not a multiple of ``compress_rate``.

    Returns:
        (sparse_indices [B, S, top_k] int, sparse_values [B, S, top_k]).
    """
    batch_size, sequence_length = query.shape[0], query.shape[1]
    num_heads, head_dim = query.shape[2], query.shape[3]
    compressed_kv_length = compressed_kv.shape[1]

    effective_top_k = min(top_k, compressed_kv_length)
    effective_compress_rate = compress_rate if compress_rate > 0 else 1
    uncompressed_kv_length = kv_len if kv_len is not None else sequence_length
    tail_length = uncompressed_kv_length % effective_compress_rate
    tail_lengths = torch.full((batch_size,), tail_length, dtype=torch.int32, device=query.device)

    query_input = query.contiguous().to(torch.bfloat16)
    key_input = _to_op_layout(compressed_kv, batch_size)

    metadata = _cached_indexer_metadata(
        num_heads,
        head_dim,
        effective_top_k,
        tail_length,
        batch_size,
        sequence_length,
        compressed_kv_length,
        query.device,
        mask_mode=mask_mode,
        compress_rate=effective_compress_rate,
    )

    sparse_indices, sparse_values = ops.lightning_indexer(
        query_input,
        key_input,
        weights,
        effective_top_k,
        metadata=metadata,
        cmp_residual_k=tail_lengths,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=mask_mode,
        cmp_ratio=effective_compress_rate,
        return_value=1,
    )
    sparse_indices = sparse_indices.view(batch_size, sequence_length, effective_top_k)
    sparse_values = sparse_values.view(batch_size, sequence_length, effective_top_k)
    return sparse_indices, sparse_values

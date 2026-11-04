# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Backend-dispatched entry point for the Qwen4-Exp QSA token-selection operator.

Implementations live in the backend packages: the exact eager reference in
``fsdp_turbo.ops.cpu.qsa`` and the fused Triton fast path (registered by the
NPU/CUDA backends) in ``fsdp_turbo.ops.triton.qsa_indexer``.
"""

from __future__ import annotations

import torch

from fsdp_turbo.ops.registry import dispatch_op


def qsa_selected_token_mask(
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
    """Select QSA tokens using the active device backend.

    Args:
        index_queries: Normalized and RoPE-rotated index queries ``[B,Q,H,D]``.
        raw_keys: Uncompressed index keys, including cache, ``[B,KV,D]``.
        full_cos/full_sin: RoPE tensors ``[B,KV,R]`` where ``R <= D``.
        attention_mask: HF four-dimensional bool or additive mask
            ``[B,1,Q,KV]``.
        k_norm_weight: Qwen4-Exp residual RMSNorm parameter ``[D]``.  The
            effective multiplier is ``1 + weight``.
        rms_norm_eps: Epsilon of the key RMSNorm.
        token_budget: Maximum number of selected tokens per query row.
        compress_ratio: Tokens pooled into one indexer block.
        query_chunk_size: Triton-only knob; query rows per score-kernel pass
            (``None`` picks automatically).
        validate_prefix_mask: Triton-only knob; verify the mask is a plain
            causal prefix before taking the fast path.
        block_keys: Precompressed keys ``[B,S/ratio,D]`` for CP. When supplied,
            raw_keys, full_cos, full_sin and k_norm_weight must be None.
        return_token_ids: Return ``[B,Q,W]`` token IDs instead of the mask.

    The Triton fast path requires a causal prefix mask: every row must be
    ``True...True,False...False`` (or ``0...0,-inf...-inf`` for additive
    masks).  Set ``validate_prefix_mask=True`` while bringing up a new data
    pipeline; non-prefix masks then use the exact eager fallback.  Disable the
    check in production to avoid a device synchronization at every QSA layer.
    """
    return dispatch_op(
        "qsa_selected_token_mask",
        index_queries,
        raw_keys,
        full_cos,
        full_sin,
        attention_mask,
        k_norm_weight,
        device_type=index_queries.device.type,
        rms_norm_eps=rms_norm_eps,
        token_budget=token_budget,
        compress_ratio=compress_ratio,
        query_chunk_size=query_chunk_size,
        validate_prefix_mask=validate_prefix_mask,
        return_token_ids=return_token_ids,
        block_keys=block_keys,
    )


def apply_query_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply partial RoPE to QSA queries shaped ``[B,Q,H,D]``."""
    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2
    rope, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated_half = torch.cat((-rope[..., half:], rope[..., :half]), dim=-1)
    rotated = rope * cos.unsqueeze(2) + rotated_half * sin.unsqueeze(2)
    return torch.cat((rotated, passthrough), dim=-1)


@torch.no_grad()
def qsa_compress_block_keys(
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    k_norm_weight: torch.Tensor,
    *,
    rms_norm_eps: float,
    compress_ratio: int,
) -> torch.Tensor:
    """Pool, normalize and rotate local keys before CP gathers compressed blocks.

    The local sequence length must be divisible by ``compress_ratio``.
    Returns block keys with shape ``[B,S/compress_ratio,D]``.
    """
    return dispatch_op(
        "qsa_compress_block_keys",
        raw_keys,
        full_cos,
        full_sin,
        k_norm_weight,
        device_type=raw_keys.device.type,
        rms_norm_eps=rms_norm_eps,
        compress_ratio=compress_ratio,
    )


__all__ = ["qsa_selected_token_mask", "qsa_compress_block_keys", "apply_query_rope"]

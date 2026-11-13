"""Backend-dispatched entry points for fused lightning indexers."""

import torch

from fsdp_turbo.ops.registry import dispatch_op


def fused_lightning_indexer(*args, **kwargs):
    return dispatch_op("fused_lightning_indexer", *args, **kwargs)


def lightning_indexer_topk(
    query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, topk: int, cu_seqlens: torch.Tensor | None = None
) -> torch.Tensor:
    """Return causal, document-local indices with -1 padding.

    Without boundaries Q/K/weights use BSND/BSD/BSN. With int32
    ``cu_seqlens=[0,...,T]`` they use TND/TD/TN and indices are [T,1,K].
    Weights include all score scaling. TND inputs contain no padding tokens.
    """
    return dispatch_op("lightning_indexer_topk", query, key, weights, topk, cu_seqlens, device_type=query.device.type)


def lightning_indexer_kl_loss(
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
    """Mean sparse teacher KL with gradients only on indexer Q/K/weights.

    Optional boundaries select unpadded TND inputs as in ``lightning_indexer_topk``;
    teacher Q/K use TND, statistics use [T,H], and the mean covers all T queries.
    """
    return dispatch_op(
        "lightning_indexer_kl_loss",
        query_index,
        key_index,
        weights,
        query,
        key,
        query_rope,
        key_rope,
        indices,
        maximum,
        denominator,
        scale,
        cu_seqlens,
        device_type=query_index.device.type,
    )


def lightning_indexer_score(
    q,
    pool_keys,
    weights,
    valid_candidates=None,
    scale=None,
):
    """Score DSA indexer queries against pooled keys.

    q is [B, S, NH, D], pool_keys is [B, P, D], weights is [B, S, NH]; the
    return is the [B, S, P] float32 pool-score tensor, with masked-out
    candidates at float32 ``min`` so it feeds straight into ``topk``.
    """
    return dispatch_op("lightning_indexer_score", q, pool_keys, weights, valid_candidates, scale)


__all__ = ["fused_lightning_indexer", "lightning_indexer_score", "lightning_indexer_topk", "lightning_indexer_kl_loss"]

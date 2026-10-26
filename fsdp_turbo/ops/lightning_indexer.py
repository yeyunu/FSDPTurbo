"""Backend-dispatched entry points for fused lightning indexers."""

from fsdp_turbo.ops.registry import dispatch_op


def fused_lightning_indexer(*args, **kwargs):
    return dispatch_op("fused_lightning_indexer", *args, **kwargs)


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


__all__ = ["fused_lightning_indexer", "lightning_indexer_score"]

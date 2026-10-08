"""Backend-dispatched entry point for fused lightning indexers."""

from fsdp_turbo.ops.registry import dispatch_op


def fused_lightning_indexer(*args, **kwargs):
    return dispatch_op("fused_lightning_indexer", *args, **kwargs)


__all__ = ["fused_lightning_indexer"]

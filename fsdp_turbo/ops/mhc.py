"""Backend-dispatched entry points for fused MHC operators."""

from fsdp_turbo.ops.registry import dispatch_op


def fused_mhc_pre_sinkhorn_premix(*args, **kwargs):
    return dispatch_op("fused_mhc_pre_sinkhorn_premix", *args, **kwargs)


def fused_mhc_post(*args, **kwargs):
    return dispatch_op("fused_mhc_post", *args, **kwargs)


__all__ = ["fused_mhc_post", "fused_mhc_pre_sinkhorn_premix"]

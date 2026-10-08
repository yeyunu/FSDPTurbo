"""Backend-dispatched entry point for the fused compressor operator."""

from fsdp_turbo.ops.registry import dispatch_op


def fused_compressor(*args, **kwargs):
    return dispatch_op("fused_compressor", *args, **kwargs)


__all__ = ["fused_compressor"]

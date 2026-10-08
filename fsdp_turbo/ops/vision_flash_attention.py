"""Backend-dispatched entry point for vision FlashAttentionScore."""

from fsdp_turbo.ops.registry import dispatch_op


def fused_vision_flash_attention(*args, **kwargs):
    return dispatch_op("fused_vision_flash_attention", *args, **kwargs)


__all__ = ["fused_vision_flash_attention"]

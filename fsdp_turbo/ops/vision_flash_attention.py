"""Backend-dispatched entry point for vision FlashAttentionScore."""

import torch

from fsdp_turbo.ops.registry import dispatch_op


def fused_vision_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sequence_ends: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Full bidirectional attention, isolated per image.

    Without ``sequence_ends``, inputs have shape [1,S,N,D] (BSND).
    Otherwise inputs are [T,N,D] (TND); ends are cumulative image token
    counts, without a leading zero. NPU uses one fused call for all images.
    """
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Vision attention requires identical Q/K/V shapes")
    if sequence_ends is None:
        if query.ndim != 4 or query.shape[0] != 1:
            raise ValueError("Vision BSND attention expects [1,S,N,D] inputs")
    elif (
        query.ndim != 3
        or not sequence_ends
        or sequence_ends[-1] != query.shape[0]
        or any(end <= start for start, end in zip((0,) + sequence_ends, sequence_ends))
    ):
        raise ValueError("Vision TND attention requires increasing image ends covering all tokens")
    return dispatch_op(
        "fused_vision_flash_attention",
        query,
        key,
        value,
        sequence_ends=sequence_ends,
        device_type=query.device.type,
    )


__all__ = ["fused_vision_flash_attention"]

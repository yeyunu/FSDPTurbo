# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Device-dispatched RoPE over an explicitly selected rotary subspace."""

import torch

from fsdp_turbo.ops import dispatch_op


def rope(x, cos, sin, *, interleaved=False, inverse=False, inplace=False, use_eager=False):
    """Rotate [B,S,D] or [B,S,H,D], retaining FP32 compute and x.dtype output.

    cos/sin are fixed D/2-wide coefficients. [S,D/2] is expanded over
    batch/heads; other shapes must broadcast to x[..., :D/2]. Interleaved
    mode pairs adjacent elements, otherwise the two halves are paired.
    inverse negates sin. Pass a view for partial RoPE. inplace uses copy_
    and retains normal autograd restrictions on writing to leaf tensors.
    use_eager selects PyTorch operations on the input device, not a CPU copy.
    """
    if x.ndim not in (3, 4) or x.shape[-1] == 0 or x.shape[-1] % 2:
        raise ValueError("RoPE requires 3-D/4-D x with a positive even last dimension")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("RoPE x must be FP16, BF16 or FP32")
    if cos.shape != sin.shape or cos.ndim == 0 or cos.shape[-1] != x.shape[-1] // 2:
        raise ValueError("RoPE cos/sin must have equal shapes ending in D/2")
    if cos.device != x.device or sin.device != x.device:
        raise ValueError("RoPE x/cos/sin must be on the same device")
    if not cos.is_floating_point() or not sin.is_floating_point():
        raise TypeError("RoPE cos/sin must be real floating-point tensors")
    if cos.requires_grad or sin.requires_grad:
        raise ValueError("RoPE cos/sin must be fixed positional coefficients")
    if cos.ndim == 2:
        shape = (1, cos.shape[0], *([1] if x.ndim == 4 else []), cos.shape[-1])
        cos, sin = cos.reshape(shape), sin.reshape(shape)
    target = (*x.shape[:-1], x.shape[-1] // 2)
    try:
        broadcast = torch.broadcast_shapes(cos.shape, target)
    except RuntimeError as error:
        raise ValueError("RoPE cos/sin do not broadcast to the rotary input") from error
    if tuple(broadcast) != target:
        raise ValueError("RoPE cos/sin must not expand the input shape")
    # Both backends save only coefficients, so in-place writeback needs no
    # extra input clone, including for aligned contiguous FP32 tensors.
    result = dispatch_op(
        "rope",
        x,
        cos,
        sin,
        interleaved=interleaved,
        inverse=inverse,
        device_type="cpu" if use_eager else x.device.type,
    )
    return x.copy_(result) if inplace else result


def inplace_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
    inverse: bool = False,
    inplace: bool = False,
    device_type: str = "cpu",
) -> torch.Tensor:
    """Rotate the trailing rotary block of a full [nope | rope] head.

    cos/sin are ``rotary_dim // 2``-wide coefficients, so the rotated block is the
    last ``rotary_dim = 2 * cos.shape[-1]`` columns and any leading nope columns
    are left untouched. Accepts [B,S,D] and [B,S,H,D]. ``device_type="npu"``
    selects the NPU implementation, which uses the Triton inplace kernel over
    the whole head when available (no ViewCopy/Slice), then falls back to CANN.
    Other device types use the eager `rope` operation. The default
    ``device_type="cpu"`` selects the eager implementation, matching the
    previous API's default behavior.

    When the triton kernel runs it mutates x's storage in place yet returns a
    fresh autograd-tracked tensor, so always use the return value
    (``x = inplace_rope(x, ...)``) or the rotary block's gradient will be dropped.
    """
    if x.ndim not in (3, 4) or x.shape[-1] == 0:
        raise ValueError("RoPE requires 3-D/4-D x with a non-empty last dimension")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("RoPE x must be FP16, BF16 or FP32")
    if cos.shape != sin.shape or cos.ndim == 0 or cos.shape[-1] == 0:
        raise ValueError("RoPE cos/sin must have equal shapes ending in rotary_dim/2")
    if cos.device != x.device or sin.device != x.device:
        raise ValueError("RoPE x/cos/sin must be on the same device")
    rotary_dim = 2 * cos.shape[-1]
    if rotary_dim > x.shape[-1]:
        raise ValueError("RoPE rotary block must not exceed the head dimension")
    if rotary_dim == x.shape[-1] and device_type != "npu":
        return rope(
            x,
            cos,
            sin,
            interleaved=interleaved,
            inverse=inverse,
            inplace=inplace,
            use_eager=device_type == "cpu",
        )
    return dispatch_op(
        "inplace_rope",
        x,
        cos,
        sin,
        interleaved=interleaved,
        inverse=inverse,
        inplace=inplace,
        device_type=device_type,
    )


def partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate the trailing rope block of q/o in place, on the best backend available.

    Equivalent to ``inplace_rope(x, cos, sin, interleaved=True, inverse=inverse,
    inplace=True)`` but with its own backend order, so tuning q/o cannot move the
    path taken by kv/latent/indexer/context-parallel callers. NPU tries the CANN
    operator, then RotaryMul, the Triton inplace kernel and the eager reference;
    other devices use ``ops/cpu/rope.py``.
    """
    return dispatch_op("partial_rope", x, cos, sin, inverse=inverse)


__all__ = ["rope", "inplace_rope", "partial_rope"]

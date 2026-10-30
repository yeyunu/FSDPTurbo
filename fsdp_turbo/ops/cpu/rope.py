# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Differentiable PyTorch RoPE reference; also usable on accelerators."""

import torch

from fsdp_turbo.ops.registry import register_op


@register_op("rope", "cpu")
def rope_torch(x, cos, sin, *, interleaved=False, inverse=False):
    value = x.float()
    cos, sin = cos.float(), sin.float()
    if inverse:
        sin = -sin
    if interleaved:
        left, right = value[..., ::2], value[..., 1::2]
    else:
        left, right = value.chunk(2, -1)
    real = left * cos - right * sin
    imag = right * cos + left * sin
    if interleaved:
        output = torch.stack((real, imag), dim=-1).flatten(-2)
    else:
        output = torch.cat((real, imag), dim=-1)
    return output.to(x.dtype)


@register_op("inplace_rope", "cpu")
def inplace_rope_cpu(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
    inverse: bool = False,
    inplace: bool = False,
) -> torch.Tensor:
    """Rotate the trailing block of a full [nope | rope] head; reference fallback."""
    rotary_dim = 2 * cos.shape[-1]
    rope_x = x[..., -rotary_dim:]
    if cos.ndim == 2:
        # rope_torch expects the broadcast shape produced by the `rope` wrapper.
        coeff_shape = (1, cos.shape[0], *([1] if rope_x.ndim == 4 else []), cos.shape[-1])
        cos = cos.reshape(coeff_shape)
        sin = sin.reshape(coeff_shape)
    rotated = rope_torch(rope_x, cos, sin, interleaved=interleaved, inverse=inverse)
    if inplace:
        x[..., -rotary_dim:] = rotated
        return x
    out = x.clone()
    out[..., -rotary_dim:] = rotated
    return out


@register_op("partial_rope", "cpu")
def partial_rope_cpu(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Device-agnostic fallback for ``partial_rope``; the reference semantics."""
    return inplace_rope_cpu(x, cos, sin, interleaved=True, inverse=inverse, inplace=True)

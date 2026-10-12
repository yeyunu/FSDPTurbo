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

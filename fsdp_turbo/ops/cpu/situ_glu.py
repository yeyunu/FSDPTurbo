# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch

from fsdp_turbo.ops.registry import register_op


def situ_glu_eager(
    x: torch.Tensor,
    *,
    dim: int = -1,
    beta: float = 1.0,
    linear_beta: float = 0.0,
    activate_left: bool = True,
) -> torch.Tensor:
    """Eager (pure PyTorch) implementation of SituGLU.

    Splits ``x`` along ``dim`` into a gate half and an up half, applies the
    SiTU activation (``beta * tanh(gate / beta) * sigmoid(gate)``) to the gate
    half, optionally applies a bounded tanh to the up half, and returns their
    product.
    """
    d = x.shape[dim] // 2
    if dim == -1:
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
    else:
        slices_gate = [slice(None)] * x.ndim
        slices_gate[dim] = slice(d)
        slices_up = [slice(None)] * x.ndim
        slices_up[dim] = slice(d, 2 * d)
        gate = x[tuple(slices_gate)].to(torch.float32)
        up = x[tuple(slices_up)].to(torch.float32)

    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)

    if linear_beta not in (0.0, None):
        up = linear_beta * torch.tanh(up / linear_beta)

    result = situ_a * up

    return result.to(x.dtype)


@register_op('situ_glu', 'cpu')
def situ_glu_cpu(
    x: torch.Tensor,
    dim: int = -1,
    beta: float = 1.0,
    linear_beta: float = 0.0,
    activate_left: bool = True,
) -> torch.Tensor:
    """CPU fallback for SituGLU using the eager implementation."""
    return situ_glu_eager(x, dim=dim, beta=beta, linear_beta=linear_beta, activate_left=activate_left)


__all__ = ['situ_glu_cpu', 'situ_glu_eager']

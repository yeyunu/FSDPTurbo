# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Copyright (c) 2001-2004 Idiap Research Institute Ronan Collobert Samy Bengio Johnny Mariethoz.
# Licensed under the Apache License, Version 2.0 (the "License");

import logging

import torch

from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.utils.log import log_warning_once

try:
    import torch_npu
except ImportError:
    torch_npu = None

try:
    import cann_ops_nn.ops  # noqa: F401

    _HAS_CANN_OPS_NN = hasattr(torch.ops.cann_ops_nn, 'situ_glu')
except (ImportError, AttributeError):
    _HAS_CANN_OPS_NN = False

logger = logging.getLogger(__name__)


class SituGLUFunction(torch.autograd.Function):
    """Autograd function wrapping the NPU fused ``situ_glu`` operator."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        dim: int = -1,
        beta: float = 1.0,
        linear_beta: float = 0.0,
        activate_left: bool = True,
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.beta = beta
        ctx.linear_beta = linear_beta
        ctx.activate_left = activate_left
        ctx.save_for_backward(x)

        out = torch.ops.cann_ops_nn.situ_glu(
            x,
            dim=dim,
            beta=beta,
            linear_beta=linear_beta,
            activate_left=activate_left,
        )
        return out

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple:
        (x,) = ctx.saved_tensors

        grad_x = torch.ops.cann_ops_nn.situ_glu_grad(
            grad_output,
            x,
            dim=ctx.dim,
            beta=ctx.beta,
            linear_beta=ctx.linear_beta,
            activate_left=ctx.activate_left,
        )

        return grad_x, None, None, None, None


@register_op('situ_glu', 'npu')
def situ_glu_npu(
    x: torch.Tensor,
    dim: int = -1,
    beta: float = 1.0,
    linear_beta: float = 0.0,
    activate_left: bool = True,
) -> torch.Tensor:
    """NPU fused SituGLU using the ``cann_ops_nn`` fused operator.

    Falls back to the eager implementation when the fused operator is not
    available (e.g. ``cann_ops_nn`` not installed).
    """
    if not _HAS_CANN_OPS_NN:
        from fsdp_turbo.ops.cpu.situ_glu import situ_glu_eager

        log_warning_once(
            logger,
            "cann_ops_nn.situ_glu is not available, falling back to eager implementation",
        )
        return situ_glu_eager(x, dim=dim, beta=beta, linear_beta=linear_beta, activate_left=activate_left)
    return SituGLUFunction.apply(x, dim, beta, linear_beta, activate_left)


__all__ = ['SituGLUFunction', 'situ_glu_npu']

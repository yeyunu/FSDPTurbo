"""Differentiable torch reference for grouped attention output projection."""

import torch

from fsdp_turbo.ops.registry import register_op


@register_op('attention_output_projection', 'cpu')
def attention_output_projection_cpu(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply independent group projections, retaining input dtype and autograd."""
    return torch.einsum('bsgd,grd->bsgr', inputs, weight)

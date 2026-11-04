"""Torch reference for the optional mHC post backend."""

import torch

from fsdp_turbo.ops.registry import register_op


@register_op("tilekernels_mhc_post", "cpu")
def tilekernels_mhc_post_cpu(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Mix input streams into output streams with FP32 arithmetic and autograd."""
    mixed = (comb.float().unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(dim=-3)
    return (post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + mixed).to(x.dtype)

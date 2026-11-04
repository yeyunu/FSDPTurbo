"""Semantic entry points for explicitly enabled TileKernels operators."""

import torch

from fsdp_turbo.ops.registry import dispatch_op


def tilekernels_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Compute mHC post and its four first-order gradients via the registry.

    The owning block's switch selects this entry point, without shape-based
    fallback. NPU uses TileKernels; CPU provides a differentiable reference.
    Unsupported NPU inputs or missing dependencies fail explicitly.
    """
    return dispatch_op("tilekernels_mhc_post", x, residual, post, comb, device_type=residual.device.type)

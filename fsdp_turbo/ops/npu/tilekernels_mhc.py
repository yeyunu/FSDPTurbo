"""Lazy TileKernels mHC post autograd adapter for Ascend 950."""

from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

from fsdp_turbo.ops.registry import register_op


@lru_cache(maxsize=None)
def _validate_device(device: torch.device) -> None:
    if "950" not in torch.accelerator.get_device_name(device):
        raise ValueError("TileKernels mHC post currently requires Ascend 950.")


class _TileKernelsMhcPostFunction(torch.autograd.Function):
    """Bind raw kernels so loss.backward() invokes the kernel backward."""

    @staticmethod
    def forward(ctx, x, residual, post, comb):
        # The raw forward kernel does not register its backward with autograd.
        try:
            from tile_kernels.mhc import mhc_post_fwd
        except ImportError as error:
            raise RuntimeError(
                "TileKernels mHC post requires matching tile-kernels and TileLang "
                "packages for Ascend 950. Disable model.tilekernels.tilekernels_mhc_post "
                "to restore the original backend."
            ) from error
        ctx.save_for_backward(x, residual, post, comb)
        return mhc_post_fwd(x, residual, post.unsqueeze(-1), comb)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        from tile_kernels.mhc import mhc_post_bwd

        x, residual, post, comb = ctx.saved_tensors
        grad_x, grad_residual, grad_post, grad_comb = mhc_post_bwd(
            x, residual, post.unsqueeze(-1), comb, grad_output.contiguous()
        )
        return grad_x, grad_residual, grad_post.squeeze(-1), grad_comb


@register_op("tilekernels_mhc_post", "npu")
def tilekernels_mhc_post_npu(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Run BF16 streams and FP32 mixes, adapting strides without shape routing.

    Contiguous copies and mix casts remain in the autograd graph. All four
    input gradients are returned in input order. Only first-order gradients
    are supported; kernel input errors propagate without native fallback.
    """
    _validate_device(residual.device)
    return _TileKernelsMhcPostFunction.apply(
        x.contiguous(), residual.contiguous(), post.float().contiguous(), comb.float().contiguous()
    )

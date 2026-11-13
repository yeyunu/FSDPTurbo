"""Parameter-free MHC module boundaries for selective activation offload."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fsdp_turbo.ops.mhc import fused_mhc_post, fused_mhc_pre_sinkhorn_premix
from fsdp_turbo.ops.sinkhorn import hc_split_sinkhorn
from fsdp_turbo.ops.tilekernels import tilekernels_mhc_post

from .model_args import ModelArgs


class MHCPre(nn.Module):
    """Compute Sinkhorn coefficients and premix without owning Block parameters.

    Weights are explicit forward inputs so checkpoint keys and FSDP parameter
    ownership remain on the parent decoder. Returns input, post, comb and the
    next sublayer's premix, preserving the reference arithmetic and dtypes.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.use_fused_mhc = args.use_fused_mhc
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.norm_eps = args.norm_eps

    def forward(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collapse [B,S,HC,D] and derive mixing coefficients from explicit weights."""
        if self.use_fused_mhc:
            return fused_mhc_pre_sinkhorn_premix(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                pre_mix,
                hc_mult=self.hc_mult,
                num_iters=self.hc_sinkhorn_iters,
                hc_eps=self.hc_eps,
                norm_eps=self.norm_eps,
            )
        flattened = x.flatten(2).float()
        rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flattened, hc_fn.float()) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
            use_eager=True,
        )
        collapsed = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)
        return collapsed, post, comb, pre


class MHCPost(nn.Module):
    """Expand the sublayer output and mix residuals without registering weights."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.use_fused_mhc = args.use_fused_mhc
        self.use_tilekernels_mhc_post = args.tilekernels.get("tilekernels_mhc_post", False)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Mix [B,S,D] outputs and [B,S,HC,D] residuals with explicit coefficients."""
        if self.use_tilekernels_mhc_post:
            return tilekernels_mhc_post(x, residual, post, comb)
        if self.use_fused_mhc:
            return fused_mhc_post(x, residual, post, comb)
        mixed = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2),
            dim=2,
        )
        return mixed.type_as(x)

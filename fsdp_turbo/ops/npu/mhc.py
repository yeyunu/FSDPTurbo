# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Training adapters for the CANN MHC pre/post operators."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.registry import register_op


class _MhcOps:
    def __init__(self, pre, pre_backward, post, post_backward):
        self.pre = pre
        self.pre_backward = pre_backward
        self.post = post
        self.post_backward = post_backward


@lru_cache(maxsize=1)
def _get_mhc_ops() -> _MhcOps:
    errors = []
    for registration_package in (
        "cann_ops_transformer_customize",
        "cann_ops_transformer_custom",
    ):
        try:
            import_module(registration_package)
        except ImportError as error:
            errors.append(f"{registration_package}: {error!r}")

    try:
        ops = import_module("cann_ops_transformer.ops")
        # Accessing the lazy exports registers the three formal operators.
        getattr(ops, "mhc_pre_sinkhorn_backward")
        getattr(ops, "mhc_post")
        getattr(ops, "mhc_post_backward")
    except (ImportError, AttributeError) as error:
        errors.append(f"cann_ops_transformer.ops: {error!r}")

    try:
        # Full experimental builds expose this module even though it is not in
        # the stable lazy-export table. A customize package may instead have
        # registered the same torch operator while it was imported above.
        import_module("cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn_premix")
    except ImportError as error:
        errors.append(f"mhc_pre_sinkhorn_premix module: {error!r}")

    try:
        namespace = torch.ops.cann_ops_transformer
        return _MhcOps(
            namespace.mhc_pre_sinkhorn_premix.default,
            namespace.mhc_pre_sinkhorn_backward.default,
            namespace.mhc_post.default,
            namespace.mhc_post_backward.default,
        )
    except AttributeError as error:
        raise RuntimeError(
            "use_fused_mhc=True requires mhc_pre_sinkhorn_premix, "
            "mhc_pre_sinkhorn_backward, mhc_post and mhc_post_backward from "
            "matching cann_ops_transformer packages. " + "; ".join(errors)
        ) from error


class _MhcPreSinkhornPremixFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        phi,
        alpha,
        bias,
        premix,
        hc_mult,
        num_iters,
        hc_eps,
        norm_eps,
    ):
        ops = _get_mhc_ops()
        outputs = ops.pre(
            x,
            phi,
            alpha,
            bias,
            premix,
            hc_mult,
            num_iters,
            hc_eps,
            norm_eps,
            True,
        )
        (
            h_in,
            h_post,
            h_res,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
        ) = outputs
        ctx.save_for_backward(
            x,
            phi,
            alpha,
            bias,
            premix,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
        )
        ctx.hc_eps = hc_eps
        ctx.mark_non_differentiable(h_pre)
        return h_in, h_post, h_res, h_pre

    @staticmethod
    def backward(ctx, grad_h_in, grad_h_post, grad_h_res, _grad_h_pre):
        (
            x,
            phi,
            alpha,
            bias,
            premix,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
        ) = ctx.saved_tensors
        if grad_h_in is None:
            grad_h_in = torch.zeros_like(x.select(-2, 0))
        if grad_h_post is None:
            grad_h_post = torch.zeros_like(h_pre)
        if grad_h_res is None:
            grad_h_res = torch.zeros(
                *h_pre.shape[:-1],
                h_pre.shape[-1] ** 2,
                dtype=h_pre.dtype,
                device=h_pre.device,
            )

        # The formal backward assumes h_in used the internally calculated
        # h_pre. With an external premix, h_in's direct gradients are handled
        # below and only h_post/h_res gradients enter the formal backward.
        formal_grad_h_in = torch.zeros_like(grad_h_in)
        grad_x, grad_phi, grad_alpha, grad_bias = _get_mhc_ops().pre_backward(
            formal_grad_h_in,
            grad_h_post.contiguous(),
            grad_h_res.contiguous(),
            x,
            phi,
            alpha,
            bias,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
            ctx.hc_eps,
        )
        direct_grad_x = (grad_h_in.unsqueeze(-2).float() * premix.unsqueeze(-1)).to(x.dtype)
        grad_x = grad_x + direct_grad_x
        grad_premix = (grad_h_in.unsqueeze(-2).float() * x.float()).sum(dim=-1)
        return (
            grad_x,
            grad_phi,
            grad_alpha,
            grad_bias,
            grad_premix,
            None,
            None,
            None,
            None,
        )


class _MhcPostFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, h_res, h_out, h_post):
        output = _get_mhc_ops().post(residual, h_res, h_out, h_post)
        ctx.save_for_backward(residual, h_res, h_out, h_post)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        residual, h_res, h_out, h_post = ctx.saved_tensors
        return _get_mhc_ops().post_backward(grad_output.contiguous(), residual, h_res, h_out, h_post)


def _validate_pre_inputs(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    premix: torch.Tensor,
    hc_mult: int,
    num_iters: int,
) -> None:
    if x.device.type != "npu":
        raise ValueError("Fused MHC operators require NPU tensors")
    if x.ndim != 4 or x.shape[-2] != hc_mult:
        raise ValueError(f"MHC input must have shape [B,S,{hc_mult},D], got {tuple(x.shape)}")
    expected_mix = hc_mult * hc_mult + 2 * hc_mult
    if phi.shape != (expected_mix, hc_mult * x.shape[-1]):
        raise ValueError(f"MHC phi must have shape {(expected_mix, hc_mult * x.shape[-1])}, got {tuple(phi.shape)}")
    if alpha.shape != (3,) or bias.shape != (expected_mix,):
        raise ValueError("MHC alpha/bias shapes do not match hc_mult")
    if premix.shape != x.shape[:-1]:
        raise ValueError(f"MHC premix must have shape {tuple(x.shape[:-1])}, got {tuple(premix.shape)}")
    if hc_mult != 4 or num_iters != 20:
        raise ValueError("Fused MHC operators currently require hc_mult=4 and num_iters=20")


def _supports_mhc_premix_autograd(x: torch.Tensor) -> bool:
    device_name = torch.accelerator.get_device_name(x.device.index)
    normalized_name = device_name.upper().replace("-", "").replace(" ", "")
    return any(model in normalized_name for model in ("ASCEND910B", "ASCEND910C", "ASCEND910_93"))


@register_op("fused_mhc_pre_sinkhorn_premix", "npu")
def fused_mhc_pre_sinkhorn_premix(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    premix: torch.Tensor,
    *,
    hc_mult: int = 4,
    num_iters: int = 20,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse MHC projection, Sinkhorn and collapse using the previous premix."""
    _validate_pre_inputs(x, phi, alpha, bias, premix, hc_mult, num_iters)
    contiguous_inputs = (
        x.contiguous(),
        phi.float().contiguous(),
        alpha.float().contiguous(),
        bias.float().contiguous(),
        premix.float().contiguous(),
    )

    if _supports_mhc_premix_autograd(x):
        # Ascend 910B/910C and Atlas A3 provide the premix operator's
        # registered autograd formula.
        # Calling it directly preserves gradients from all four public outputs.
        h_in, h_post, h_res, h_pre, *_ = _get_mhc_ops().pre(
            *contiguous_inputs,
            hc_mult,
            num_iters,
            hc_eps,
            norm_eps,
            True,
        )
        return h_in, h_post, h_res.unflatten(-1, (hc_mult, hc_mult)), h_pre

    h_in, h_post, h_res, operator_h_pre = _MhcPreSinkhornPremixFunction.apply(
        *contiguous_inputs,
        hc_mult,
        num_iters,
        hc_eps,
        norm_eps,
    )

    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (x, phi, alpha, bias)):
        flat_x = x.flatten(-2).float()
        inv_rms = torch.rsqrt(flat_x.square().mean(dim=-1, keepdim=True) + norm_eps)
        eager_h_pre = F.linear(flat_x, phi[:hc_mult].float()) * inv_rms
        eager_h_pre = torch.sigmoid(eager_h_pre * alpha[0].float() + bias[:hc_mult].float()) + hc_eps
        # Preserve the operator's forward values while routing the cross-layer
        # h_pre gradient through the equivalent small eager projection.
        h_pre = operator_h_pre + eager_h_pre - eager_h_pre.detach()
    else:
        h_pre = operator_h_pre

    return h_in, h_post, h_res.unflatten(-1, (hc_mult, hc_mult)), h_pre


@register_op("fused_mhc_post", "npu")
def fused_mhc_post(
    h_out: torch.Tensor,
    residual: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> torch.Tensor:
    """Fuse residual-stream mixing and broadcast of a sublayer output."""
    if residual.device.type != "npu":
        raise ValueError("Fused MHC operators require NPU tensors")
    if residual.ndim != 4 or h_out.shape != residual.shape[:2] + residual.shape[3:]:
        raise ValueError("MHC post inputs have incompatible shapes")
    if h_post.shape != residual.shape[:-1]:
        raise ValueError("MHC h_post shape does not match the residual stream")
    if h_res.shape != residual.shape[:-2] + residual.shape[-2:-1] * 2:
        raise ValueError("MHC h_res shape does not match the residual stream")
    return _MhcPostFunction.apply(
        residual.contiguous(),
        h_res.float().contiguous(),
        h_out.contiguous(),
        h_post.float().contiguous(),
    )


__all__ = ["fused_mhc_post", "fused_mhc_pre_sinkhorn_premix"]

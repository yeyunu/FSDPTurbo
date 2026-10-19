"""Autograd adapter for CANN ClippedSwiglu / ClippedSwigluGrad."""

from functools import lru_cache
from importlib import import_module
from types import SimpleNamespace

import torch
from torch.autograd.function import once_differentiable

from fsdp_turbo.ops.registry import register_op


@lru_cache(maxsize=1)
def _get_clipped_swiglu_ops():
    try:
        ops = import_module("cann_ops_nn")
        # Resolve both lazy exports before the first forward, so a package
        # without backward support fails before starting training.
        return SimpleNamespace(forward=ops.clipped_swiglu, backward=ops.clipped_swiglu_grad)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Fused clipped SwiGLU requires cann_ops_nn.clipped_swiglu and "
            "cann_ops_nn.clipped_swiglu_grad from a compatible CANN package"
        ) from error


def _operator_args(limit):
    # CANN defaults describe GPT-OSS, whereas DeepSeek uses ordinary SiLU,
    # no linear bias, contiguous halves, and clipping before activation.
    return dict(group_index=None, dim=-1, alpha=1.0, limit=limit, bias=0.0, interleaved=False, clamp_mode=0)


class _ClippedSwiglu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, limit):
        ops = _get_clipped_swiglu_ops()
        ctx.save_for_backward(x)
        ctx.limit = limit
        return ops.forward(x, **_operator_args(limit))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        ops = _get_clipped_swiglu_ops()
        grad_y = grad_output.contiguous()
        kwargs = _operator_args(ctx.limit)
        try:
            grad_x = ops.backward(grad_y, x, **kwargs)
        except TypeError as error:
            # Some CANN packages register a positional schema against a
            # keyword-only Python kernel. Use that same CANN kernel directly;
            # unrelated binding or execution errors must still propagate.
            if "_clipped_swiglu_grad() takes 2 positional arguments" not in str(error):
                raise
            module = import_module("cann_ops_nn.ops.activation.clipped_swiglu_grad.clipped_swiglu_grad")
            implementation = module._clipped_swiglu_grad
            grad_x = implementation(grad_y, x, **kwargs)
            ops.backward = implementation
        return grad_x, None


@register_op("clipped_swiglu", "npu")
def clipped_swiglu_npu(x, limit):
    if x.device.type != "npu" or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Fused clipped SwiGLU requires FP16, BF16 or FP32 NPU tensors")
    # Keep the original, unclipped input for the backward clamp masks.
    return _ClippedSwiglu.apply(x.contiguous(), limit)

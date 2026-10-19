"""Clipped SwiGLU with DeepSeek's contiguous gate/up layout."""

import math
from importlib import import_module

import torch

from fsdp_turbo.ops.registry import dispatch_op


def clipped_swiglu(x: torch.Tensor, limit: float, *, use_eager: bool = False) -> torch.Tensor:
    """Compute ``silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)``.

    Split the last dimension into gate followed by up. Preserve the input
    dtype; callers that require FP32 activation arithmetic must cast first.
    Empty tensors use the reference path because CANN's backward rejects them.
    """
    if not 1 <= x.ndim <= 8 or x.shape[-1] % 2:
        raise ValueError("clipped_swiglu requires a rank 1-8 tensor with an even last dimension")
    if not x.is_floating_point():
        raise TypeError("clipped_swiglu requires floating point input")
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("clipped_swiglu limit must be finite and positive")
    backend = "npu" if x.device.type == "npu" and not use_eager and x.numel() else "cpu"
    if backend == "npu":
        # Do not let optional-backend import failures silently select eager
        # execution when the user explicitly requested fused activation.
        import_module("fsdp_turbo.ops.npu.clipped_swiglu")
    return dispatch_op("clipped_swiglu", x, limit, device_type=backend)

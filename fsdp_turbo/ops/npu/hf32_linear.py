# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Head projection with scoped Ascend HF32 forward and backward matmuls."""

from contextlib import contextmanager
from importlib import import_module

import torch
from torch.nn import functional as F

from fsdp_turbo.ops.registry import register_op


@contextmanager
def _hf32_matmul():
    try:
        npu = import_module("torch_npu").npu
        matmul = npu.matmul
        mode = npu.CubeMathType.USE_HF32
        previous = matmul.cube_math_type
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "head_compute_dtype='hf32' requires torch_npu.npu.matmul.cube_math_type "
            "and torch_npu.npu.CubeMathType.USE_HF32."
        ) from error
    try:
        matmul.cube_math_type = mode
        yield
    finally:
        if previous is None:
            # The property setter accepts enums only. An empty option restores
            # the unset state, including precedence of allow_hf32/environment.
            npu.set_option({"CUBE_MATH_TYPE": ""})
        else:
            matmul.cube_math_type = previous


class _HF32Linear(torch.autograd.Function):
    @staticmethod
    def forward(x, weight):
        # HF32 is a math mode for FP32 tensors, not an autocast dtype.
        with torch.autocast(device_type=x.device.type, enabled=False), _hf32_matmul():
            return F.linear(x, weight)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_x = grad_weight = None
        # Backward runs after the forward scope exits, so it needs its own scope.
        with torch.autocast(device_type=x.device.type, enabled=False), _hf32_matmul():
            grad_output = grad_output.float()
            if ctx.needs_input_grad[0]:
                grad_x = grad_output.matmul(weight)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.reshape(-1, weight.size(0)).t().matmul(x.reshape(-1, weight.size(1)))
        return grad_x, grad_weight


@register_op("hf32_linear", "npu")
def hf32_linear_npu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Project NPU inputs in HF32 with FP32 output and independent storage dtype."""
    if x.device.type != "npu" or weight.device.type != "npu":
        raise ValueError("head_compute_dtype='hf32' requires NPU tensors.")
    return _HF32Linear.apply(x.float(), weight.float())


__all__ = ["hf32_linear_npu"]

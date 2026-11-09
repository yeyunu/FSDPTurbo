# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Reference MoonEP grouped-GEMM primitives.

These implementations are intentionally simple: they read the group boundaries
back to the host and loop. They exist as the numerical oracle for the unit
tests and as the fallback when neither ``torch._grouped_mm`` nor
``torch_npu.npu_grouped_matmul`` is available. The accelerator paths must stay
bit-comparable with these within GEMM tolerance.
"""

from __future__ import annotations

import torch

from fsdp_turbo.ops.registry import register_op


def _group_slices(group_ends: torch.Tensor):
    """Yield ``(group_index, start, end)`` for every group, including empty ones."""
    ends = group_ends.detach().to(device="cpu", dtype=torch.int64).tolist()
    start = 0
    for group, end in enumerate(ends):
        if end < start:
            raise RuntimeError(f"MoonEP group ends must be non-decreasing; group {group} ends at {end} after {start}.")
        yield group, start, end
        start = end


def _semantic(weights: torch.Tensor, weights_transposed: bool) -> torch.Tensor:
    """View the stored weights in the ``[G, out, in]`` convention."""
    return weights.transpose(-2, -1) if weights_transposed else weights


@register_op("moonep_grouped_matmul", "cpu")
def moonep_grouped_matmul_cpu(inputs, group_ends, weights, weights_transposed=False):
    weights = _semantic(weights, weights_transposed)
    output = inputs.new_zeros((inputs.shape[0], weights.shape[1]))
    for group, start, end in _group_slices(group_ends):
        if end > start:
            output[start:end] = inputs[start:end] @ weights[group].transpose(0, 1)
    return output


@register_op("moonep_grouped_matmul_dgrad", "cpu")
def moonep_grouped_matmul_dgrad_cpu(grad_output, group_ends, weights, weights_transposed=False):
    weights = _semantic(weights, weights_transposed)
    grad_inputs = grad_output.new_zeros((grad_output.shape[0], weights.shape[2]))
    for group, start, end in _group_slices(group_ends):
        if end > start:
            grad_inputs[start:end] = grad_output[start:end] @ weights[group]
    return grad_inputs


@register_op("moonep_grouped_matmul_wgrad", "cpu")
def moonep_grouped_matmul_wgrad_cpu(grad_output, inputs, group_ends, group_ends_host=None, weights_transposed=False):
    # The reference always loops, so it has no use for either hint.
    left, right = (inputs, grad_output) if weights_transposed else (grad_output, inputs)
    grad_weights = inputs.new_zeros((group_ends.numel(), left.shape[1], right.shape[1]))
    for group, start, end in _group_slices(group_ends):
        if end > start:
            grad_weights[group] = left[start:end].transpose(0, 1) @ right[start:end]
    return grad_weights

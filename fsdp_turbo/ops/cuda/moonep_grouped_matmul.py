# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CUDA MoonEP grouped-GEMM primitives built on PyTorch's grouped matmul.

The forward and the data gradient use the 2-D x 3-D form, which takes MoonEP's
int32 cumulative ends directly.

The weight gradient would need the 2-D x 2-D form, where the offsets partition
the *contraction* dimension; a ragged K split whose last segment is not
tile-aligned has been observed to hang the kernel rather than fail. This file
loops over the groups instead, which MoonEP's compact group set keeps to
``2 * E/R`` iterations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.registry import register_op


def resolve_grouped_mm():
    """Return whichever grouped-GEMM entry point this PyTorch build exposes.

    Naming only the public alias is what made every expert forward abort with an
    ``AttributeError`` on builds that predate it, so neither name is hard-coded.
    """
    for module, name in ((F, "grouped_mm"), (torch, "_grouped_mm")):
        function = getattr(module, name, None)
        if function is not None:
            return function
    return None


def grouped_mm(mat_a, mat_b, offs):
    """``mat_a @ mat_b`` per group, with int32 cumulative end offsets."""
    function = resolve_grouped_mm()
    if function is None:
        raise RuntimeError(
            "MoonEP's expert GEMM needs a grouped matmul operator, but neither "
            "torch.nn.functional.grouped_mm nor torch._grouped_mm exists in PyTorch "
            f"{torch.__version__}. Use dispatcher='eager', or upgrade to a build that "
            "provides one of them."
        )
    return function(mat_a, mat_b, offs=offs)


def _offsets(group_ends: torch.Tensor) -> torch.Tensor:
    # The operator requires int32 offsets; cu_seqlens is already int32 in MoonEP,
    # so this is a no-op cast in the common case.
    return group_ends if group_ends.dtype == torch.int32 else group_ends.to(torch.int32)


@register_op("moonep_grouped_matmul", "cuda")
def moonep_grouped_matmul_cuda(inputs, group_ends, weights, weights_transposed=False):
    # The kernel wants the right operand as [G, in, out]. A transposed layout is
    # already stored that way, so it needs no view at all; the [G, out, in]
    # convention needs one, which keeps the VMM mapping intact and costs nothing
    # because the base tensor is contiguous.
    operand = weights if weights_transposed else weights.transpose(-2, -1)
    return grouped_mm(inputs, operand, _offsets(group_ends))


@register_op("moonep_grouped_matmul_dgrad", "cuda")
def moonep_grouped_matmul_dgrad_cuda(grad_output, group_ends, weights, weights_transposed=False):
    # This one contracts over `out`, so the operand must be [G, out, in] -- the
    # mirror image of the forward.
    operand = weights.transpose(-2, -1) if weights_transposed else weights
    return grouped_mm(grad_output, operand, _offsets(group_ends))


@register_op("moonep_grouped_matmul_wgrad", "cuda")
def moonep_grouped_matmul_wgrad_cuda(grad_output, inputs, group_ends, group_ends_host=None, weights_transposed=False):
    # Per-group form. `group_ends_host` lets the caller amortize the one
    # device-to-host read across the two projections of a layer.
    ends = group_ends_host
    if ends is None:
        ends = group_ends.detach().to(device="cpu", dtype=torch.int64).tolist()
    # Swapping the operands emits the transpose of the result, which is exactly
    # the stored layout when weights_transposed.
    left, right = (inputs, grad_output) if weights_transposed else (grad_output, inputs)
    grad_weights = torch.empty(
        len(ends),
        left.shape[1],
        right.shape[1],
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    start = 0
    for group, end in enumerate(ends):
        if end > start:
            torch.mm(
                left[start:end].transpose(0, 1),
                right[start:end],
                out=grad_weights[group],
            )
        else:
            grad_weights[group].zero_()
        start = end
    return grad_weights

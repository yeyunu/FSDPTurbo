# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch.nn.functional as F
from torch.autograd.function import FunctionCtx
from fsdp_turbo.ops.registry import register_op


def _grouped_mm(input_tensor: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    grouped_mm = getattr(F, 'grouped_mm', None)
    if grouped_mm is None:
        grouped_mm = getattr(torch, '_grouped_mm', None)
    if grouped_mm is None:
        raise RuntimeError('CUDA grouped matmul requires torch.nn.functional.grouped_mm or torch._grouped_mm.')
    return grouped_mm(input_tensor, mat_b, offs=offs)


class GroupedMatmulCUDA(torch.autograd.Function):
    """
    CUDA grouped matmul using the public API or its PyTorch 2.9 predecessor.

    Weights follow the linear-layer convention [num_groups, output_dim, input_dim]
    (i.e. [out_features, in_features] per expert). The fused kernel expects the
    right operand in [num_groups, K, N] layout, so weights are transposed before
    the forward call.
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx, input_tensor: torch.Tensor, weights: torch.Tensor, m_split: torch.Tensor
    ) -> torch.Tensor:
        # offs[i] marks the end of group i; the kernel requires int32 offsets.
        offs = torch.cumsum(m_split, dim=0).to(torch.int32)
        ctx.save_for_backward(input_tensor, weights, offs)
        if input_tensor.shape[0] == 0:
            return input_tensor.new_empty((0, weights.shape[1]))

        # out = input @ weight.T  ->  mat_b = [num_groups, K, N]
        mat_b = weights.transpose(-2, -1)
        output = _grouped_mm(input_tensor, mat_b, offs)

        return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        input_tensor, weights, offs = ctx.saved_tensors
        if grad_output.shape[0] == 0:
            return torch.zeros_like(input_tensor), torch.zeros_like(weights), None
        grad_output = grad_output.contiguous()

        # grad_input = grad_output @ weight  (per group) -> mat_b = [num_groups, N, K]
        grad_input = _grouped_mm(grad_output, weights, offs)

        # grad_weight = grad_output_g.T @ input_g (per group) -> [num_groups, N, K]
        grad_weight = torch.zeros_like(weights)
        start = 0
        for group_index in range(weights.shape[0]):
            end = offs[group_index].item()
            grad_weight[group_index] = grad_output[start:end].T @ input_tensor[start:end]
            start = end

        return grad_input, grad_weight, None


@register_op('grouped_matmul', 'cuda')
def grouped_matmul_cuda(
    inputs: torch.Tensor, m_split: torch.Tensor, weights: torch.Tensor, weights_layout: str = 'linear'
) -> torch.Tensor:
    """Compute ragged linear projections without changing operand precision.

    Uses ``F.grouped_mm`` when present and ``torch._grouped_mm`` on PyTorch2.9.
    The returned tensor and both first-order adjoints preserve the input dtype.
    Weights use ``[groups, output_dim, input_dim]`` by default, or
    ``[groups, input_dim, output_dim]`` when ``weights_layout='input_major'``.
    Weight gradients retain the caller's layout.
    """
    if weights_layout == 'input_major':
        weights = weights.transpose(-2, -1)
    return GroupedMatmulCUDA.apply(inputs, weights, m_split)

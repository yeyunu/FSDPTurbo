# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch.nn.functional as F
from fsdp_turbo.ops.registry import register_op


class GroupedMatmulCUDA(torch.autograd.Function):
    """
    CUDA fused grouped matmul backed by torch.nn.functional.grouped_mm.

    Weights follow the linear-layer convention [num_groups, output_dim, input_dim]
    (i.e. [out_features, in_features] per expert). The fused kernel expects the
    right operand in [num_groups, K, N] layout, so weights are transposed before
    the forward call.
    """

    @staticmethod
    def forward(ctx, input_tensor, weights, m_split):
        # offs[i] marks the end of group i; the kernel requires int32 offsets.
        offs = torch.cumsum(m_split, dim=0).to(torch.int32)

        # out = input @ weight.T  ->  mat_b = [num_groups, K, N]
        mat_b = weights.transpose(-2, -1)
        output = F.grouped_mm(input_tensor, mat_b, offs=offs)

        ctx.save_for_backward(input_tensor, weights, offs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weights, offs = ctx.saved_tensors

        # grad_input = grad_output @ weight  (per group) -> mat_b = [num_groups, N, K]
        grad_input = F.grouped_mm(grad_output, weights, offs=offs)

        # grad_weight = grad_output_g.T @ input_g (per group) -> [num_groups, N, K]
        grad_weight = torch.zeros_like(weights)
        start = 0
        for group_index in range(weights.shape[0]):
            end = offs[group_index].item()
            grad_weight[group_index] = grad_output[start:end].T @ input_tensor[start:end]
            start = end

        return grad_input, grad_weight, None


@register_op('grouped_matmul', 'cuda')
def grouped_matmul_cuda(inputs, m_split, weights, weights_layout="linear"):
    if weights_layout == "input_major":
        weights = weights.transpose(-2, -1)
    return GroupedMatmulCUDA.apply(inputs, weights, m_split)

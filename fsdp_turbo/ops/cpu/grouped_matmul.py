# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2021, OpenAI.

import torch
from fsdp_turbo.ops.registry import register_op


class _GroupedMatmulCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, m_split, weights):
        ctx.matmul_dtype = inputs.dtype
        ctx.group_list = [0] + torch.cumsum(m_split, dim=0).tolist()
        ctx.save_for_backward(inputs, weights)
        output = torch.zeros((inputs.size(0), weights.size(1)), dtype=inputs.dtype, device=inputs.device)
        for group_index in range(len(ctx.group_list) - 1):
            start_index = ctx.group_list[group_index]
            end_index = ctx.group_list[group_index + 1]
            if start_index == end_index:
                continue
            group_output = torch.matmul(inputs[start_index:end_index], weights[group_index].transpose(0, 1))
            ctx.matmul_dtype = group_output.dtype
            output[start_index:end_index] = group_output
            del group_output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        grad_inputs = torch.zeros_like(inputs) if ctx.needs_input_grad[0] else None
        grad_weights = torch.zeros_like(weights) if ctx.needs_input_grad[2] else None
        for group_index in range(len(ctx.group_list) - 1):
            start_index = ctx.group_list[group_index]
            end_index = ctx.group_list[group_index + 1]
            if start_index == end_index:
                continue
            group_grad_output = grad_output[start_index:end_index].to(ctx.matmul_dtype)
            if grad_inputs is not None:
                grad_inputs[start_index:end_index] = torch.matmul(
                    group_grad_output, weights[group_index].to(ctx.matmul_dtype)
                )
            if grad_weights is not None:
                grad_weights[group_index] = torch.matmul(
                    group_grad_output.transpose(0, 1), inputs[start_index:end_index].to(ctx.matmul_dtype)
                )
        return grad_inputs, None, grad_weights


@register_op('grouped_matmul', 'cpu')
def grouped_matmul_cpu(inputs, m_split, weights):
    """
    Grouped matrix multiplication.

    Weights follow the linear-layer convention: [num_groups, output_dim, input_dim]
    (i.e. [out_features, in_features] per expert). They are always transposed to
    [num_groups, input_dim, output_dim] before the matmul. This avoids the
    unreliable shape-based detection that breaks when input_dim == output_dim.

    Forward and first-order backward write into preallocated buffers without
    building per-expert CopySlices backward chains. Empty groups have zero
    weight gradients, and only requested input/weight gradients are allocated.

    Args:
        inputs: Tensor of shape [batch_size, input_dim]
        m_split: Tensor of group sizes that sum to batch_size
        weights: Weight tensor of shape [num_groups, output_dim, input_dim]

    Returns:
        Tensor of shape [batch_size, output_dim]
    """
    return _GroupedMatmulCPU.apply(inputs, m_split, weights)

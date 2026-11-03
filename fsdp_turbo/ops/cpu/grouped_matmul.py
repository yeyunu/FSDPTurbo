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
def grouped_matmul_cpu(inputs, m_split, weights, weights_layout="linear"):
    """
    Grouped matrix multiplication.

    Weights follow the linear-layer convention: [num_groups, output_dim, input_dim]
    (i.e. [out_features, in_features] per expert). They are always transposed to
    [num_groups, input_dim, output_dim] before the matmul. This avoids the
    unreliable shape-based detection that breaks when input_dim == output_dim.
    ``weights_layout="input_major"`` accepts [num_groups, input_dim, output_dim]
    and transposes it to the linear convention first; the returned weight
    gradient keeps the caller's layout through autograd.

    Forward and first-order backward write into preallocated buffers without
    building per-expert CopySlices backward chains. Empty groups have zero
    weight gradients, and only requested input/weight gradients are allocated.

    Args:
        inputs: Tensor of shape [batch_size, input_dim]
        m_split: Tensor of group sizes that sum to batch_size
        weights: Weight tensor of shape [num_groups, output_dim, input_dim]
        weights_layout: "linear" (default) or "input_major"

    Returns:
        Tensor of shape [batch_size, output_dim]
    """
    if weights_layout == "input_major":
        weights = weights.transpose(1, 2)
    return _GroupedMatmulCPU.apply(inputs, m_split, weights)


@register_op('w4a16_grouped_matmul', 'cpu')
def w4a16_grouped_matmul_cpu(
    x,
    weights,
    bias=None,
    group_list=None,
    group_type=0,
    output_dtype=None,
    group_list_type=0,
    split_item=2,
):
    """Low-precision (W4A16) grouped matmul CPU fallback.

    W4A16 relies on the NPU low-precision grouped matmul kernel and the
    NPU float4 dtype, so there is no meaningful CPU implementation. This
    fallback keeps the module importable on non-NPU devices and raises a
    clear error if the W4A16 path is actually exercised on CPU.
    """
    raise NotImplementedError(
        "W4A16 low-precision grouped matmul is only supported on NPU. No CPU fallback is available."
    )

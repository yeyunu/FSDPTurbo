# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from fsdp_turbo.ops import dispatch_op


def grouped_matmul(inputs, m_split, weights, use_eager=False):
    """
    Grouped matrix multiplication with automatic device dispatch.

    Args:
        inputs: Input tensor
        m_split: Group sizes
        weights: Weight tensor
        use_eager: If False, use fused (optimized) implementation for current device.
                   If True, use CPU implementation.

    Returns:
        Result of grouped matrix multiplication
    """
    device_type = 'cpu' if use_eager else None  # Use CPU implementation if not fused
    return dispatch_op('grouped_matmul', inputs, m_split, weights, device_type=device_type)


def all2all_grouped_matmul(
    inputs, weights, group, send_counts, recv_counts, shared_inputs=None, shared_weight=None, use_eager=False
):
    """
    All-to-all communication followed by grouped matrix multiplication.

    Args:
        inputs: Input tensor
        weights: Weight tensor
        group: Process group for distributed communication
        send_counts: Counts of elements to send to each rank
        recv_counts: Counts of elements to receive from each rank
        shared_inputs: Optional shared expert inputs
        shared_weight: Optional shared expert weight
        use_eager: If False, use fused (optimized) implementation for current device.
                   If True, use CPU implementation.

    Returns:
        Output tensor(s) - expert output and optionally shared expert output
    """
    device_type = 'cpu' if use_eager else None  # Use CPU implementation if not fused
    return dispatch_op(
        'all2all_grouped_matmul',
        inputs,
        weights,
        group,
        send_counts,
        recv_counts,
        shared_inputs,
        shared_weight,
        device_type=device_type,
    )


def grouped_matmul_all2all(
    inputs, weights, group, send_counts, recv_counts, shared_inputs=None, shared_weight=None, use_eager=False
):
    """
    Grouped matrix multiplication followed by all-to-all communication.

    Args:
        inputs: Input tensor
        weights: Weight tensor
        group: Process group for distributed communication
        send_counts: Counts of elements to send to each rank
        recv_counts: Counts of elements to receive from each rank
        shared_inputs: Optional shared expert inputs
        shared_weight: Optional shared expert weight
        use_eager: If False, use fused (optimized) implementation for current device.
                   If True, use CPU implementation.

    Returns:
        Output tensor(s) - expert output and optionally shared expert output
    """
    device_type = 'cpu' if use_eager else None  # Use CPU implementation if not fused
    return dispatch_op(
        'grouped_matmul_all2all',
        inputs,
        weights,
        group,
        send_counts,
        recv_counts,
        shared_inputs,
        shared_weight,
        device_type=device_type,
    )


def permute(tokens, indices, num_out_tokens=None, use_eager=False):
    """
    Token permutation for MoE (Mixture of Experts).

    Args:
        tokens: Input tokens tensor [num_tokens, hidden_dim]
        indices: Expert indices tensor [num_tokens] or [num_tokens, topk]
        use_eager: If False, use fused (optimized) implementation for current device.
                   If True, use CPU implementation.

    Returns:
        permuted_tokens: Permuted tokens
        sorted_indices: Indices for unpermutation
    """
    device_type = 'cpu' if use_eager else None  # Use CPU implementation if not fused
    return dispatch_op('permute', tokens, indices, num_out_tokens, device_type=device_type)


def unpermute(permuted_tokens, sorted_indices, probs=None, restore_shape=None, use_eager=False):
    """
    Token unpermutation for MoE (Mixture of Experts).

    Args:
        permuted_tokens: Permuted tokens tensor
        sorted_indices: Indices from permutation operation
        probs: Optional probability weights [num_tokens, topk]
        use_eager: If False, use fused (optimized) implementation for current device.
                   If True, use CPU implementation.

    Returns:
        unpermuted_tokens: Unpermuted tokens
    """
    device_type = 'cpu' if use_eager else None  # Use CPU implementation if not fused
    # if permuted_tokens.size(0) != sorted_indices.numel():
    #     raise AssertionError(f'permuted tokens({permuted_tokens.size(0)}) != sorted indices({sorted_indices.size()})')
    return dispatch_op('unpermute', permuted_tokens, sorted_indices, probs, restore_shape, device_type=device_type)


def swiglu(inputs, dim=-1, fused=True):
    """
    SwiGLU activation function.

    Args:
        inputs: Input tensor
        dim: Dimension along which to split the tensor
        fused: If True, use fused (optimized) implementation for current device.
               If False, use CPU implementation.

    Returns:
        Output tensor after applying SwiGLU activation
    """
    device_type = 'cpu' if not fused else None  # Use CPU implementation if not fused
    return dispatch_op('swiglu', inputs, dim, device_type=device_type)


def clamp_swiglu(inputs, dim=-1, fused=True, limit=0.0):
    y1, y2 = torch.chunk(inputs, 2, dim=-1)
    if limit > 0:
        y1 = y1.clamp(min=None, max=limit)
        y2 = y2.clamp(min=-limit, max=limit)
    return swiglu(torch.cat([y1, y2], dim=-1), dim=dim, fused=fused)


def eager_clipped_swiglu(inputs, dim=-1, swiglu_alpha=1.702, swiglu_limit=7.0):
    """MiniMax clipped SwiGLU eager"""
    if dim < 0:
        dim = inputs.dim() + dim
    if inputs.numel() == 0 or inputs.shape[dim] % 2 != 0:
        raise ValueError("clipped_swiglu input dim must be even and non-empty")
    if swiglu_limit < 0:
        raise ValueError(f"swiglu_limit must be >= 0, got {swiglu_limit}")

    gate, up = torch.chunk(inputs, 2, dim=dim)
    gate = gate.clamp(max=swiglu_limit)
    up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return (up + 1.0) * gate * torch.sigmoid(gate * swiglu_alpha)


def clipped_swiglu(inputs, dim=-1, fused=True, swiglu_alpha=1.702, swiglu_limit=7.0):
    return eager_clipped_swiglu(inputs, dim=dim, swiglu_alpha=swiglu_alpha, swiglu_limit=swiglu_limit)

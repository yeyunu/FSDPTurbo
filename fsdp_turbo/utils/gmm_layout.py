# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Conversions between the stored weight layout and the grouped-matmul op
layout [E, K, N]. All layout changes go through here.

Stored layouts: 2D [E*K, N] / [E*N, K], 3D [E, N, K] / [E, K, N] (op layout).
A stored tensor needs a transpose to reach [E, K, N] iff the contraction
dim ``k`` is not already its dim-1 (after viewing as 3D).
"""

import torch


def _stored_transposed(shape: tuple, num_experts: int, k: int) -> bool:
    """True when a stored tensor of ``shape`` needs a transpose to reach op
    layout [E, K, N]: 2D [E*N, K] or 3D [E, N, K].
    """
    if len(shape) == 2:
        return shape[0] != num_experts * k
    return shape[1] != k


def to_gmm_3d(weight: torch.Tensor, num_experts: int, k: int) -> torch.Tensor:
    """Stored layout -> op layout [E, K, N]. Zero-copy view when no transpose
    is needed (op layout [E, K, N] and 2D [E*K, N] both view directly).
    """
    w3d = weight.view(num_experts, -1, weight.shape[-1])
    if _stored_transposed(weight.shape, num_experts, k):
        w3d = w3d.transpose(1, 2).contiguous()
    return w3d


def from_gmm_3d(weight: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """Op layout [E, K, N] -> stored layout (inverse of to_gmm_3d).

    The contraction dim ``k`` is taken from the op-layout tensor itself
    (dim-1), so no caller-side ``k`` is needed.
    """
    if _stored_transposed(original_shape, weight.shape[0], weight.shape[1]):
        return weight.transpose(1, 2).contiguous().reshape(original_shape)
    if len(original_shape) == 2:
        return weight.reshape(original_shape)
    return weight

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Conversions between the stored weight layout (2D [E*K, N] / 3D [E, N, K])
and the grouped-matmul op layout [E, K, N]. All layout changes go through here.
"""

import torch


def to_gmm_3d(weight: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Stored layout (2D [E*K, N] or 3D [E, N, K]) -> op layout [E, K, N]."""
    if weight.dim() == 2:
        return weight.view(num_experts, weight.shape[0] // num_experts, -1)
    return weight.view(num_experts, -1, weight.shape[2]).transpose(1, 2).contiguous()


def from_gmm_3d(weight: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """Op layout [E, K, N] -> stored layout (inverse of to_gmm_3d)."""
    if len(original_shape) == 2:
        return weight.reshape(original_shape)
    return weight.transpose(1, 2).contiguous().reshape(original_shape)

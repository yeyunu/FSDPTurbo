# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import os
import torch

# `torch.bincount` is used for counting expert assignments in EP. This is slower than `torch.histc` but more stable.
# `torch.histc` is not a deterministic operation and `torch.bincount` is deterministic.
# Set environment variable export MM_EP_BINCOUNT_DISPATCH=1 to activate.
EP_BINCOUNT_DISPATCH = os.environ.get('EP_BINCOUNT_DISPATCH', '0') == '1'


def get_num_global_tokens_per_local_expert(
    selected_experts: torch.Tensor,
    num_global_experts: int,
) -> torch.Tensor:
    if EP_BINCOUNT_DISPATCH:
        num_local_tokens_per_expert = torch.bincount(selected_experts.reshape(-1), minlength=num_global_experts)
    else:
        num_local_tokens_per_expert = torch.histc(
            selected_experts.reshape(-1), bins=num_global_experts, min=0, max=num_global_experts
        )
    return num_local_tokens_per_expert


def normalize_expert_args(top_k_index, top_k_weights):
    """
    Ensure top_k_index is integer tensor (indices) and top_k_weights is float tensor (weights).
    Swap if necessary and adjust dimensions if needed.

    Args:
        top_k_index: Tensor that could be either indices or weights
        top_k_weights: Tensor that could be either weights or indices

    Returns:
        (correct_top_k_index, correct_top_k_weights)
    """
    # Swap if top_k_index is floating point (actually weights)
    if torch.is_floating_point(top_k_index):
        top_k_index, top_k_weights = top_k_weights, top_k_index

    # Ensure weights have the same shape as indices
    if top_k_weights.size() != top_k_index.size():
        # Gather weights using indices (assume top_k_weights has shape [batch_size, num_experts])
        # and top_k_index has shape [batch_size, top_k]
        top_k_weights = top_k_weights.gather(1, top_k_index)

    return top_k_index, top_k_weights


def fixed_router_for_debug(top_k_index: torch.Tensor, top_k_weights: torch.Tensor, num_global_experts: int):
    """
    Debug-only helper that replaces router output with deterministic uniform routing.
    Each token is assigned to experts in round-robin fashion with equal weights.

    Args:
        top_k_index: Original top-k expert indices [num_tokens, top_k]
        top_k_weights: Original top-k weights [num_tokens, top_k]
        num_global_experts: Total number of experts

    Returns:
        (fixed_top_k_index, fixed_top_k_weights)
    """
    is_1d = top_k_index.dim() == 1
    num_tokens = top_k_index.shape[0]
    top_k = 1 if is_1d else top_k_index.shape[1]

    with torch.no_grad():
        base = torch.arange(num_tokens, device=top_k_index.device, dtype=top_k_index.dtype)
        offset = torch.arange(top_k, device=top_k_index.device, dtype=top_k_index.dtype)
        fixed_index = (base.unsqueeze(1) + offset.unsqueeze(0)) % num_global_experts
        if is_1d:
            fixed_index = fixed_index.squeeze(1)
        fixed_weights = torch.full_like(top_k_weights, 1.0 / top_k)

    return fixed_index, fixed_weights

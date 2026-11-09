# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import os
import torch
from typing import NamedTuple, Optional

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


# ---------------------------------------------------------------------------
# MoE a2a split-size cache: activation checkpointing re-runs dispatch_preprocess
# during recompute (the counting op, the EP all-gather and the blocking D2H
# copies), so the forward caches the split sizes and recompute reuses them.
# Recompute detection matches FSDP2 _fsdp_common.is_bw:
# torch._C._current_graph_task_id() != -1.
_DISPATCH_SPLIT_CACHE_ATTR = "_fsdp_turbo_a2a_split_cache"
# Eager and mc2 payloads have different shapes, and mc2 may fall back to eager
# within one forward, so they must not share one stack.
MC2_DISPATCH_SPLIT_CACHE_ATTR = "_fsdp_turbo_a2a_split_cache_mc2"
# Attached by expert_parallelize_modules at EP init; absent means disabled.
DISPATCH_SPLIT_POLICY_ATTR = "_fsdp_turbo_a2a_split_policy"


class DispatchSplitPolicy(NamedTuple):
    """a2a split-cache settings carried by each expert container."""

    enabled: bool = False
    cache_cap: int = 8


def is_backward_recompute() -> bool:
    return torch._C._current_graph_task_id() != -1


def is_dispatch_split_reuse_enabled(expert_module: Optional[torch.nn.Module]) -> bool:
    """Whether this expert container carries an enabled a2a split-reuse policy."""
    if expert_module is None:
        return False
    policy = getattr(expert_module, DISPATCH_SPLIT_POLICY_ATTR, None)
    return policy is not None and policy.enabled


def push_dispatch_splits(
    expert_module: Optional[torch.nn.Module],
    splits: tuple,
    cache_attr: str = _DISPATCH_SPLIT_CACHE_ATTR,
) -> None:
    """Cache this expert module's a2a split sizes (normal grad-enabled forward only)."""
    if expert_module is None or not torch.is_grad_enabled() or is_backward_recompute():
        return
    policy = getattr(expert_module, DISPATCH_SPLIT_POLICY_ATTR, None)
    if policy is None or not policy.enabled:
        return
    cache = getattr(expert_module, cache_attr, None)
    if cache is None:
        cache = []
        setattr(expert_module, cache_attr, cache)
    if len(cache) >= policy.cache_cap:
        cache.clear()
    cache.append(splits)


def pop_dispatch_splits(
    expert_module: Optional[torch.nn.Module],
    cache_attr: str = _DISPATCH_SPLIT_CACHE_ATTR,
) -> Optional[tuple]:
    """Take the forward-cached a2a split sizes during recompute; None otherwise.

    Returns None when nothing is cached (abnormal paths); the caller then
    falls back to the original computation, preserving semantics.
    """
    if expert_module is None or not is_backward_recompute():
        return None
    policy = getattr(expert_module, DISPATCH_SPLIT_POLICY_ATTR, None)
    if policy is None or not policy.enabled:
        return None
    cache = getattr(expert_module, cache_attr, None)
    if not cache:
        return None
    return cache.pop()


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

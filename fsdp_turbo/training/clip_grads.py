# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""Gradient norm computation and clipping utilities.

This module provides decoupled, FSDP-aware gradient norm operations:
- ``compute_grad_norm``: Compute the total gradient norm (L2 or L-inf) without clipping.
``edp_efsdp``).
- ``clip_grad_norm``: Compute the gradient norm and clip in one call.

DTensor (``torch.distributed.tensor.DTensor``) gradients are handled
transparently: they are converted to local shards via ``.to_local()``
before computation, and the total norm is synchronized across the
sharding group(s) via ``all_reduce``.
"""

from collections import defaultdict
from typing import Optional, Union

import torch
from torch import inf

try:
    from torch.distributed._tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard

    try:
        from torch.distributed.tensor.placement_types import _StridedShard
    except ImportError:
        _StridedShard = None

    _HAS_DTENSOR = True
except ImportError:
    _HAS_DTENSOR = False
    _StridedShard = None


# ------------------------------------------------------------------
# DTensor helpers
# ------------------------------------------------------------------


def _to_local_if_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local shard if *tensor* is a DTensor, else return as-is."""
    if _HAS_DTENSOR and isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _get_shard_groups_if_dtensor(
    tensor: torch.Tensor,
    current_group: Optional[torch.distributed.ProcessGroup] = None,
) -> list:
    """Return the process groups along which *tensor* is sharded.

    For regular tensors, returns ``[current_group]`` if *current_group*
    is not None, else ``[]``.

    For DTensors, inspects the tensor's ``placements`` and returns the
    process group for **every** mesh dimension that is ``Shard``.
    Gradient norms must be all-reduced across each of these groups to
    obtain the correct global total (e.g. both the FSDP and TP
    dimensions for an HSDP+TP mesh).
    """
    if not (_HAS_DTENSOR and isinstance(tensor, DTensor)):
        return [current_group] if current_group is not None else []

    mesh = tensor.device_mesh

    # Collect every mesh dimension that is Shard in the placements.
    # Each such dimension splits the tensor across ranks and therefore
    # requires an all-reduce for norm sync.
    groups = []
    for dim_idx, placement in enumerate(tensor.placements):
        is_shard = isinstance(placement, Shard)
        if not is_shard and _StridedShard is not None:
            is_shard = isinstance(placement, _StridedShard)
        if is_shard:
            groups.append(mesh.get_group(dim_idx))
    return groups


def compute_grad_norm(
    parameters: Union[list, torch.Tensor],
    norm_type: Union[int, float] = 2.0,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Compute the total p-norm of gradients across parameters.

    Gradients are cast to FP32 for the norm computation to avoid
    precision loss with BF16/FP16 gradients.  DTensor gradients are
    converted to their local shards before computation.

    When *group* is provided (or DTensor gradients are found), the
    partial norm is **all-reduced** across the sharding group(s) so
    that every rank sees the same total norm.  Parameters sharded on
    different mesh dimensions (e.g. dense on fsdp, experts on efsdp)
    are reduced independently.

    Args:
        parameters: Model parameters whose ``.grad`` will be read.
        norm_type: Type of the p-norm. Use ``float('inf')`` for the
            infinity norm. Defaults to 2.0 (L2 norm).
        group: Process group for all-reducing the norm across
            sharding ranks.  If ``None`` and DTensor gradients
            are present, the group is inferred from the DTensors.
    Returns:
        The total gradient norm as a Python float.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    dev = torch.accelerator.current_device()
    # Group parameters by their shard-group signature so that gradients
    # sharing the same set of sharded mesh dimensions can be aggregated
    # locally before a single all-reduce per group.  This is required
    # because a parameter that is *Replicate* along a dimension must NOT
    # be summed across that dimension (otherwise its contribution is
    # multiplied by the group size).  Each group is reduced only along
    # the dimensions its members are actually sharded on.
    grouped_grads = defaultdict(list)
    grouped_groups = {}
    for p in parameters:
        if p.grad is not None:
            local_g = _to_local_if_dtensor(p.grad).detach()
            # With cpu_offload, gradients are on CPU; pinned memory allows non_blocking H2D
            if local_g.device != dev:
                local_g = local_g.to(dev, non_blocking=True)
            local_g = local_g.float()
            shard_groups = _get_shard_groups_if_dtensor(p.grad, group)
            sig = tuple(id(g) for g in shard_groups)
            grouped_grads[sig].append(local_g)
            grouped_groups[sig] = shard_groups

    if not grouped_grads:
        return 0.0

    norm_type = float(norm_type)

    if norm_type == inf:
        # For each group: local max over its members, all-reduce MAX across
        # every sharded dimension, then take the global max across groups.
        global_max = torch.zeros((), dtype=torch.float, device=dev)
        for sig, grads in grouped_grads.items():
            local_max = max(g.abs().max() for g in grads)
            local_max_t = local_max.to(torch.float).reshape(())
            for grp in grouped_groups[sig]:
                torch.distributed.all_reduce(local_max_t, op=torch.distributed.ReduceOp.MAX, group=grp)
            global_max = torch.max(global_max, local_max_t)
        return global_max.item()

    # General p-norm: for each group, sum local ||grad||^p over its members,
    # all-reduce SUM across every sharded dimension, then accumulate across
    # groups.
    total_sum = torch.zeros((), dtype=torch.float, device=dev)
    for sig, grads in grouped_grads.items():
        local_sum = torch.zeros((), dtype=torch.float, device=dev)
        if norm_type == 2.0:
            for g in grads:
                local_sum += g.norm() ** 2
        else:
            for g in grads:
                local_sum += g.norm(norm_type) ** norm_type
        for grp in grouped_groups[sig]:
            torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM, group=grp)
        total_sum += local_sum

    return total_sum.item() ** (1.0 / norm_type)


def clip_grad_by_norm(
    parameters: Union[list, torch.Tensor],
    max_norm: float,
    total_norm: float,
) -> None:
    """Clip gradients in-place so that their total norm <= ``max_norm``.

    This is a **no-op** when ``total_norm <= max_norm``.
    DTensor gradients are clipped on their local shards directly
    (the clip coefficient is the same on every rank after all-reduce).

    Args:
        parameters: Model parameters whose ``.grad`` will be scaled.
        max_norm: Maximum allowed total norm.
        total_norm: The current total gradient norm (as returned by
            :func:`compute_grad_norm`).
    """
    if total_norm == 0.0 or max_norm <= 0.0:
        return

    clip_coeff = max_norm / (total_norm + 1e-6)
    if clip_coeff >= 1.0:
        return

    dev = torch.accelerator.current_device()

    for p in parameters:
        if p.grad is not None:
            local_g = _to_local_if_dtensor(p.grad).detach()
            if local_g.device != dev:
                # Fallback when no cache: H2D -> mul -> D2H write back to original storage
                device_g = local_g.to(dev, non_blocking=True)
                device_g.mul_(clip_coeff)
                local_g.copy_(device_g, non_blocking=True)
            else:
                local_g.mul_(clip_coeff)


def clip_grad_norm(
    model: torch.nn.Module,
    max_norm: float,
    norm_type: Union[int, float] = 2.0,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Compute gradient norm and clip in one call.

    For FSDP-wrapped models (that expose ``model.clip_grad_norm_``),
    this delegates to the FSDP-native implementation which is both
    correct (handles sharded parameters) and efficient (avoids
    redundant all-gathers).

    For regular models (including those with DTensor gradients),
    this falls back to :func:`compute_grad_norm` +
    :func:`clip_grad_by_norm`, using *group* for cross-rank
    norm synchronization.

    Args:
        model: The model whose parameters will be clipped.
        max_norm: Maximum allowed total gradient norm.
        norm_type: Type of the p-norm. Defaults to 2.0.
        group: Process group for all-reducing the gradient norm
            across sharding ranks.  If ``None`` and DTensor
            gradients are present, the group is inferred automatically.

    Returns:
        The total gradient norm **before** clipping (as a float),
        useful for logging / monitoring.
    """
    # FSDP models provide their own clip_grad_norm_ which correctly
    # handles sharded parameters and avoids unnecessary all-gathers.
    if hasattr(model, "clip_grad_norm_"):
        total_norm = model.clip_grad_norm_(max_norm, norm_type)
        return total_norm.item() if isinstance(total_norm, torch.Tensor) else float(total_norm)

    params = list(model.parameters())
    total_norm = compute_grad_norm(params, norm_type, group=group)
    clip_grad_by_norm(params, max_norm, total_norm)
    return total_norm

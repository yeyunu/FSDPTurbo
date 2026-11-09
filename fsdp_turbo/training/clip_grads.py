# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""Gradient norm computation and clipping utilities.

This module provides decoupled, FSDP-aware gradient norm operations:
- ``compute_grad_norm``: Compute the total gradient norm (L2 or L-inf) without clipping.
- ``clip_grad_by_norm``: Clip gradients in-place given a pre-computed total norm.
- ``clip_grad_norm``: Compute the gradient norm and clip in one call.

DTensor (``torch.distributed.tensor.DTensor``) gradients are handled
transparently: they are converted to local shards via ``.to_local()``
before computation, and the total norm is synchronized across the
sharding group(s) via ``all_reduce``.

Device-resident gradients are batched via ``torch._foreach_norm`` /
``torch._foreach_mul_`` instead of one kernel per parameter; host-resident
gradients (cpu_offload) are moved to the device one at a time so only a
single gradient copy is resident at any moment.
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

    groups = []
    for dim_idx, placement in enumerate(tensor.placements):
        is_shard = isinstance(placement, Shard)
        if not is_shard and _StridedShard is not None:
            is_shard = isinstance(placement, _StridedShard)
        if is_shard:
            groups.append(mesh.get_group(dim_idx))
    return groups


def _batched_sum_of_squares(grads: list) -> torch.Tensor:
    """Sum of squared L2 norms over a same-dtype list of tensors, batched.

    One fused ``torch._foreach_norm`` launch instead of two kernels per
    parameter; per-tensor norms are stored in the gradients' dtype, so
    low-precision gradients round each scalar before squaring (well below
    the gradients' own quantization noise).
    """
    norms = torch._foreach_norm(grads)
    return torch.stack(norms).float().square().sum()


def _reduction_device() -> torch.device:
    """Return the accelerator device for norm reduction and clipping.

    Normalizes the int ordinal from ``current_device_index()`` to a
    ``torch.device`` so ``tensor.device != dev`` classification is valid.
    """
    dev = torch.accelerator.current_device_index()
    if isinstance(dev, torch.device):
        return dev
    return torch.device(torch.accelerator.current_accelerator().type, int(dev))


def compute_grad_norm(
    parameters: Union[list, torch.Tensor],
    norm_type: Union[int, float] = 2.0,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Compute the total p-norm of gradients across parameters.

    The L2 norm is batched via ``torch._foreach_norm`` (no full FP32 copy
    of the gradients); other p-norms upcast to FP32 per parameter.
    Host-resident gradients (cpu_offload) move to the reduction device and
    FP32-upcast one gradient at a time.  DTensor gradients are
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

    dev = _reduction_device()
    # Gradients sharing the same set of sharded mesh dimensions are aggregated
    # locally, then all-reduced once per group.  A *Replicate* dimension must
    # NOT be summed across (its contribution would be multiplied by the group
    # size), so each group reduces only along dimensions its members are
    # actually sharded on.
    grouped_grads = defaultdict(list)  # device-resident local shards
    grouped_host_grads = defaultdict(list)  # host local shards (cpu_offload)
    grouped_groups = {}
    for p in parameters:
        grad = p.grad
        if grad is None:
            continue
        local_g = _to_local_if_dtensor(grad)
        if local_g.requires_grad:
            # create_graph makes grads require grad; detach stays read-only under autograd.
            local_g = local_g.detach()
        shard_groups = _get_shard_groups_if_dtensor(grad, group)
        sig = tuple(id(g) for g in shard_groups)
        if local_g.device != dev:
            grouped_host_grads[sig].append(local_g)
        else:
            grouped_grads[sig].append(local_g)
        grouped_groups[sig] = shard_groups

    if not grouped_groups:
        return 0.0

    norm_type = float(norm_type)

    if norm_type == inf:
        # Per group: local max over members, all-reduce MAX across every
        # sharded dimension, then the global max across groups.
        global_max = torch.zeros((), dtype=torch.float, device=dev)
        for sig, shard_groups in grouped_groups.items():
            # Each .to() copy dies with the comprehension item: host grads still move one at a time.
            max_per_grad = [g.abs().max() for g in grouped_grads.get(sig, [])]
            max_per_grad += [g.to(dev, non_blocking=True).abs().max() for g in grouped_host_grads.get(sig, [])]
            local_max = max(max_per_grad).float().reshape(())
            for grp in shard_groups:
                torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX, group=grp)
            global_max = torch.max(global_max, local_max)
        return global_max.item()

    # General p-norm: for each group, sum local ||grad||^p over its members,
    # all-reduce SUM across every sharded dimension, then accumulate across
    # groups.
    total_sum = torch.zeros((), dtype=torch.float, device=dev)
    for sig, shard_groups in grouped_groups.items():
        local_grads = grouped_grads.get(sig, [])
        host_grads = grouped_host_grads.get(sig, [])
        local_sum = torch.zeros((), dtype=torch.float, device=dev)
        if norm_type == 2.0:
            # Batch per dtype: mixed dtypes knock _foreach_norm off its fused fast path.
            by_dtype = defaultdict(list)
            for g in local_grads:
                by_dtype[g.dtype].append(g)
            for bucket in by_dtype.values():
                local_sum += _batched_sum_of_squares(bucket)
            # Host grads (cpu_offload): FP32-upcast one at a time.
            for g in host_grads:
                local_sum += g.to(dev, non_blocking=True).float().norm() ** 2
        else:
            for g in local_grads:
                local_sum += g.float().norm(norm_type) ** norm_type
            for g in host_grads:
                local_sum += g.to(dev, non_blocking=True).float().norm(norm_type) ** norm_type
        for grp in shard_groups:
            torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM, group=grp)
        total_sum += local_sum

    return total_sum.item() ** (1.0 / norm_type)


def clip_grad_by_norm(
    parameters: Union[list, torch.Tensor],
    max_norm: float,
    total_norm: float,
    keep_on_device: bool = False,
) -> None:
    """Clip gradients in-place so that their total norm <= ``max_norm``.

    This is a **no-op** when ``total_norm <= max_norm``.
    DTensor gradients are clipped on their local shards directly
    (the clip coefficient is the same on every rank after all-reduce).

    When *keep_on_device* is True and a gradient lives on the host (cpu_offload),
    the existing ``p.grad`` DTensor is re-backed with its clipped device copy
    in place (mutating ``_local_tensor``, since the autograd grad setter forbids
    assigning a device grad to a host param). Optimizers that already skip the
    H2D upload for device-resident local grads (AdamWSwap) then pick it up
    directly, so the clip D2H write-back and the step H2D copy are both
    avoided, and the host gradient storage is released with the replacement.
    With any other optimizer the gradient would sit on the device while the
    parameter stays on the host, so this must only be enabled for AdamWSwap.

    Args:
        parameters: Model parameters whose ``.grad`` will be scaled.
        max_norm: Maximum allowed total norm.
        total_norm: The current total gradient norm (as returned by
            :func:`compute_grad_norm`).
        keep_on_device: Re-point host gradients at their clipped device copies
            so AdamWSwap consumes them without a host round-trip.
    """
    if total_norm == 0.0 or max_norm <= 0.0:
        return

    clip_coeff = max_norm / (total_norm + 1e-6)
    if clip_coeff >= 1.0:
        return

    dev = _reduction_device()
    device_grads = []  # device-resident local shards, scaled in place
    host_grads = []  # (local_host_g, grad) for offloaded grads

    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad
        local_g = _to_local_if_dtensor(grad)
        if local_g.requires_grad:
            # create_graph makes grads require grad; detach keeps the clip autograd-safe.
            local_g = local_g.detach()
        if local_g.device != dev:
            host_grads.append((local_g, grad))
        else:
            device_grads.append(local_g)

    if device_grads:
        torch._foreach_mul_(device_grads, clip_coeff)

    for local_g, grad in host_grads:
        # cpu_offload: only one gradient copy on the device at a time.
        device_g = local_g.to(dev, non_blocking=True)
        device_g.mul_(clip_coeff)
        if keep_on_device:
            # p.grad cannot be assigned a device grad (host param), so re-back
            # _local_tensor in place; AdamWSwap then skips its H2D upload.
            if _HAS_DTENSOR and isinstance(grad, DTensor):
                grad._local_tensor = device_g
            else:
                # Plain-tensor grads cannot be re-backed; keep the host write-back.
                local_g.copy_(device_g, non_blocking=True)
        else:
            # H2D -> clip on device -> D2H write back to original storage
            local_g.copy_(device_g, non_blocking=True)


def clip_grad_norm(
    model: torch.nn.Module,
    max_norm: float,
    norm_type: Union[int, float] = 2.0,
    group: Optional[torch.distributed.ProcessGroup] = None,
    keep_on_device: bool = False,
) -> float:
    """Compute gradient norm and clip in one call.

    Modules may provide a ``custom_clip_grad_norm`` capability for gradient
    storage that is not represented by ``parameter.grad``. All providers for a
    model must use the same handler.

    For other FSDP-wrapped models (that expose ``model.clip_grad_norm_``), this
    delegates to the FSDP-native implementation, which handles sharded
    parameters without redundant all-gathers.

    For regular models (including those with DTensor gradients),
    this falls back to :func:`compute_grad_norm` +
    :func:`clip_grad_by_norm`, using *group* for cross-rank
    norm synchronization.

    Args:
        model: The model whose parameters will be clipped.
        max_norm: Maximum allowed total gradient norm.
        norm_type: Type of the p-norm. Defaults to 2.0.
        group: Process group for all-reducing the gradient norm
            across sharding ranks.  If ``None`` and DTensor gradients
            are present, the group is inferred automatically.
        keep_on_device: Re-point host gradients at their clipped device copies
            so AdamWSwap consumes them without a host round-trip. Only honored
            on the fallback compute/clip path (custom handlers and FSDP-native
            ``clip_grad_norm_`` ignore it).

    Returns:
        The total gradient norm **before** clipping (as a float),
        useful for logging / monitoring.
    """
    custom_handlers = {}
    for module in model.modules():
        handler = getattr(type(module), "custom_clip_grad_norm", None)
        if callable(handler):
            custom_handlers.setdefault(handler, []).append(module)
    if len(custom_handlers) > 1:
        raise RuntimeError("A model cannot use multiple custom gradient clipping handlers.")
    if custom_handlers:
        handler, modules = next(iter(custom_handlers.items()))
        return handler(model, modules, max_norm, norm_type, group)

    # FSDP models provide their own clip_grad_norm_ which correctly handles
    # sharded parameters and avoids unnecessary all-gathers.
    if hasattr(model, "clip_grad_norm_"):
        total_norm = model.clip_grad_norm_(max_norm, norm_type)
        return total_norm.item() if isinstance(total_norm, torch.Tensor) else float(total_norm)

    params = list(model.parameters())
    total_norm = compute_grad_norm(params, norm_type, group=group)
    clip_grad_by_norm(params, max_norm, total_norm, keep_on_device=keep_on_device)
    return total_norm

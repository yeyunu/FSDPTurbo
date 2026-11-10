# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Model-independent Hybrid CP process groups, coordinates and adapter binding."""

from dataclasses import dataclass

import torch.distributed as dist
from torch import nn

from .sequence_partition import SequencePartitionPlan


@dataclass(frozen=True)
class HybridCPRuntime:
    """Read-only process-group references, bound to adapters before FSDP.

    Model/backend selection belongs to the application and its model adapters.
    A missing state group represents a singleton state axis, never WORLD.
    """

    cp_group: dist.ProcessGroup
    ulysses_group: dist.ProcessGroup
    state_group: dist.ProcessGroup | None

    def layout(self, local_length: int, *, zigzag: bool = False) -> SequencePartitionPlan:
        """Build physical coordinates from an already padded local sequence.

        The caller supplies its resolved layout; this method neither selects an
        attention backend nor shards/pads tensors. ``local_length`` is per rank.
        """
        size = dist.get_world_size(self.cp_group)
        return SequencePartitionPlan(local_length * size, size, dist.get_rank(self.cp_group), zigzag)


def bind_hybrid_cp(
    model: nn.Module,
    runtime: HybridCPRuntime,
    *,
    required_roles: tuple[str, ...] = (),
) -> None:
    """Bind opt-in subclass adapters without replacing model forward methods.

    Each adapter provides ``configure_hybrid_cp(runtime)`` and returns its role.
    The caller supplies any algorithm-required roles. Missing adapters/roles
    raise ValueError before attention communication. Hook exceptions propagate.
    """
    roles = []
    for module in model.modules():
        configure = getattr(module, "configure_hybrid_cp", None)
        if callable(configure):
            roles.append(configure(runtime))
    if not roles:
        raise ValueError("Hybrid CP requires subclass adapters implementing configure_hybrid_cp().")
    missing = sorted(set(required_roles).difference(roles))
    if missing:
        raise ValueError("Hybrid CP requires adapter roles: " + ", ".join(missing) + ".")

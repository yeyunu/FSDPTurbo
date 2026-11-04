# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_gather_tensor_autograd


def kv_allgather(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    gather_dim: int = 2,
) -> torch.Tensor:
    """All-gather a tensor along ``gather_dim`` across ``process_group``.

    Each rank contributes its local shard and receives the concatenation of
    all shards.  Gradients are propagated across workers via
    ``all_gather_tensor_autograd``.

    Args:
        input_: Local tensor shard.
        process_group: Process group for the collective operation.
        gather_dim: Dimension along which to concatenate gathered shards.

    Returns:
        Tensor with all shards concatenated along ``gather_dim``.
    """
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    return all_gather_tensor_autograd(input_.contiguous(), gather_dim=gather_dim, group=process_group)


def get_kv_allgather_context():
    """Return ``(kv_allgather_group, kv_rank, kv_size)``; group is ``None`` when disabled."""
    from fsdp_turbo.distributed.parallel_state import get_parallel_state

    parallel_state = get_parallel_state()
    kv_size = parallel_state.get_kvallgather_group_size()
    if kv_size <= 1:
        return None, 0, 1
    return parallel_state.get_kvallgather_group(), parallel_state.get_kvallgather_rank(), kv_size


def get_flattened_cp_context():
    """Return ``(cp_group, cp_rank, cp_size)`` for the flattened kvallgather x ulysses mesh.

    The flattened ``cp`` group ranks are ordered kvallgather-major, so a
    contiguous split over it yields shards whose concatenation is in natural
    sequence order.
    """
    from fsdp_turbo.distributed.parallel_state import get_parallel_state

    parallel_state = get_parallel_state()
    cp_size = parallel_state.get_cp_group_size()
    if cp_size <= 1:
        return None, 0, 1
    return parallel_state.get_cp_group(), parallel_state.get_cp_rank(), cp_size

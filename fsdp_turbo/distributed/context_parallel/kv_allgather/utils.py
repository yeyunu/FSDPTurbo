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

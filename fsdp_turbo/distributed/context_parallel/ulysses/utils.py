# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Autograd-capable Ulysses all-to-all communication."""

from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single


def ulysses_all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    gather_size: Optional[int] = None,
) -> torch.Tensor:
    """Scatter one dimension and gather another across a process group."""
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    scatter_size = input_.size(scatter_dim)
    if scatter_size % world_size != 0:
        raise ValueError(f"scatter_dim size ({scatter_size}) must be divisible by world_size ({world_size}).")

    if gather_size is None:
        gather_size = input_.size(gather_dim) * world_size
    elif gather_size % world_size != 0:
        raise ValueError(f"gather_size ({gather_size}) must be divisible by world_size ({world_size}).")

    scatter_size_per_rank = scatter_size // world_size
    gather_size_per_rank = gather_size // world_size
    input_split_sizes = [scatter_size_per_rank * input_.size(gather_dim)] * world_size
    output_split_sizes = [scatter_size_per_rank * gather_size_per_rank] * world_size

    min_dim = min(scatter_dim, gather_dim)
    max_dim = max(scatter_dim, gather_dim)
    batch_dims = input_.shape[:min_dim]
    hidden_dims = input_.shape[max_dim + 1 :]

    scatter_before_gather = scatter_dim < gather_dim
    input_swapped = (
        input_.contiguous() if scatter_before_gather else input_.transpose(scatter_dim, gather_dim).contiguous()
    )
    input_merged = input_swapped.reshape(
        *batch_dims,
        input_.size(scatter_dim) * input_.size(gather_dim),
        *hidden_dims,
    )

    input_for_a2a = input_merged.transpose(min_dim, 0).contiguous()
    output_from_a2a = input_for_a2a.new_empty((sum(output_split_sizes), *input_for_a2a.shape[1:]))
    output_from_a2a = all_to_all_single(
        output_from_a2a,
        input_for_a2a,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=process_group,
    )
    output_merged = output_from_a2a.transpose(0, min_dim).contiguous()

    output_chunks = output_merged.reshape(
        *batch_dims,
        world_size,
        scatter_size_per_rank,
        gather_size_per_rank,
        *hidden_dims,
    )
    output_swapped = output_chunks.transpose(min_dim, min_dim + 1).reshape(
        *batch_dims,
        scatter_size_per_rank,
        gather_size,
        *hidden_dims,
    )
    if scatter_before_gather:
        return output_swapped.contiguous()
    return output_swapped.transpose(scatter_dim, gather_dim).contiguous()

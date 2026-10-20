# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Variable-size all-to-all (alltoallv) communication primitives for vocabulary parallelism."""

from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single_autograd


def alltoallv(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup],
    input_splits: List[int],
    output_splits: List[int],
) -> torch.Tensor:
    """Run an alltoallv (per-rank variable send/recv sizes) along the first tensor dimension."""
    if group is None or dist.get_world_size(group) == 1:
        return input_
    return all_to_all_single_autograd(input_, output_splits, input_splits, group)


def alltoallv_with_gradient_divide_factor(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup],
    input_splits: List[int],
    output_splits: List[int],
    gradient_divide_factor: float,
) -> torch.Tensor:
    """Alltoallv that divides the incoming gradient by ``gradient_divide_factor``.

    The divide factor gives pure-VP (singleton V-FSDP) parameters the same
    gradient-averaging semantics FSDP reduce-scatter applies to dense
    parameters: the return exchange sums token contributions from every rank,
    so the gradient is scaled before flowing back through it. The factor is
    read (closed over) at forward time, like the autograd context before it.
    """
    output = alltoallv(input_, group, input_splits, output_splits)
    if gradient_divide_factor != 1.0 and output.requires_grad:
        output.register_hook(lambda grad: grad / gradient_divide_factor)
    return output

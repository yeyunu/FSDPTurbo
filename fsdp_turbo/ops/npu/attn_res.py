# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Select the available attention-residual implementation for NPU devices."""

import logging

import torch

from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.utils.log import print_rank


logger = logging.getLogger(__name__)

try:
    from cann_ops_transformer import block_attention_residuals as _attn_res_impl
except ImportError:
    # Keep the fallback import conditional, as in the chunk_kda backend selector.
    from fsdp_turbo.ops.cpu.attn_res import attn_res_cpu as _attn_res_impl  # pylint: disable=ungrouped-imports

    ATTN_RES_BACKEND = 'eager'
    print_rank(
        logger.warning,
        "AscendC attention residuals are unavailable; falling back to the eager implementation.",
    )
else:
    ATTN_RES_BACKEND = 'ascendc'
    # This module executes at import time, before BaseTrainer._init_distributed
    # configures logging, so a logger record would be dropped; print straight to
    # stdout via print_rank (rank-0 only) so the marker shows in every launch mode.
    print_rank(print, "[attn_res] backend=ascendc -> fused cann_ops_transformer.block_attention_residuals")


@register_op('attn_res', 'npu')
def attn_res_npu(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Run the fused NPU implementation or fall back to eager when permitted."""
    if ATTN_RES_BACKEND == 'ascendc':
        return _attn_res_impl(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps=variance_epsilon)
    return _attn_res_impl(prefix_sum, block_residual, proj_weight, norm_weight, variance_epsilon)


__all__ = ['attn_res_npu']

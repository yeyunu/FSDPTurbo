# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Select the available attention-residual implementation for NPU devices."""

import logging

from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.utils.log import print_rank


logger = logging.getLogger(__name__)

try:
    from cann_ops_transformer import block_attention_residuals as _attn_res_impl
except ImportError:
    # Keep the fallback import conditional, as in the chunk_kda backend selector.
    from fsdp_turbo.ops.cpu.attn_res import attn_res_cpu as _attn_res_impl  # pylint: disable=ungrouped-imports

    _ATTN_RES_BACKEND = 'eager'
    print_rank(
        logger.warning,
        "AscendC attention residuals are unavailable; falling back to the eager implementation.",
    )
else:
    _ATTN_RES_BACKEND = 'ascendc'


@register_op('attn_res', 'npu')
def attn_res_npu(prefix_sum, block_residual, proj_weight, norm_weight, variance_epsilon):
    """Run attention residuals with the selected NPU implementation."""
    if _ATTN_RES_BACKEND == 'ascendc':
        return _attn_res_impl(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps=variance_epsilon)
    return _attn_res_impl(prefix_sum, block_residual, proj_weight, norm_weight, variance_epsilon)


__all__ = ['attn_res_npu']

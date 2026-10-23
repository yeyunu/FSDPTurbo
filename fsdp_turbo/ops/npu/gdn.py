# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Select the best available GDN implementation for NPU devices."""

import logging

from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.utils.device import IS_FLA_NPU_AVAILABLE, IS_MINDSPEED_OPS_AVAILABLE
from fsdp_turbo.utils.log import log_warning_once
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

logger = logging.getLogger(__name__)


@register_op('chunk_gated_delta_rule', 'npu')
def chunk_gdn_npu(*args, **kwargs):
    """Run GDN with the selected NPU implementation."""
    if IS_FLA_NPU_AVAILABLE and IS_MINDSPEED_OPS_AVAILABLE:
        from fsdp_turbo.ops.npu.chunk_gdn_ascendc import chunk_gdn_ascendc

        return chunk_gdn_ascendc(*args, **kwargs)
    else:
        log_warning_once(
            logger, "AscendC GDN implementation is unavailable; falling back to the triton implementation."
        )
        return chunk_gated_delta_rule(*args, **kwargs)


@register_op('fused_recurrent_gated_delta_rule', 'npu')
def fused_recurrent_gdn_npu(*args, **kwargs):
    """Call FLA recurrent GDN while leaving backend selection to FLA."""
    return fused_recurrent_gated_delta_rule(*args, **kwargs)


__all__ = [
    "chunk_gdn_npu",
    "fused_recurrent_gdn_npu",
]

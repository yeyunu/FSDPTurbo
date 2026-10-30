# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""NPU GDN core computation, independent of checkpoint replay and Training."""

import logging

import torch

from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.utils.device import IS_FLA_NPU_AVAILABLE, IS_MINDSPEED_OPS_AVAILABLE
from fsdp_turbo.utils.log import log_warning_once


logger = logging.getLogger(__name__)


@register_op('chunk_gated_delta_rule', 'npu')
def chunk_gdn_npu(
    *args: object,
    implementation: str = 'auto',
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use AscendC with fla_npu and MindSpeed Ops, otherwise delegate to FLA."""
    if IS_FLA_NPU_AVAILABLE and IS_MINDSPEED_OPS_AVAILABLE:
        from fsdp_turbo.ops.npu.chunk_gdn_ascendc import chunk_gdn_ascendc

        return chunk_gdn_ascendc(*args, **kwargs)
    if implementation == 'ascendc':
        raise RuntimeError('AscendC GDN requires fla_npu and mindspeed_ops.')
    log_warning_once(logger, 'AscendC GDN is unavailable; attempting FLA fallback.')
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule(*args, **kwargs)


@register_op('fused_recurrent_gated_delta_rule', 'npu')
def fused_recurrent_gdn_npu(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reuse FLA recurrent GDN for cache continuation on NPU tensors."""
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    return fused_recurrent_gated_delta_rule(*args, **kwargs)


__all__ = [
    "chunk_gdn_npu",
    "fused_recurrent_gdn_npu",
]

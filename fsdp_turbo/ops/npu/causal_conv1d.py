# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Select the best available causal-convolution implementation for NPU."""

# Backend imports are intentionally conditional to preserve selection priority.
# pylint: disable=ungrouped-imports

import logging
from typing import Optional

import torch

from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.utils.device import IS_FLA_NPU_AVAILABLE, IS_TRITON_AVAILABLE
from fsdp_turbo.utils.log import print_rank


logger = logging.getLogger(__name__)


if IS_FLA_NPU_AVAILABLE:
    from fsdp_turbo.ops.npu.causal_conv1d_ascendc import causal_conv1d_ascendc as _causal_conv1d_impl

    _CAUSAL_CONV1D_BACKEND = 'ascendc'
elif IS_TRITON_AVAILABLE:
    try:
        from mindspeed_ops.api.triton.convolution import causal_conv1d as _causal_conv1d_impl

        _CAUSAL_CONV1D_BACKEND = 'mindspeed'
    except ImportError:
        from fsdp_turbo.ops.triton.causal_conv1d import causal_conv1d_triton as _causal_conv1d_impl

        _CAUSAL_CONV1D_BACKEND = 'triton'
else:
    from fsdp_turbo.ops.cpu.causal_conv1d import causal_conv1d_cpu as _causal_conv1d_impl

    _CAUSAL_CONV1D_BACKEND = 'eager'

    print_rank(
        logger.warning,
        "AscendC and Triton Ascend causal-convolution implementations are unavailable; "
        "falling back to the eager implementation.",
    )


@register_op('causal_conv1d', 'npu')
def causal_conv1d_npu(
    *args: object, head_num: int = 1, implementation: str = 'auto', **kwargs: object
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run causal convolution with the selected NPU implementation."""
    if implementation == 'ascendc' and _CAUSAL_CONV1D_BACKEND != 'ascendc':
        raise RuntimeError('AscendC causal convolution requires fla_npu.')
    if _CAUSAL_CONV1D_BACKEND == 'ascendc':
        return _causal_conv1d_impl(*args, head_num=head_num, **kwargs)
    return _causal_conv1d_impl(*args, **kwargs)


@register_op('causal_conv1d_update', 'npu')
def causal_conv1d_update_npu(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    activation: str = None,
):
    from fsdp_turbo.ops.triton.causal_conv1d_kernel import causal_conv1d_update_bdt_impl

    output = causal_conv1d_update_bdt_impl(
        x=x.transpose(1, 2).contiguous(),
        conv_state=cache,
        weight=weight,
        bias=bias,
        activation=activation,
    ).transpose(1, 2)
    if residual is not None:
        output = output + residual
    return output, cache


__all__ = ['causal_conv1d_npu', 'causal_conv1d_update_npu']

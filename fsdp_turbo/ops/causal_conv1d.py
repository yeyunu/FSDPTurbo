# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Copyright 2025 XPU-Forces Team
# Copyright (c) 2023-2025 Songlin Yang Yu Zhang
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Optional

import torch

from fsdp_turbo.ops.registry import dispatch_op


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    head_num: int = 1,
    implementation: str = 'auto',
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply a causal depthwise convolution with the selected backend."""
    if implementation not in ('auto', 'eager', 'ascendc'):
        raise ValueError(f'Unsupported causal convolution implementation: {implementation}')
    device_type = 'cpu' if implementation == 'eager' else x.device.type
    if implementation == 'ascendc' and device_type != 'npu':
        raise ValueError('AscendC causal convolution requires NPU tensors.')
    backend_options = {'implementation': implementation} if device_type == 'npu' else {}
    return dispatch_op(
        'causal_conv1d',
        x,
        weight,
        bias,
        residual,
        initial_state,
        activation,
        cu_seqlens,
        output_final_state,
        head_num=head_num,
        device_type=device_type,
        **backend_options,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    implementation: str = 'auto',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the convolution cache, using the eager path when requested."""
    if implementation not in ('auto', 'eager', 'ascendc'):
        raise ValueError(f'Unsupported causal convolution implementation: {implementation}')
    device_type = 'cpu' if implementation == 'eager' else x.device.type
    if implementation == 'ascendc' and device_type != 'npu':
        raise ValueError('AscendC causal convolution requires NPU tensors.')
    return dispatch_op(
        'causal_conv1d_update',
        x,
        cache,
        weight,
        bias,
        residual,
        activation,
        device_type=device_type,
    )


__all__ = ['causal_conv1d', 'causal_conv1d_update']

# Copyright (c) 2026, Huawei Technologies Co., Ltd.
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
):
    """Apply a causal depthwise convolution using the active device backend."""
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
    )


def causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
):
    """Apply an incremental causal convolution and update ``cache`` in place."""
    return dispatch_op(
        'causal_conv1d_update',
        x,
        cache,
        weight,
        bias,
        residual,
        activation,
    )


__all__ = ['causal_conv1d', 'causal_conv1d_update']

# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Optional

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.registry import register_op


def _validate_activation(activation: Optional[str]) -> None:
    if activation not in (None, 'silu', 'swish'):
        raise ValueError("activation must be one of None, 'silu', or 'swish'")


def _convolve_sequence(
    sequence: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    activation: Optional[str],
) -> torch.Tensor:
    kernel_size, hidden_size = weight.shape
    if initial_state is None:
        history = sequence.new_zeros(hidden_size, kernel_size - 1)
    else:
        history = initial_state[:, 1:]

    convolution_input = torch.cat((history, sequence.transpose(0, 1)), dim=-1).unsqueeze(0)
    output = (
        F.conv1d(
            convolution_input,
            weight.transpose(0, 1).unsqueeze(1),
            bias=bias,
            groups=hidden_size,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    if activation in ('silu', 'swish'):
        output = F.silu(output)
    return output


def _final_state(
    sequence: torch.Tensor,
    kernel_size: int,
    initial_state: Optional[torch.Tensor],
) -> torch.Tensor:
    hidden_size = sequence.shape[-1]
    if initial_state is None:
        initial_state = sequence.new_zeros(hidden_size, kernel_size)
    state_input = torch.cat((initial_state, sequence.transpose(0, 1)), dim=-1)
    return state_input[:, -kernel_size:]


@register_op('causal_conv1d', 'cpu')
def causal_conv1d_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
):
    """Pure PyTorch implementation used as the portable fallback."""
    _validate_activation(activation)
    if x.ndim != 3 or weight.ndim != 2 or x.shape[-1] != weight.shape[-1]:
        raise ValueError("x must be [B, T, D] and weight must be [W, D]")

    kernel_size = weight.shape[0]
    if cu_seqlens is None:
        sequences = [x[batch_index] for batch_index in range(x.shape[0])]
        state_count = x.shape[0]
    else:
        if x.shape[0] != 1:
            raise ValueError("x batch size must be 1 when cu_seqlens is provided")
        sequence_offsets = cu_seqlens.tolist()
        sequences = [x[0, start:end] for start, end in zip(sequence_offsets[:-1], sequence_offsets[1:])]
        state_count = len(sequences)

    if initial_state is not None and initial_state.shape != (state_count, x.shape[-1], kernel_size):
        raise ValueError("initial_state must be [N, D, W]")

    outputs = []
    final_states = []
    for sequence_index, sequence in enumerate(sequences):
        sequence_state = None if initial_state is None else initial_state[sequence_index]
        outputs.append(_convolve_sequence(sequence, weight, bias, sequence_state, activation))
        if output_final_state:
            final_states.append(_final_state(sequence, kernel_size, sequence_state))

    output = torch.stack(outputs) if cu_seqlens is None else torch.cat(outputs).unsqueeze(0)
    if residual is not None:
        output = output + residual
    final_state = torch.stack(final_states) if output_final_state else None
    return output, final_state


@register_op('causal_conv1d_update', 'cpu')
def causal_conv1d_update_cpu(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
):
    output, final_state = causal_conv1d_cpu(
        x=x,
        weight=weight.transpose(0, 1),
        bias=bias,
        residual=residual,
        initial_state=cache,
        activation=activation,
        output_final_state=True,
    )
    cache.copy_(final_state)
    return output, cache


__all__ = ['causal_conv1d_cpu', 'causal_conv1d_update_cpu']

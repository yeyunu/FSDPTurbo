# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Differentiable recurrent oracle for the staged KDA P2P interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fsdp_turbo.ops.cpu.chunk_kda import kda_gate, l2norm
from fsdp_turbo.ops.registry import register_op

if TYPE_CHECKING:
    from fsdp_turbo.ops.kda_p2p import KdaStateContext


@register_op("chunk_kda_p2p", "cpu")
def chunk_kda_p2p_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_context: KdaStateContext,
    lower_bound: float | None,
) -> torch.Tensor:
    """Compute affine interval summaries, exchange prefixes, then local outputs."""
    cu_seqlens = state_context.cu_seqlens
    output_dtype = v.dtype
    q, k, v = l2norm(q).float(), l2norm(k).float(), v.float()
    gate = kda_gate(g, A_log, dt_bias, lower_bound).exp()
    heads, dim = k.shape[2:]
    identity = torch.eye(dim, device=q.device).expand(heads, dim, dim)
    summaries = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        transition, state = identity, torch.zeros_like(identity)
        for token in range(start, end):
            key = k[0, token]
            transition = transition * gate[0, token, :, :, None]
            transition = transition - beta[0, token, :, None, None] * key[..., None] * (
                key[..., None] * transition
            ).sum(-2, keepdim=True)
            state = state * gate[0, token, :, :, None]
            delta = beta[0, token, :, None] * (v[0, token] - (key[..., None] * state).sum(-2))
            state = state + key[..., None] * delta[..., None, :]
        summaries.append(torch.cat((state, transition), -1))
    initial_states = state_context.compute_initial_states(torch.stack(summaries))
    outputs = []
    for index, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        state = initial_states[index]
        for token in range(start, end):
            key = k[0, token]
            state = state * gate[0, token, :, :, None]
            delta = beta[0, token, :, None] * (v[0, token] - (key[..., None] * state).sum(-2))
            state = state + key[..., None] * delta[..., None, :]
            outputs.append((q[0, token, :, :, None] * state).sum(-2) * dim**-0.5)
    return torch.stack(outputs).unsqueeze(0).to(output_dtype)

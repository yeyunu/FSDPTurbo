# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Pure-torch CPU implementation of the Engram gated-memory operator.

Semantics match the NPU PyPTO-Pro fused kernels (``engram_forward`` /
``engram_backward``): raw key/value projections are cached in the operand
dtype, scores/gates stay FP32, and all RMSNorm/gate/dot math runs in FP32.

The packed-wkv block glue (``EngramWkvFunction``) is device-neutral and lives
in the semantic layer :mod:`fsdp_turbo.ops.engram`; this backend only
registers it for CPU alongside the eager kernels.
"""

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.engram import EngramWkvFunction
from fsdp_turbo.ops.registry import register_op


def _rms_norm_forward(x: torch.Tensor, gamma: torch.Tensor, eps: float):
    x_float = x.float()
    rms = torch.sqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    normed_without_gamma = x_float / rms
    return normed_without_gamma * gamma.float(), rms, normed_without_gamma


def _rms_norm_backward(grad_output, gamma, rms, normed_without_gamma):
    grad_normed = grad_output * gamma.float()
    inner = (grad_normed * normed_without_gamma).sum(dim=-1, keepdim=True) / gamma.shape[-1]
    grad_x = (grad_normed - normed_without_gamma * inner) / rms
    grad_gamma = (grad_output * normed_without_gamma).flatten(0, -2).sum(dim=0)
    return grad_x, grad_gamma


@register_op("engram_forward", "cpu")
def engram_forward_cpu(
    hidden_states,
    embeddings,
    key_proj_weights,
    value_proj_weights,
    key_gamma,
    query_gamma,
    clamp_value: float = 1e-6,
    eps: float = 1e-6,
):
    head_count = key_proj_weights.shape[0]
    scale = hidden_states.shape[-1] ** -0.5

    head_dim = key_proj_weights.shape[-1]
    packed_key_weights = key_proj_weights.transpose(1, 2).reshape(head_count * head_dim, -1)
    keys = F.linear(embeddings, packed_key_weights).unflatten(-1, (head_count, head_dim))
    value = embeddings @ value_proj_weights

    normed_keys, _, _ = zip(*(_rms_norm_forward(keys[:, :, head], key_gamma[head], eps) for head in range(head_count)))
    normed_queries = [
        _rms_norm_forward(hidden_states[:, :, head], query_gamma[head], eps)[0] for head in range(head_count)
    ]
    scores = torch.stack(
        [(normed_keys[head] * normed_queries[head]).sum(dim=-1) * scale for head in range(head_count)],
        dim=2,
    )
    logits = torch.sign(scores) * scores.abs().clamp_min(clamp_value).sqrt()
    gates = torch.sigmoid(logits)
    value_out = gates.unsqueeze(-1) * value.float().unsqueeze(2)

    return (
        value_out.to(hidden_states.dtype),
        scores,
        keys.to(key_proj_weights.dtype),
        value.to(value_proj_weights.dtype),
        gates,
    )


@register_op("engram_backward", "cpu")
def engram_backward_cpu(
    grad_output,
    hidden_states,
    embeddings,
    key_proj_weights,
    value_proj_weights,
    key_gamma,
    query_gamma,
    scores,
    gates,
    keys,
    value,
    clamp_value: float = 1e-6,
    eps: float = 1e-6,
    pretransposed_weights: bool = False,
):
    _, _, head_count, dim = hidden_states.shape
    scale = dim**-0.5
    output_dtype = hidden_states.dtype
    if pretransposed_weights:
        # Caller passed the backward-kernel layouts [M, H, De] / [H, De].
        key_proj_weights = key_proj_weights.transpose(1, 2).contiguous()
        value_proj_weights = value_proj_weights.t().contiguous()

    grad_output_float = grad_output.float()
    grad_gates = (grad_output_float * value.float().unsqueeze(2)).sum(dim=-1)
    grad_value = (grad_output_float * gates.unsqueeze(-1)).sum(dim=2)

    grad_embeddings = (grad_value @ value_proj_weights.t().float()).float()
    grad_value_proj_weights = embeddings.float().flatten(0, -2).t() @ grad_value.flatten(0, -2)

    sigmoid_grad = gates * (1.0 - gates)
    active = scores.abs() > clamp_value
    logits_grad = active / (2.0 * scores.abs().clamp_min(clamp_value).sqrt() + 1e-12)
    grad_scores = grad_gates * sigmoid_grad * logits_grad

    grad_hidden_states = []
    grad_key_proj_weights = []
    grad_key_gamma = []
    grad_query_gamma = []
    embeddings_2d = embeddings.reshape(-1, embeddings.shape[-1]).float()
    for head in range(head_count):
        grad_score = grad_scores[:, :, head].unsqueeze(-1) * scale
        normed_key, key_rms, key_n = _rms_norm_forward(keys[:, :, head], key_gamma[head], eps)
        normed_query, query_rms, query_n = _rms_norm_forward(hidden_states[:, :, head], query_gamma[head], eps)
        grad_h, grad_qgamma = _rms_norm_backward(grad_score * normed_key, query_gamma[head], query_rms, query_n)
        grad_k, grad_kgamma = _rms_norm_backward(grad_score * normed_query, key_gamma[head], key_rms, key_n)
        grad_hidden_states.append(grad_h)
        grad_key_gamma.append(grad_kgamma)
        grad_query_gamma.append(grad_qgamma)
        grad_embeddings += grad_k @ key_proj_weights[head].t().float()
        grad_key_proj_weights.append(embeddings_2d.t() @ grad_k.flatten(0, -2))

    return (
        torch.stack(grad_hidden_states, dim=2).to(output_dtype),
        grad_embeddings.to(embeddings.dtype),
        torch.stack(grad_key_proj_weights, dim=0).to(key_proj_weights.dtype),
        grad_value_proj_weights.to(value_proj_weights.dtype),
        torch.stack(grad_key_gamma, dim=0).to(key_gamma.dtype),
        torch.stack(grad_query_gamma, dim=0).to(query_gamma.dtype),
    )


@register_op("engram_wkv", "cpu")
def engram_wkv_cpu(
    hidden_states,
    memory,
    wkv_weight,
    key_gamma,
    query_gamma,
    clamp_value: float = 1e-6,
    eps: float = 1e-6,
    token_mask=None,
):
    return EngramWkvFunction.apply(
        hidden_states,
        memory,
        wkv_weight,
        key_gamma,
        query_gamma,
        eps,
        clamp_value,
        token_mask,
    )

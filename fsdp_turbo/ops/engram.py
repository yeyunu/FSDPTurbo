# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Semantic entry points for the Engram gated-memory operator.

``engram_forward`` computes the per-head key/value projections, dual RMSNorm
scaled-dot score, signed-sqrt sigmoid gate and the gated value (without the
residual); ``engram_backward`` returns its six gradients. ``fused_engram_wkv``
is the full packed-wkv block (weight split, token mask and residual add) used
by the model. Dispatch follows :mod:`fsdp_turbo.ops.registry` (NPU fused
kernel with CPU fallback).

``EngramWkvFunction`` is the device-neutral autograd glue assembling the full
block over the two dispatched kernels. It lives in this backend-free module
because its body is pure torch (packed wkv split, mask and residual) and the
same class serves every backend; the per-backend packages only register it.
"""

import torch

from fsdp_turbo.ops.registry import dispatch_op


def add_engram_residual(residual, branch):
    # Low-precision add already accumulates in FP32 and rounds once on output.
    if residual.dtype == branch.dtype and residual.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return residual + branch
    return (residual.float() + branch.float()).to(residual.dtype)


def engram_forward(
    hidden_states,
    embeddings,
    key_proj_weights,
    value_proj_weights,
    key_gamma,
    query_gamma,
    clamp_value: float = 1e-6,
    eps: float = 1e-6,
):
    return dispatch_op(
        "engram_forward",
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
        clamp_value=clamp_value,
        eps=eps,
    )


def engram_backward(
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
    return dispatch_op(
        "engram_backward",
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
        clamp_value=clamp_value,
        eps=eps,
        pretransposed_weights=pretransposed_weights,
    )


class EngramWkvFunction(torch.autograd.Function):
    """Packed-wkv Engram block assembled over the dispatched fwd/bwd kernels.

    The fused kernels cover the projection/RMSNorm/gating math; the DS4.1
    packed ``wkv`` Linear (key/value split and gradient repack), the
    dead-token mask and the residual add are glue assembled here:

    * forward materializes the ``[M, De, H]`` key and ``[De, H]`` value
      projection weights for the forward kernel;
    * backward consumes the packed parameter's contiguous ``[M, H, De]`` /
      ``[H, De]`` slice views directly (``pretransposed_weights=True``),
      skipping the wrapper transposes;
    * masked rows pass through as residual-only: the gate path is zeroed at
      them while the upstream gradient still reaches ``hidden_states``.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        memory,
        wkv_weight,
        key_gamma,
        query_gamma,
        eps,
        clamp_value,
        token_mask,
    ):
        head_count, dim = key_gamma.shape
        key_weight, value_weight = wkv_weight.split([head_count * dim, dim], dim=0)
        key_proj_weight = key_weight.reshape(head_count, dim, -1).transpose(1, 2).contiguous()
        value_proj_weight = value_weight.t().contiguous()

        value_out, scores, keys, value, gates = engram_forward(
            hidden_states.contiguous(),
            memory.contiguous(),
            key_proj_weight,
            value_proj_weight,
            key_gamma,
            query_gamma,
            clamp_value=clamp_value,
            eps=eps,
        )
        if token_mask is not None:
            value_out = value_out * token_mask[:, :, None, None]

        output = add_engram_residual(hidden_states, value_out)
        # Backward consumes the weights as zero-copy views of the packed
        # parameter, so the fwd-only materialized projections are not saved.
        ctx.save_for_backward(
            hidden_states,
            memory,
            wkv_weight,
            key_gamma,
            query_gamma,
            scores,
            gates,
            keys,
            value,
            token_mask if token_mask is not None else torch.empty(0, dtype=torch.bool),
        )
        ctx.has_token_mask = token_mask is not None
        ctx.eps = eps
        ctx.clamp_value = clamp_value
        ctx.wkv_dtype = wkv_weight.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            hidden_states,
            memory,
            wkv_weight,
            key_gamma,
            query_gamma,
            scores,
            gates,
            keys,
            value,
            token_mask,
        ) = ctx.saved_tensors
        head_count, head_dim = key_gamma.shape
        embedding_dim = wkv_weight.shape[-1]
        key_weight, value_weight = wkv_weight.split([head_count * head_dim, head_dim], dim=0)
        # The backward kernel consumes [M,H,De] / [H,De]: both are contiguous
        # views of the packed wkv parameter, so no transpose copy is needed.
        key_weight_transposed = key_weight.view(head_count, head_dim, embedding_dim)
        if not ctx.has_token_mask:
            token_mask = None

        gate_grad_output = grad_output
        if token_mask is not None:
            gate_grad_output = grad_output * token_mask[:, :, None, None]

        (
            grad_hidden,
            grad_memory,
            grad_key_proj,
            grad_value_proj,
            grad_key_gamma,
            grad_query_gamma,
        ) = engram_backward(
            gate_grad_output.contiguous(),
            hidden_states.contiguous(),
            memory.contiguous(),
            key_weight_transposed,
            value_weight,
            key_gamma,
            query_gamma,
            scores,
            gates,
            keys,
            value,
            clamp_value=ctx.clamp_value,
            eps=ctx.eps,
            pretransposed_weights=True,
        )

        # Residual pass-through: gradient flows to hidden_states at every row,
        # including masked rows whose gate path gradient was zeroed.
        grad_hidden = add_engram_residual(grad_hidden, grad_output)

        grad_wkv_weight = torch.empty(
            (head_count + 1, head_dim, embedding_dim),
            device=wkv_weight.device,
            dtype=ctx.wkv_dtype,
        )
        grad_wkv_weight[:head_count].copy_(grad_key_proj.transpose(1, 2))
        grad_wkv_weight[head_count].copy_(grad_value_proj.t())
        grad_wkv_weight = grad_wkv_weight.view(-1, embedding_dim)

        return (
            grad_hidden,
            grad_memory.to(memory.dtype),
            grad_wkv_weight,
            grad_key_gamma.to(key_gamma.dtype),
            grad_query_gamma.to(query_gamma.dtype),
            None,
            None,
            None,
        )


def fused_engram_wkv(
    hidden_states,
    memory,
    wkv_weight,
    key_gamma,
    query_gamma,
    clamp_value: float = 1e-6,
    eps: float = 1e-6,
    token_mask=None,
):
    """Run the packed wkv Engram block and add its gated output to the residual.

    ``wkv_weight`` packs ``head_count * dim`` key rows followed by ``dim``
    shared value rows, matching the DS4.1 ``wkv`` Linear. Rows where
    ``token_mask`` is False pass through as the untouched residual.
    """
    return dispatch_op(
        "engram_wkv",
        hidden_states,
        memory,
        wkv_weight,
        key_gamma,
        query_gamma,
        clamp_value,
        eps,
        token_mask,
    )


__all__ = [
    "engram_forward",
    "engram_backward",
    "EngramWkvFunction",
    "fused_engram_wkv",
    "add_engram_residual",
]

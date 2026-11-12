# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Device-dispatched GLM53 token selection and sparse Indexer KL."""

import torch

from fsdp_turbo.ops.registry import dispatch_op


def lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return BSK indices/scores for equal-length causal BSND training inputs."""
    return dispatch_op("glm_lightning_indexer", query, key, weights, topk, device_type=query.device.type)


def indexer_kl_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    maximum: torch.Tensor,
    total: torch.Tensor,
    scale: float,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
) -> torch.Tensor:
    """Return token-mean selected-support KL; only Indexer inputs receive gradients.

    Attention teacher inputs are detached. NPU uses the original fused kernel
    with its complete four-output result so checkpoint OpReplay can cache it.
    """
    return dispatch_op(
        "glm_indexer_kl_loss",
        query.detach(),
        key.detach(),
        query_index,
        key_index,
        weights,
        indices,
        maximum,
        total,
        scale,
        query_rope.detach(),
        key_rope.detach(),
        device_type=query.device.type,
    )


class _AttachIndexerLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, loss, backward_scale):
        ctx.save_for_backward(loss)
        ctx.backward_scale = backward_scale
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        (loss,) = ctx.saved_tensors
        return output_gradient, torch.ones_like(loss) * ctx.backward_scale, None


def attach_indexer_loss(output: torch.Tensor, loss: torch.Tensor, backward_scale: float) -> torch.Tensor:
    """Attach the KL gradient without changing attention values or main-loss normalization."""
    return _AttachIndexerLoss.apply(output, loss, backward_scale)

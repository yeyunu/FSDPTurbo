# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""NPU GLM53 kernels, preserving the validated fused KL ABI."""

import torch
import torch_npu

from fsdp_turbo.ops.registry import register_op


@register_op("glm_lightning_indexer", "npu")
def lightning_indexer_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select causal top-k with the native BSND fused Indexer."""
    indices, scores = torch_npu.npu_lightning_indexer(
        query.contiguous().to(torch.bfloat16),
        key.contiguous().to(torch.bfloat16),
        weights.contiguous().to(torch.bfloat16),
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=None,
        layout_query="BSND",
        layout_key="BSND",
        sparse_count=topk,
        sparse_mode=3,
        return_value=True,
    )
    return indices.squeeze(2).to(torch.int32), scores.squeeze(2)


class _IndexerKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, query_index, key_index, weights, indices, maximum, total, scale, query_rope, key_rope):
        grad_query, grad_key, grad_weights, loss = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query,
            key,
            query_index,
            key_index,
            weights,
            indices.unsqueeze(2).contiguous().to(torch.int32),
            maximum,
            total,
            scale_value=scale,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=None,
            actual_seq_klen=None,
            layout="BSND",
            sparse_mode=3,
            pre_tokens=1048576,
            next_tokens=0,
        )
        ctx.save_for_backward(grad_query, grad_key, grad_weights)
        return loss[0] if isinstance(loss, (list, tuple)) else loss

    @staticmethod
    def backward(ctx, loss_gradient):
        gradients = tuple(value * loss_gradient for value in ctx.saved_tensors)
        return (None, None, *gradients, None, None, None, None, None, None)


@register_op("glm_indexer_kl_loss", "npu")
def indexer_kl_loss_npu(
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
    """Normalize the fused raw KL once by B*S, matching the original LLM path."""
    loss = _IndexerKL.apply(
        query, key, query_index, key_index, weights, indices, maximum, total, scale, query_rope, key_rope
    )
    return loss / (query.shape[0] * query.shape[1])

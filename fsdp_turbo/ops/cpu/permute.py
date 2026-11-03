# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021, OpenAI.

import torch
from fsdp_turbo.ops.registry import register_op


@register_op('permute', 'cpu')
def permute_cpu(tokens, indices, num_out_tokens=None):
    """
    CPU implementation of token permutation for MoE (Mixture of Experts).

    Args:
        tokens: Input tokens tensor [num_tokens, hidden_dim]
        indices: Expert indices tensor [num_tokens] or [num_tokens, topk]

    Returns:
        permuted_tokens: Permuted tokens
        sorted_indices: Indices for unpermutation
    """
    topk = 1 if indices.dim() == 1 else indices.size(1)
    indices_dtype = indices.dtype
    sorted_indices = torch.argsort(indices.float().view(-1), stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    sorted_indices1 = torch.argsort(sorted_indices.float(), stable=True).to(indices_dtype)
    if num_out_tokens:
        permuted_tokens = permuted_tokens[:num_out_tokens]
    return permuted_tokens, sorted_indices1


@register_op('unpermute', 'cpu')
def unpermute_cpu(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor | None = None,
    restore_shape: tuple[int, ...] | list[int] | torch.Size | None = None,
) -> torch.Tensor:
    """
    CPU implementation of token unpermutation for MoE (Mixture of Experts).

    Args:
        permuted_tokens: Permuted tokens tensor
        sorted_indices: Inverse permutation from permute_cpu, mapping each
            original route to its row in permuted_tokens.
        probs: Optional probability weights [num_tokens, topk]. Without weights,
            return the restored tokens directly without a reduction allocation.
        restore_shape: Optional output buffer shape before applying weights.

    Returns:
        unpermuted_tokens: Unpermuted tokens
    """
    num_tokens, topk = (permuted_tokens.size(0), 1) if probs is None else (probs.numel(), probs.size(1))
    if restore_shape:
        unpermuted_tokens = torch.zeros(restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
        permuted_tokens = permuted_tokens[: restore_shape[0]]
    else:
        unpermuted_tokens = torch.zeros(
            [num_tokens, permuted_tokens.shape[-1]], dtype=permuted_tokens.dtype, device=permuted_tokens.device
        )
    unpermuted_tokens.index_copy_(0, torch.argsort(sorted_indices, stable=True), permuted_tokens)
    if probs is None:
        return unpermuted_tokens
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    unpermuted_tokens *= probs.unsqueeze(-1)
    # Under NPU autocast, sum(dim=1) promotes bf16 to fp32; the fused kernel
    # preserves the input dtype, so restore it for the eager path.
    return unpermuted_tokens.sum(dim=1).to(permuted_tokens.dtype)

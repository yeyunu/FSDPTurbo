# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


@register_op('permute', 'npu')
def permute_npu(
    tokens: torch.Tensor, indices: torch.Tensor, num_out_tokens: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    NPU implementation of token permutation for MoE (Mixture of Experts).
    Uses optimized NPU kernel for better performance.

    Args:
        tokens: Input tokens tensor [num_tokens, hidden_dim]
        indices: Expert indices tensor [num_tokens] or [num_tokens, topk]

    Returns:
        permuted_tokens: Permuted tokens
        sorted_indices: Indices for unpermutation
    """
    return torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens=num_out_tokens)


@register_op('unpermute', 'npu')
def unpermute_npu(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor | None = None,
    restore_shape: tuple[int, ...] | list[int] | torch.Size | None = None,
) -> torch.Tensor:
    """
    NPU implementation of token unpermutation for MoE (Mixture of Experts).
    Uses optimized NPU kernel for better performance.
    Low-precision activations and FP32 routing probabilities retain their dtypes.

    Args:
        permuted_tokens: Permuted tokens tensor
        sorted_indices: Indices from permutation operation
        probs: Optional probability weights [num_tokens, topk]

    Returns:
        unpermuted_tokens: Unpermuted tokens
    """
    if restore_shape is None and permuted_tokens.size(0) != sorted_indices.numel():
        raise AssertionError(f"permuted tokens({permuted_tokens.size(0)}) != sorted indices({sorted_indices.size()})")
    return torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs, restore_shape=restore_shape)

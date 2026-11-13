# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Differentiable reference for full, image-isolated vision attention."""

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.registry import register_op


@register_op("fused_vision_flash_attention", "cpu")
def fused_vision_flash_attention_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sequence_ends: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Evaluate each packed image separately with SDPA, preserving autograd."""
    if sequence_ends is None:
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)
    return torch.cat(
        [
            F.scaled_dot_product_attention(
                query[start:end].transpose(0, 1),
                key[start:end].transpose(0, 1),
                value[start:end].transpose(0, 1),
            ).transpose(0, 1)
            for start, end in zip((0,) + sequence_ends, sequence_ends)
        ],
        dim=0,
    )

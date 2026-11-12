# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Native BSND fused attention over the exact selected support."""

import torch
import torch_npu

from fsdp_turbo.ops.registry import register_op


@register_op("selected_attention", "npu")
def selected_attention_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve native FA outputs/statistics used by GLM53 fused Indexer KL."""
    result = torch_npu.npu_fusion_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        query.shape[2],
        "BSND",
        atten_mask=mask,
        scale=scale,
        keep_prob=1.0,
        sparse_mode=0,
    )
    maximum, total = (tensor[..., 0].permute(0, 2, 1).unsqueeze(1).contiguous() for tensor in result[1:3])
    return result[0].contiguous(), maximum, total

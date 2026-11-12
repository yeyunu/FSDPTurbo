# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Portable reference for selected-support attention."""

import torch

from fsdp_turbo.ops.registry import register_op


@register_op("selected_attention", "cpu")
def selected_attention_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BSND values and B1SH max/sum using FP32 reference arithmetic."""
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * scale
    scores = scores.masked_fill(mask, float("-inf"))
    maximum = scores.amax(-1)
    total = (scores - maximum[..., None]).exp().sum(-1)
    output = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), value.float()).to(query.dtype)
    statistics = tuple(tensor.transpose(1, 2).unsqueeze(1).contiguous() for tensor in (maximum, total))
    return output.contiguous(), *statistics

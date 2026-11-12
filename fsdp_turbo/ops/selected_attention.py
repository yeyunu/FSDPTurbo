# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Attention on exactly the token support selected by an indexer."""

import torch

from fsdp_turbo.ops.registry import dispatch_op


def selected_support_mask(indices: torch.Tensor, key_length: int) -> torch.Tensor:
    """Return a B1QK mask (True means excluded), ignoring invalid/duplicate slots.

    ``indices`` has shape BQK_selected. The extra sentinel column avoids
    invalid slots overwriting the validity of selected token zero.
    """
    safe = indices.to(dtype=torch.int64, copy=True)
    safe.masked_fill_((safe < 0) | (safe >= key_length), key_length)
    mask = torch.ones(*indices.shape[:2], key_length + 1, dtype=torch.bool, device=indices.device)
    mask.scatter_(-1, safe, False)
    return mask[..., :key_length].unsqueeze(1).contiguous()


def selected_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute BSND attention and B1SH softmax max/sum on selected support.

    The caller supplies detached top-k selections, already constrained by the
    desired causal support. No additional dense attention support is added.
    """
    mask = selected_support_mask(indices, key.shape[1])
    return dispatch_op("selected_attention", query, key, value, mask, scale, device_type=query.device.type)

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU eager attention fallback (matmul + softmax float32).

Numerically identical to transformers ``eager_attention_forward``.
"""

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.registry import register_op


@register_op("flash_attention", "cpu")
def flash_attention_cpu(
    module,
    query,
    key,
    value,
    attention_mask=None,
    *,
    scaling=None,
    dropout=0.0,
    is_causal=True,
    **kwargs,
):
    """Eager attention: matmul + softmax(float32) + matmul.

    Returns output in BSHD ``[B, S, H, D]``.
    """
    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    n_rep = getattr(module, "num_key_value_groups", 1)
    if n_rep > 1:
        b, kv_h, s, d = key.shape
        key = key[:, :, None, :].expand(b, kv_h, n_rep, s, d).reshape(b, kv_h * n_rep, s, d)
        value = value[:, :, None, :].expand(b, kv_h, n_rep, s, d).reshape(b, kv_h * n_rep, s, d)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
    elif is_causal:
        causal_mask = torch.ones(attn_weights.shape[-2:], device=query.device, dtype=torch.bool).triu(diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous(), None

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch.nn.functional as F
from fsdp_turbo.ops.registry import register_op


@register_op('npu_fusion_attention', 'cpu')
def npu_fusion_attention_cpu(query, key, value, head_num, atten_mask, scale, keep_prob):
    """CPU fallback for the fused-attention op (BNSD layout via SDPA).

    Same mask convention as the NPU op: ``atten_mask`` is boolean with
    True == masked out.
    """
    attn_mask = None
    if atten_mask is not None:
        attn_mask = torch.zeros_like(atten_mask, dtype=query.dtype)
        attn_mask.masked_fill_(atten_mask, torch.finfo(query.dtype).min)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=1.0 - keep_prob,
        scale=scale,
    )

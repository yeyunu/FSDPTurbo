# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


@register_op('npu_fusion_attention', 'npu')
def npu_fusion_attention_npu(query, key, value, head_num, atten_mask, scale, keep_prob):
    """CANN fused attention, BNSD layout, boolean mask (True == masked out)."""
    return torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        head_num=head_num,
        input_layout="BNSD",
        pse=None,
        atten_mask=atten_mask,
        scale=scale,
        keep_prob=keep_prob,
        pre_tockens=2147483647,
        next_tockens=2147483647,
        inner_precise=1,
        sparse_mode=0,
    )[0]

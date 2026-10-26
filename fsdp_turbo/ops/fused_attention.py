# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from fsdp_turbo.ops import dispatch_op


def npu_fusion_attention(query, key, value, *, head_num, atten_mask=None, scale, keep_prob=1.0):
    """
    Fused attention with automatic device dispatch (BNSD layout).

    Args:
        query / key / value: [batch, heads, seq, dim] tensors (already padded
            to a common head dim and made contiguous by the caller).
        head_num: number of attention heads.
        atten_mask: boolean mask, True == masked out; None for full attention.
        scale: softmax scale.
        keep_prob: 1 - dropout probability.

    Returns:
        Attention output, [batch, heads, seq, dim].
    """
    return dispatch_op('npu_fusion_attention', query, key, value, head_num, atten_mask, scale, keep_prob)

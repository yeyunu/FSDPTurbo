# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from fsdp_turbo.ops.registry import register_op
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


@register_op("chunk_gated_delta_rule", "cuda")
def chunk_gdn_cuda(*args, **kwargs):
    """Call FLA chunk GDN through the active FLA backend."""
    return chunk_gated_delta_rule(*args, **kwargs)


@register_op("fused_recurrent_gated_delta_rule", "cuda")
def fused_recurrent_gdn_cuda(*args, **kwargs):
    """Call FLA recurrent GDN while leaving backend selection to FLA."""
    return fused_recurrent_gated_delta_rule(*args, **kwargs)


__all__ = [
    "chunk_gdn_cuda",
    "fused_recurrent_gdn_cuda",
]

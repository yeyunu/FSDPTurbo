# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from fsdp_turbo.ops.registry import register_op


@register_op("chunk_gated_delta_rule", "npu")
@register_op("chunk_gated_delta_rule", "cuda")
def chunk_gated_delta_rule(*args, **kwargs):
    """Call FLA chunk GDN through the active FLA backend."""

    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
    )

    return fla_chunk_gated_delta_rule(*args, **kwargs)


@register_op("fused_recurrent_gated_delta_rule", "npu")
@register_op("fused_recurrent_gated_delta_rule", "cuda")
def fused_recurrent_gated_delta_rule(*args, **kwargs):
    """Call FLA recurrent GDN while leaving backend selection to FLA."""

    from fla.ops.gated_delta_rule import (
        fused_recurrent_gated_delta_rule as fla_fused_recurrent_gated_delta_rule,
    )

    return fla_fused_recurrent_gated_delta_rule(*args, **kwargs)


__all__ = [
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
]

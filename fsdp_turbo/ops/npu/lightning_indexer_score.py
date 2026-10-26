# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""NPU backend of the DSA (lightning) indexer scoring operator.

Registers the fused Triton-Ascend implementation from
``fsdp_turbo.ops.triton.lightning_indexer_score``; arch35 devices (no cube-tile
support for this kernel) take the eager reference instead.
"""

from fsdp_turbo.ops.cpu.lightning_indexer_score import lightning_indexer_score_torch
from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.ops.triton.lightning_indexer_score import lightning_indexer_score_triton
from fsdp_turbo.ops.triton.utils import is_arch35


@register_op("lightning_indexer_score", "npu")
def lightning_indexer_score_npu(
    q,
    pool_keys,
    weights,
    valid_candidates=None,
    scale=None,
):
    """Run DSA indexer scoring with the Triton-Ascend fast path."""
    if is_arch35():
        return lightning_indexer_score_torch(q, pool_keys, weights, valid_candidates, scale)
    return lightning_indexer_score_triton(q, pool_keys, weights, valid_candidates, scale)


__all__ = ["lightning_indexer_score_npu"]

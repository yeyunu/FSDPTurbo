# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""NPU CCE registration without importing Triton during package initialization."""

from fsdp_turbo.ops.registry import register_op


@register_op('cce_loss', 'npu')
def cce_loss_npu(hidden_states, head_weight, shift_labels, vocab_tile_size=4096, ignore_index=-100):
    from fsdp_turbo.ops.triton.chunkloss_cce_kernels import ChunkLossCceFused

    return ChunkLossCceFused.apply(hidden_states, head_weight, shift_labels, vocab_tile_size, ignore_index)

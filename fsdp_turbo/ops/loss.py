# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Loss operators with device dispatch."""

from fsdp_turbo.ops import dispatch_op


def cce_loss(hidden_states, head_weight, shift_labels, vocab_tile_size=4096, ignore_index=-100):
    """Compute summed linear cross-entropy for flattened, already-shifted tokens."""
    return dispatch_op(
        'cce_loss',
        hidden_states,
        head_weight,
        shift_labels,
        vocab_tile_size,
        ignore_index,
        device_type=hidden_states.device.type,
    )

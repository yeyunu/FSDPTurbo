# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Pure PyTorch reference for summed linear cross-entropy."""

import torch.nn.functional as F

from fsdp_turbo.ops.registry import register_op


@register_op('cce_loss', 'cpu')
def cce_loss_cpu(hidden_states, head_weight, shift_labels, vocab_tile_size=4096, ignore_index=-100):
    logits = F.linear(hidden_states, head_weight).float()
    return F.cross_entropy(logits, shift_labels, ignore_index=ignore_index, reduction='sum')

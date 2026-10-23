# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CCE loss with optional token chunking; label shifting and normalization belong to the caller."""

import torch

from fsdp_turbo.ops.loss import cce_loss


def chunk_loss_cce_fused(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    shift_labels: torch.Tensor,
    vt: int = 4096,
    ignore_index: int = -100,
    seq_chunk_size: int | None = None,
) -> torch.Tensor:
    """Return the summed cross-entropy without constructing full logits on NPU.

    ``hidden_states`` is [B, S, H] or [N, H], ``head_weight`` is the full
    [V, H] classifier weight, and ``shift_labels`` contains the corresponding
    already-shifted labels. The NPU implementation uses bf16 inputs and Triton.
    ``vt`` controls vocabulary tile size. A positive ``seq_chunk_size`` limits
    the flattened valid tokens per invocation; None or <= 0 disables this limit.
    The CPU fallback uses ordinary PyTorch cross-entropy and materializes logits.
    """
    if vt <= 0:
        raise ValueError("vt (vocab tile size) must be positive")

    hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    labels = shift_labels.reshape(-1).contiguous()
    valid = labels != ignore_index
    num_valid = int(valid.sum())
    if num_valid == 0:
        # Keep both inputs in the graph without projecting ignored tokens.
        return hidden[:0].sum().float() + head_weight[:0].sum().float()
    if num_valid < labels.numel():
        hidden = hidden[valid]
        labels = labels[valid]

    if seq_chunk_size and 0 < seq_chunk_size < hidden.shape[0]:
        loss = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden.shape[0], seq_chunk_size):
            stop = start + seq_chunk_size
            loss = loss + cce_loss(hidden[start:stop], head_weight, labels[start:stop], vt, ignore_index)
        return loss
    return cce_loss(hidden, head_weight, labels, vt, ignore_index)

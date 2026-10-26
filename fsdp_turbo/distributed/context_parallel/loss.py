# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Causal-LM loss helpers for contiguous context-parallel sequence shards."""

from functools import wraps
from inspect import Parameter, signature
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fsdp_turbo.distributed.context_parallel.batch import pad_tensor_to_group, split_tensor_to_group


# Inspect once; CP supplies these four arguments explicitly.
_CROSS_ENTROPY_KWARGS = frozenset(
    name
    for name, parameter in signature(F.cross_entropy).parameters.items()
    if parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
) - {"input", "target", "ignore_index", "reduction"}


class _AllReduceForwardBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        output = input_.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * dist.get_world_size(ctx.group), None


def cp_all_reduce_forward_backward(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """All-reduce in forward and compensate for FSDP's CP-rank averaging."""
    return _AllReduceForwardBackward.apply(input_, group)


def cp_fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    group: dist.ProcessGroup,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    **kwargs: object,
) -> torch.Tensor:
    """Compute globally normalized CP cross entropy, ignoring model-forward kwargs.

    Existing cross-entropy options, including legacy reduction flags, are
    preserved; model-forward kwargs are ignored and CP sets ``reduction``.
    ``num_items_in_batch``, when provided, is the global normalization count.
    """
    # Collect the valid input parameters passed from different versions of transformers for cross_entropy.
    cross_entropy_kwargs = {key: value for key, value in kwargs.items() if key in _CROSS_ENTROPY_KWARGS}
    local_loss = F.cross_entropy(source, target, ignore_index=ignore_index, reduction="none", **cross_entropy_kwargs)

    valid_mask = target.ne(ignore_index)
    loss_sum = (local_loss * valid_mask).sum()

    reduce_item_count = num_items_in_batch is None
    if reduce_item_count:
        total_items = valid_mask.sum().to(loss_sum.device, dtype=loss_sum.dtype)
    else:
        total_items = torch.as_tensor(num_items_in_batch, device=loss_sum.device, dtype=loss_sum.dtype)

    loss_sum = cp_all_reduce_forward_backward(loss_sum, group)
    if reduce_item_count:
        dist.all_reduce(total_items, op=dist.ReduceOp.SUM, group=group)

    return loss_sum / total_items.clamp_min(1)


def cp_compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group: dist.ProcessGroup,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    *,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs: object,
) -> torch.Tensor:
    """Align complete labels with local CP logits, shifting only when needed.

    ``shift_labels``, if provided, must be complete and already shifted; it
    takes precedence over ``labels``. Both inputs follow the full-sequence contract.
    """
    logits = logits.float()

    if shift_labels is None:
        shifted_labels = torch.full_like(labels, ignore_index)
        shifted_labels[..., :-1] = labels[..., 1:]
    else:
        shifted_labels = shift_labels
    seq_dim = shifted_labels.dim() - 1
    shifted_labels = pad_tensor_to_group(shifted_labels, group, dim=seq_dim, pad_value=ignore_index)
    labels = split_tensor_to_group(shifted_labels, group, dim=seq_dim)
    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            "CP loss label/logit shape mismatch after complete-sequence shift and split: "
            f"local_labels={tuple(labels.shape)}, logits_without_vocab={tuple(logits.shape[:-1])}. "
            "Provide complete unshifted labels or complete pre-shifted shift_labels; "
            "model-side CP must produce local logits."
        )

    return cp_fixed_cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.reshape(-1).to(logits.device),
        group=group,
        num_items_in_batch=num_items_in_batch,
        ignore_index=ignore_index,
        **kwargs,
    )


def get_cp_causal_lm_loss_fn(fn: Callable[..., torch.Tensor], group: dist.ProcessGroup) -> Callable[..., torch.Tensor]:
    """Wrap a Transformers-style loss for CP-local logits and complete labels."""

    @wraps(fn)
    def wrapper(
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
        num_items_in_batch: torch.Tensor | int | None = None,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return cp_compute_lm_loss(
            logits,
            labels,
            group=group,
            num_items_in_batch=num_items_in_batch,
            ignore_index=ignore_index,
            shift_labels=shift_labels,
            **kwargs,
        )

    return wrapper

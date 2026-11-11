# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Causal-LM loss helpers for contiguous or front/back CP sequence shards."""

from functools import wraps
from inspect import Parameter, signature
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .sequence_partition import SequencePartitionPlan, pad_tensor_to_length


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


def shift_labels_globally(
    labels: torch.Tensor,
    boundaries=(),
    ignore_index: int = -100,
) -> torch.Tensor:
    """Shift a complete token stream by one, masking the stream tail and document starts.

    The shift happens on the whole stream, so a token at a document end would
    otherwise predict the next document's first token; every boundary is masked
    for that reason. Callers then slice their own segment out of the result.
    """
    masked = labels.clone()
    for boundary in boundaries:
        masked[..., boundary] = ignore_index
    return torch.nn.functional.pad(masked[..., 1:], (0, 1), value=ignore_index)


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


def _shift_and_shard_labels(
    labels: torch.Tensor, group: dist.ProcessGroup | None, ignore_index: int, *, zigzag: bool = False
) -> torch.Tensor:
    """Shift complete labels globally and split them onto the local CP shard.

    With ``group=None`` (no context parallelism) the labels are only shifted,
    so callers can share one implementation across CP and local runs.
    """
    shifted_labels = torch.full_like(labels, ignore_index)
    shifted_labels[..., :-1] = labels[..., 1:]
    if group is None:
        return shifted_labels
    partition = SequencePartitionPlan(
        shifted_labels.shape[-1], dist.get_world_size(group), dist.get_rank(group), zigzag
    )
    return partition.pad_and_shard_tensor(shifted_labels, dim=shifted_labels.dim() - 1, pad_value=ignore_index)


def cp_compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group: dist.ProcessGroup,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    zigzag: bool = False,
    *,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs: object,
) -> torch.Tensor:
    """Align complete labels with local CP logits, shifting only when needed.

    ``shift_labels``, if provided, must be complete and already shifted; it
    takes precedence over ``labels``. Both inputs follow the full-sequence contract.
    """
    logits = logits.float()

    seq_dim = (shift_labels if shift_labels is not None else labels).dim() - 1
    global_logits_length = logits.size(-2) * dist.get_world_size(group)
    label_length = (shift_labels if shift_labels is not None else labels).size(seq_dim)
    if label_length > global_logits_length:
        raise ValueError(
            "CP loss received complete labels longer than the global logits: "
            f"labels_length={label_length}, global_logits_length={global_logits_length}."
        )

    if shift_labels is None:
        # Shift on the complete sequence before padding to the model's length.
        shift_labels = _shift_and_shard_labels(labels, None, ignore_index)
    shift_labels = pad_tensor_to_length(shift_labels, seq_dim, global_logits_length, ignore_index)
    partition = SequencePartitionPlan(global_logits_length, dist.get_world_size(group), dist.get_rank(group), zigzag)
    labels = partition.pad_and_shard_tensor(shift_labels, dim=seq_dim, pad_value=ignore_index)
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


def cp_chunk_lm_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    head_weight: torch.Tensor,
    group: Optional[dist.ProcessGroup],
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    chunk_size: int = 1024,
    *,
    chunk_loss_fn: Callable[[torch.Tensor, torch.Tensor, list[dict[str, object]], int], torch.Tensor],
) -> torch.Tensor:
    """Chunked causal-LM loss on CP-local hidden states.

    ``chunk_loss_fn`` is the caller-supplied chunked LM kernel (per-chunk
    ``torch.func.grad_and_value``). Keeping it out of this module preserves the
    distributed/memory boundary (dev-guide rule 4): this helper only shifts and
    splits the complete labels, reduces the loss across the CP group, and
    normalizes by the global token count. Only one ``[chunk_size, vocab]``
    logits block exists at any time; the full ``[local_seq, vocab]`` fp32
    materialization of ``cp_compute_lm_loss`` costs ~10 bytes per entry and is
    infeasible at 1M-token sequences. The cross-rank normalization matches
    ``cp_fixed_cross_entropy``: all-reduce the loss sum with the backward
    compensation, divide by the global token count. ``num_items_in_batch``
    accepts the same Tensor-or-int form as ``cp_fixed_cross_entropy``.
    """
    if hidden_states.dim() != 3:
        raise ValueError(
            "CP chunk loss expects hidden_states with shape [batch, local_sequence, hidden]; "
            f"got {tuple(hidden_states.shape)}."
        )
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer; got {chunk_size!r}.")

    labels = labels.to(device=hidden_states.device)
    local_labels = _shift_and_shard_labels(labels, group, ignore_index)
    if local_labels.shape != hidden_states.shape[:-1]:
        raise ValueError(
            "CP chunk loss label/hidden shape mismatch after complete-sequence shift and split: "
            f"local_labels={tuple(local_labels.shape)}, hidden_without_last={tuple(hidden_states.shape[:-1])}. "
            "Labels must enter the model complete and unshifted, and model-side CP must produce local hidden states."
        )

    world_size = dist.get_world_size(group) if group is not None and dist.is_initialized() else 1

    local_valid = local_labels.ne(ignore_index)
    reduce_item_count = num_items_in_batch is None
    if reduce_item_count:
        total_items = local_valid.sum().to(device=hidden_states.device, dtype=torch.float32)
    else:
        total_items = torch.as_tensor(num_items_in_batch, device=hidden_states.device, dtype=torch.float32)

    label_chunks = torch.split(local_labels, chunk_size, dim=1)
    # fixed_cross_entropy always divides by alpha under reduction="sum"; the
    # unit alpha keeps every chunk an unnormalized sum so the CP reduction can
    # normalize once against the global token count.
    unit_alpha = torch.tensor(1.0, device=hidden_states.device)
    loss_kwargs_chunks = [
        {
            "shift_labels": label_chunk,
            "alpha": unit_alpha,
            "ignore_index": ignore_index,
            "reduction": "sum",
        }
        for label_chunk in label_chunks
    ]
    local_loss_sum = chunk_loss_fn(hidden_states, head_weight, loss_kwargs_chunks, chunk_size)

    if world_size > 1:
        loss_sum = cp_all_reduce_forward_backward(local_loss_sum, group)
        if reduce_item_count:
            dist.all_reduce(total_items, op=dist.ReduceOp.SUM, group=group)
    else:
        loss_sum = local_loss_sum
    return loss_sum / total_items.clamp_min(1)


def get_cp_causal_lm_loss_fn(
    fn: Callable[..., torch.Tensor], group: dist.ProcessGroup, *, zigzag: bool = False
) -> Callable[..., torch.Tensor]:
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
            zigzag=zigzag,
            shift_labels=shift_labels,
            **kwargs,
        )

    return wrapper

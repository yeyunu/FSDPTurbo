# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Model-agnostic context-parallel sequence sharding helpers."""

import torch
import torch.distributed as dist

from .communication import split_forward_gather_backward
from .utils import cal_split_sizes


DEFAULT_SEQUENCE_PAD_VALUES = {
    "input_ids": 0,
    "attention_mask": 0,
    "mm_token_type_ids": 0,
}


def sequence_parallel_range(seq_len: int, group):
    """Return the contiguous sequence interval owned by the current rank."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    sizes = cal_split_sizes(seq_len, world_size)
    start = sum(sizes[:rank])
    return start, start + sizes[rank]


def pad_tensor_to_group(tensor, group, dim: int, pad_value=0):
    """Right-pad ``tensor`` so ``dim`` is divisible by the group size.

    torch_npu's F.pad copy kernel computes element offsets in int32 and faults
    with an "MTE invalid GM address" aicore error once the tensor exceeds
    2^31 elements. Pad via torch.empty + chunked narrow copies instead.
    """
    dim %= tensor.dim()
    world_size = dist.get_world_size(group)
    pad_size = (world_size - tensor.size(dim) % world_size) % world_size
    if pad_size == 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] += pad_size
    out = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    out.narrow(dim, tensor.size(dim), pad_size).fill_(pad_value)
    # Chunk the copies so each narrow().copy_() stays well below 2^31 elements.
    per_row = max(1, tensor.numel() // tensor.size(dim))
    chunk = max(1, (1 << 29) // per_row)
    for s in range(0, tensor.size(dim), chunk):
        e = min(s + chunk, tensor.size(dim))
        out.narrow(dim, s, e - s).copy_(tensor.narrow(dim, s, e - s))
    return out


def split_tensor_to_group(tensor, group, dim: int = 1):
    """Return the current rank's contiguous shard from a divisible tensor."""
    dim %= tensor.dim()
    start, end = sequence_parallel_range(tensor.size(dim), group)
    shard = tensor.narrow(dim, start, end - start)
    if shard.is_contiguous():
        # A contiguous narrow (e.g. batch=1 split along the sequence dim) is
        # still a view that pins the complete tensor's storage -- ~15 GiB of
        # inputs_embeds per rank at 1M tokens with CP=16. Clone to release
        # the source; non-contiguous narrows keep the contiguous() copy path.
        return shard.clone()
    return shard.contiguous()


def split_sequence_batch(
    batch,
    group,
    split_dim: int = 1,
    pad_values=None,
    *,
    sequence_keys=("input_ids", "attention_mask", "mm_token_type_ids"),
    build_position_ids: bool = False,
):
    """Pad and split model inputs once over the complete CP group.

    ``labels`` deliberately remain complete and unshifted. The CP-aware loss
    performs the global shift, padding and matching split exactly once.
    """
    batch = dict(batch)
    input_ids = batch["input_ids"]
    if build_position_ids and not isinstance(batch.get("position_ids"), torch.Tensor):
        seq_len = input_ids.size(split_dim)
        batch["position_ids"] = (
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
        )

    pad_values = DEFAULT_SEQUENCE_PAD_VALUES if pad_values is None else pad_values
    seq_len = input_ids.size(split_dim)
    for key in set(sequence_keys):
        value = batch.get(key)
        if not isinstance(value, torch.Tensor) or value.dim() <= split_dim:
            continue
        if value.size(split_dim) != seq_len:
            continue
        value = pad_tensor_to_group(value, group, split_dim, pad_values.get(key, 0))
        batch[key] = split_tensor_to_group(value, group, split_dim)

    position_ids = batch.get("position_ids")
    if isinstance(position_ids, torch.Tensor) and position_ids.size(-1) == seq_len:
        position_ids = pad_tensor_to_group(position_ids, group, dim=-1, pad_value=0)
        batch["position_ids"] = split_tensor_to_group(position_ids, group, dim=-1)
    return batch


def shard_fused_sequence_inputs(
    *,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    group,
    gather_backward: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Pad and shard a complete fused multimodal sequence before the LM."""
    if inputs_embeds.dim() != 3:
        raise ValueError(
            f"Fused CP inputs_embeds must have shape [batch, sequence, hidden], got {tuple(inputs_embeds.shape)}."
        )
    if position_ids.dim() < 2 or position_ids.size(-1) != inputs_embeds.size(1):
        raise ValueError(
            "Fused CP position_ids must share the complete sequence length with "
            f"inputs_embeds; got position_ids={tuple(position_ids.shape)}, "
            f"inputs_embeds={tuple(inputs_embeds.shape)}."
        )
    if attention_mask is not None and (attention_mask.dim() != 2 or attention_mask.size(1) != inputs_embeds.size(1)):
        raise ValueError(
            "Post-fusion CP sharding supports only a [batch, sequence] attention_mask matching inputs_embeds."
        )

    world_size = dist.get_world_size(group)
    if attention_mask is None and inputs_embeds.size(1) % world_size:
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)

    inputs_embeds = pad_tensor_to_group(inputs_embeds, group, dim=1, pad_value=0)
    if gather_backward:
        split_sizes = cal_split_sizes(inputs_embeds.size(1), world_size)
        inputs_embeds = split_forward_gather_backward(inputs_embeds, group, dim=1, split_sizes=split_sizes)
    else:
        inputs_embeds = split_tensor_to_group(inputs_embeds, group, dim=1)
    position_ids = split_tensor_to_group(pad_tensor_to_group(position_ids, group, dim=-1, pad_value=0), group, dim=-1)
    if attention_mask is not None:
        attention_mask = split_tensor_to_group(
            pad_tensor_to_group(attention_mask, group, dim=1, pad_value=0), group, dim=1
        )
    return inputs_embeds, position_ids, attention_mask

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CP sequence coordinates, local sharding and post-collective reordering.

This module operates on tensors and integer coordinates only. It performs no
communication and has no model, process-group or device-backend dependency.
"""

from dataclasses import dataclass

import torch


_PAD_COPY_MAX_ELEMENTS = 1 << 29


def pad_tensor_to_length(tensor: torch.Tensor, dim: int, padded_length: int, pad_value: float = 0) -> torch.Tensor:
    """Right-pad one axis without using the NPU F.pad int32-offset path.

    Preserve dtype, device and local autograd. Return the original tensor when
    its length already matches; otherwise allocate a contiguous tensor and copy
    bounded sequence chunks. Empty axes are supported. Never truncate: a target
    shorter than the input raises ValueError. Invalid axes raise IndexError.
    """
    source_length = tensor.size(dim)
    dim %= tensor.ndim
    pad_size = padded_length - source_length
    if pad_size < 0:
        raise ValueError("padded_length must not be shorter than the input axis.")
    if pad_size == 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = padded_length
    output = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    output.narrow(dim, source_length, pad_size).fill_(pad_value)
    if source_length:
        elements_per_slice = max(1, tensor.numel() // source_length)
        chunk_size = max(1, _PAD_COPY_MAX_ELEMENTS // elements_per_slice)
        for start in range(0, source_length, chunk_size):
            length = min(chunk_size, source_length - start)
            output.narrow(dim, start, length).copy_(tensor.narrow(dim, start, length))
    else:
        # Preserve the local autograd edge even when the source axis is empty.
        output.narrow(dim, 0, 0).copy_(tensor)
    return output


@dataclass(frozen=True)
class SequencePartitionPlan:
    """Equal contiguous or front/back CP shards with right-side padding.

    ``global_length`` is the complete, unpadded token-axis length, including any
    caller-provided padding. It is not a valid-token count. ``cp_size`` is positive
    and ``rank`` is a coordinate in [0, cp_size); zero-length sequences are allowed.
    This layout does not describe packed sample boundaries or unequal shards.

    ``rank = k_rank * ulysses_size + u_rank``. Post-Ulysses front blocks increase
    in u order and back blocks decrease. Reordering methods do not communicate.
    """

    global_length: int
    cp_size: int
    rank: int
    zigzag: bool = False

    def __post_init__(self) -> None:
        if self.global_length < 0 or self.cp_size < 1:
            raise ValueError("global_length must be nonnegative and cp_size positive.")
        if not 0 <= self.rank < self.cp_size:
            raise ValueError("rank must be in [0, cp_size).")

    @property
    def padded_length(self) -> int:
        """Complete length rounded up to cp_size or 2*cp_size intervals."""
        parts = self.cp_size * (2 if self.zigzag else 1)
        return (self.global_length + parts - 1) // parts * parts

    @property
    def owned_intervals(self) -> tuple[tuple[int, int], ...]:
        """Global half-open intervals, in this rank's local storage order."""
        parts = self.cp_size * (2 if self.zigzag else 1)
        width = self.padded_length // parts
        slots = (self.rank, parts - 1 - self.rank) if self.zigzag else (self.rank,)
        return tuple((slot * width, (slot + 1) * width) for slot in slots)

    def pad_and_shard_tensor(self, tensor: torch.Tensor, dim: int = 1, pad_value: float = 0) -> torch.Tensor:
        """Right-pad a complete token axis and select this rank's local shard.

        The axis must equal global_length, not an already padded/local length;
        mismatches raise ValueError. Return a contiguous tensor with preserved
        dtype/device and local autograd, without modifying the input. The result
        owns its storage when cp_size > 1, so a local view cannot pin the full
        sequence allocation. With cp_size == 1 it may alias the input. No gradient
        gather or CP scaling is performed; use the communication autograd API.
        """
        if tensor.size(dim) != self.global_length:
            raise ValueError("pad_and_shard_tensor requires a complete token axis matching global_length.")
        tensor = pad_tensor_to_length(tensor, dim, self.padded_length, pad_value)
        if self.cp_size == 1:
            return tensor.contiguous()
        intervals = self.owned_intervals
        if not self.zigzag:
            start, end = intervals[0]
            return tensor.narrow(dim, start, end - start).clone(memory_format=torch.contiguous_format)
        return torch.cat([tensor.narrow(dim, start, end - start) for start, end in intervals], dim).contiguous()

    def reorder_ulysses(self, tensor: torch.Tensor, ulysses_size: int, *, inverse: bool = False) -> torch.Tensor:
        """Reorder [batch, sequence, ...] between A2A rank order and work order.

        In the Zigzag path the sequence axis must contain 2*ulysses_size equal
        blocks; inverse=True restores rank order before the reverse All-to-All.
        Ulysses size is a validated mesh dimension. Identity paths return input.
        """
        if not self.zigzag or ulysses_size == 1:
            return tensor
        if tensor.size(1) % (2 * ulysses_size):
            raise ValueError("Zigzag Ulysses sequence must contain 2*ulysses_size equal blocks.")
        order = list(range(0, 2 * ulysses_size, 2)) + list(range(2 * ulysses_size - 1, 0, -2))
        if inverse:
            order = sorted(range(len(order)), key=order.__getitem__)
        blocks = tensor.chunk(2 * ulysses_size, dim=1)
        return torch.cat([blocks[index] for index in order], dim=1).contiguous()

    def restore_gathered(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Restore chronological order after gathering every rank's shard.

        The gathered axis must equal padded_length; padding is retained. This
        method performs no AllGather and no cross-rank gradient synchronization.
        """
        if tensor.size(dim) != self.padded_length:
            raise ValueError("gathered token axis must match padded_length.")
        if not self.zigzag or self.cp_size == 1:
            return tensor
        blocks = tensor.chunk(2 * self.cp_size, dim)
        order = list(range(0, 2 * self.cp_size, 2)) + list(range(2 * self.cp_size - 1, 0, -2))
        return torch.cat([blocks[index] for index in order], dim).contiguous()

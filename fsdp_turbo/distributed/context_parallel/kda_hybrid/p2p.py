# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Ordered, subgroup-safe P2P exchange and differentiable affine prefix scan."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


class _Exchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, slots, owners, edges, group):
        ctx.exchange_spec = slots, owners, edges, group
        return _exchange(values, slots, owners, edges, group)

    @staticmethod
    def backward(ctx, gradient):
        slots, owners, edges, group = ctx.exchange_spec
        reverse_edges = tuple((target, source) for source, target in edges)
        return (
            _exchange(gradient.contiguous(), slots, owners, reverse_edges, group),
            None,
            None,
            None,
            None,
        )


def _exchange(values, slots, owners, edges, group):
    indices = {slot: index for index, slot in enumerate(slots)}
    peers = dist.get_process_group_ranks(group) if group is not None else [0]
    output = torch.zeros_like(values)
    operations, sends, receives = [], [], []
    for source, target in edges:
        if source in indices and target in indices:
            output[indices[target]].add_(values[indices[source]])
        elif source in indices:
            sends.append(values[indices[source]].contiguous())
            operations.append(dist.P2POp(dist.isend, sends[-1], peers[owners[target]], group))
        elif target in indices:
            buffer = torch.empty_like(values[indices[target]])
            receives.append((indices[target], buffer))
            operations.append(dist.P2POp(dist.irecv, buffer, peers[owners[source]], group))
    if operations:
        for request in dist.batch_isend_irecv(operations):
            request.wait()
    for index, buffer in receives:
        output[index].add_(buffer)
    return output


def exchange_parts(
    values: torch.Tensor,
    slots: tuple[int, ...],
    owners: tuple[int, ...],
    edges: tuple[tuple[int, int], ...],
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Exchange logical intervals in edge order; backward reverses and sums edges.

    Peers are translated from subgroup ranks to global ranks. Repeated messages
    between the same peers use the same globally ordered edge list, without tags.
    Every rank must participate, including ranks with no incoming edges.
    """
    return _Exchange.apply(values, slots, owners, edges, group)


def _compose(earlier, later):
    state_dim = later.shape[-2]
    result = later[..., state_dim:] @ earlier
    return torch.cat((result[..., :state_dim] + later[..., :state_dim], result[..., state_dim:]), -1)


@dataclass(frozen=True)
class KdaCPContext:
    """One or two logical intervals on a K-axis rank, with no saved activations."""

    group: dist.ProcessGroup | None
    rank: int
    size: int
    interval_length: int
    zigzag: bool

    @property
    def slots(self) -> tuple[int, ...]:
        """Logical chronological interval IDs owned by this K rank."""
        return (self.rank, 2 * self.size - 1 - self.rank) if self.zigzag else (self.rank,)

    @property
    def owners(self) -> tuple[int, ...]:
        """Owner subgroup rank for every logical interval."""
        front = tuple(range(self.size))
        return front + front[::-1] if self.zigzag else front

    @property
    def cu_seqlens(self) -> tuple[int, ...]:
        """Local scan resets; initial states reconnect the separated intervals."""
        return tuple(index * self.interval_length for index in range(len(self.slots) + 1))

    def compute_initial_states(self, summaries: torch.Tensor) -> torch.Tensor:
        """Return exclusive initial states in slots order, preserving autograd.

        Summaries are [E | M] per local interval. All state ranks participate;
        the first global interval receives zeros. No activations are retained.
        """
        return self._scan(summaries, forward=True)

    def compute_terminal_gradients(self, summaries: torch.Tensor) -> torch.Tensor:
        """Return terminal-state gradients from backward [E | M] summaries.

        Reverse the logical scan and return results in slots order. The last
        global interval receives zeros. Called explicitly by staged backward.
        """
        return self._scan(summaries, forward=False)

    def _scan(self, summaries: torch.Tensor, *, forward: bool) -> torch.Tensor:
        """Exclusive affine scan with logarithmic P2P rounds and local same-owner edges.

        Summaries are ``[E | M]``, mapping H to M H + E. Reverse logical order
        for terminal-state gradients. No state AllGather or final shift occurs.
        """
        count = len(self.owners)
        slots = self.slots if forward else tuple(count - 1 - slot for slot in self.slots)
        owners = self.owners if forward else self.owners[::-1]
        state_dim = summaries.shape[-2]
        identity = torch.eye(state_dim, device=summaries.device, dtype=summaries.dtype)
        exclusive = torch.cat(
            (
                torch.zeros_like(summaries[..., :state_dim]),
                identity.expand_as(summaries[..., state_dim:]),
            ),
            -1,
        )
        inclusive = summaries
        distance = 1
        while distance < count:
            edges = tuple((source, source + distance) for source in range(count - distance))
            received = exchange_parts(inclusive, slots, owners, edges, self.group)
            anchor = received.sum() * 0
            exclusive = torch.stack(
                [
                    _compose(received[index], exclusive[index]) if slot >= distance else exclusive[index] + anchor
                    for index, slot in enumerate(slots)
                ]
            )
            inclusive = torch.stack(
                [
                    _compose(received[index], inclusive[index]) if slot >= distance else inclusive[index] + anchor
                    for index, slot in enumerate(slots)
                ]
            )
            # Keep all exchange nodes in a single backward order on every rank.
            inclusive = inclusive + exclusive.sum() * 0
            exclusive = exclusive + inclusive.sum() * 0
            distance *= 2
        return exclusive[..., :state_dim]


def exchange_causal_history(parts: torch.Tensor, history_length: int, context: KdaCPContext) -> torch.Tensor:
    """Return chronological predecessor tokens for each local logical interval.

    ``parts`` has shape [local_intervals, interval_length, ...], ordered as
    ``context.slots``. Return [local_intervals, history_length, ...], preserving
    dtype/device and autograd. Missing global predecessors contribute zeros;
    history may span multiple intervals. Every rank must call with matching
    history length and context, even without incoming edges. Zero history
    returns an empty view without communication. No weights or activations are
    inspected; the caller owns convolution and any head/channel slicing.
    """
    if history_length < 0:
        raise ValueError("history_length must be nonnegative.")
    if parts.ndim < 2 or parts.shape[:2] != (len(context.slots), context.interval_length):
        raise ValueError("parts must match the local logical intervals and interval length.")
    if history_length == 0:
        return parts[:, :0]
    if context.interval_length < 1:
        raise ValueError("Positive history requires nonempty intervals.")
    tails = parts[:, -min(history_length, context.interval_length) :].contiguous()
    history = []
    for distance in range(1, (history_length + context.interval_length - 1) // context.interval_length + 1):
        edges = tuple((source, source + distance) for source in range(len(context.owners) - distance))
        history.append(exchange_parts(tails, context.slots, context.owners, edges, context.group))
    return torch.cat(history[::-1], dim=1)[:, -history_length:]

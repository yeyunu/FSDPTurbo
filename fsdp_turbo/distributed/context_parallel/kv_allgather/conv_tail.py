# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Causal-conv1d tail exchange across the KV-AllGather sequence axis.

After the Ulysses head all-to-all, each rank holds one contiguous sequence
segment (``[1, segment, local_channels]``) for its local head shard.  A causal
convolution over that segment needs the previous ``conv_kernel - 1`` tokens,
which live on the previous KV-AllGather rank (same head shard, because the
KV-AllGather group fixes the Ulysses coordinate).

The exchange is a single all-gather of every rank's segment tail: each rank
contributes its last ``width`` tokens once and prepends its predecessor's
contribution.  This keeps the op symmetric across ranks (no neighbor-only
send/recv chains whose blocking ordering has to be reasoned about) at the
cost of receiving ``K x width`` tokens instead of ``width`` — a few hundred
KB at most, negligible next to the layer's other traffic.  The backward is
the canonical pairing: every rank contributes a zero block holding the
gradient of the tokens it borrowed, and one ``reduce_scatter_tensor`` sums
in-network and hands each owner the total gradient of its own tail.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class _ConvTailExchange(torch.autograd.Function):
    """Prepend the previous rank's segment tail; return gradients symmetrically.

    Forward: all-gather the tails, rank ``r > 0`` concatenates rank ``r - 1``'s
    contribution in front of its own segment; rank 0 returns its input
    unchanged.  Backward: each rank contributes a ``[K, B, width, C]`` block
    that is zero except the slot of the predecessor it borrowed from;
    ``reduce_scatter_tensor`` sums the contributions and delivers every rank
    the gradient of its own tail (i.e. its successor's borrowed-prefix
    gradient).
    """

    @staticmethod
    def forward(ctx, x, group, width):
        rank = dist.get_rank(group)
        size = dist.get_world_size(group)
        ctx.group = group
        ctx.width = width
        ctx.rank = rank
        ctx.size = size
        if size == 1 or width == 0:
            ctx.identity = True
            return x
        ctx.identity = False

        tail = x[:, -width:, :].contiguous()
        shards = [torch.empty_like(tail) for _ in range(size)]
        dist.all_gather(shards, tail, group=group)
        if rank == 0:
            return x
        return torch.cat([shards[rank - 1], x], dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.identity:
            return grad_output, None, None
        group, width = ctx.group, ctx.width
        rank, size = ctx.rank, ctx.size

        # Zero everywhere except the predecessor's slot; reduce-scatter sums
        # across ranks so each owner receives its successor's contribution.
        # The exchange is flattened to 1-D: gloo's reduce_scatter_tensor
        # requires input and output to share their rank count, and the flat
        # chunk boundaries coincide exactly with one rank's contribution.
        contribution = grad_output.new_zeros((size, grad_output.shape[0], width, grad_output.shape[2]))
        if rank > 0:
            contribution[rank - 1] = grad_output[:, :width, :]
        incoming = torch.empty(
            (grad_output.shape[0], width, grad_output.shape[2]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.reduce_scatter_tensor(
            incoming.view(-1),
            contribution.view(-1),
            op=dist.ReduceOp.SUM,
            group=group,
        )

        local_grad = grad_output[:, width:, :] if rank > 0 else grad_output
        if rank < size - 1:
            local_grad = local_grad.clone()
            local_grad[:, -width:, :] += incoming
        return local_grad, None, None


def conv_tail_exchange(x: torch.Tensor, group, width: int):
    """Exchange causal-conv tails across the KV-AllGather group.

    Args:
        x: post-all-to-all segment activations ``[1, segment, channels]``.
        group: KV-AllGather process group.
        width: number of leading tokens to borrow (``conv_kernel - 1``).

    Returns:
        ``(x_ext, borrowed)`` where ``x_ext`` has ``width`` tokens prepended
        when this rank borrowed them, and ``borrowed`` says whether the first
        ``width`` convolution outputs must be discarded.
    """
    size = dist.get_world_size(group)
    borrowed = size > 1 and width > 0 and dist.get_rank(group) > 0
    if size == 1 or width == 0:
        return x, False
    x_ext = _ConvTailExchange.apply(x, group, width)
    return x_ext, bool(borrowed)

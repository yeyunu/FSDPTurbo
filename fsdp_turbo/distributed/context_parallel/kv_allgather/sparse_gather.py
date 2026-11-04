# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Data-dependent row gather across a CP group via variable-size all-to-all.

Used by the QSA sparse-attention path: after per-chunk top-k selection, each
rank needs the K/V rows of a small, rank-specific set of tokens owned by the
other CP ranks.  The exchange is a two-phase variable-size all-to-all (request
offsets, then data) whose backward is the transposed all-to-all, so gradients
flow back to the owning ranks exactly.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class _UnequalAllToAll(torch.autograd.Function):
    """Variable-split all-to-all whose backward is the transposed exchange.

    ``torch.distributed.nn.functional.all_to_all`` changes signature across
    torch releases (2.7 ships the tensor-list variant without split sizes),
    so the exchange is wrapped directly around ``dist.all_to_all_single``.
    Splits must be pairwise consistent, which the count exchange below
    guarantees; the gloo backend additionally requires each rank's
    self-split to be balanced, and a rank's requests to itself are identical
    on both sides by construction.
    """

    @staticmethod
    def forward(ctx, input_tensor, output_split_sizes, input_split_sizes, group):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        output = input_tensor.new_empty((sum(output_split_sizes), *input_tensor.shape[1:]))
        dist.all_to_all_single(
            output,
            input_tensor.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty((sum(ctx.input_split_sizes), *grad_output.shape[1:]))
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
        )
        return grad_input, None, None, None


def _owner_rank(global_indices: torch.Tensor, shard_length: int) -> torch.Tensor:
    return torch.div(global_indices, shard_length, rounding_mode="floor")


def sparse_gather_rows(
    tensors: tuple[torch.Tensor, ...],
    global_indices: torch.Tensor,
    shard_length: int,
    group,
):
    """Gather rows of CP-sharded tensors at global token indices.

    Args:
        tensors: local shards ``[1, H, T, D]`` (identical ``T``), sequence dim 2.
        global_indices: sorted unique ``int64`` tensor of GLOBAL token ids.
        shard_length: per-rank sequence length ``T``.
        group: flattened CP process group.

    Returns:
        Tuple of ``[1, H, n, D]`` tensors whose dim-2 rows align with
        ``global_indices`` order.  Gradients are routed back to the owning
        ranks (scatter-added through the serving rank's ``index_select``).
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tuple(t.index_select(2, global_indices) for t in tensors)
    if not tensors:
        raise ValueError("sparse_gather_rows requires at least one tensor")
    if global_indices.numel() == 0:
        shapes = tensors[0].shape
        return tuple(t.new_zeros((shapes[0], shapes[1], 0, shapes[3])) for t in tensors)

    indices = global_indices.to(torch.int64)
    owners = _owner_rank(indices, shard_length)
    if int(owners.max()) >= world_size:
        raise ValueError("sparse_gather_rows received out-of-range global indices")

    # Requests I send: for each source rank s, my local offsets into its shard.
    # ``indices`` is sorted, hence owner-grouped, so the flat order already
    # groups requests by source rank in rank order.
    send_counts = torch.bincount(owners, minlength=world_size).to(torch.int64)
    local_offsets = indices - owners * shard_length

    # Phase 1a: exchange request counts (one integer per rank).
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)

    # Phase 1b: exchange the requested local offsets.  ``recv_offsets`` is
    # grouped by requester rank, preserving each requester's order, which is
    # exactly the order the data phase must answer in.
    recv_offsets = torch.empty(int(recv_counts.sum()), dtype=torch.int64, device=indices.device)
    dist.all_to_all_single(
        recv_offsets,
        local_offsets,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
        group=group,
    )

    # Phase 2: serve the received requests and exchange the data rows.  The
    # backward of the unequal all-to-all is the transposed exchange, and
    # ``index_select``'s backward scatter-adds duplicate requests.
    served = [t.index_select(2, recv_offsets) for t in tensors]
    head_count = tensors[0].shape[1]
    head_dim = tensors[0].shape[3]
    # [n_recv, len(tensors) * H * D], rows laid out as t[0, :, idx, :].flatten.
    payload = torch.cat([s.permute(0, 2, 1, 3).flatten(2).squeeze(0) for s in served], dim=-1).contiguous()
    gathered = _UnequalAllToAll.apply(payload, send_counts.tolist(), recv_counts.tolist(), group)

    n_needed = indices.numel()
    row_width = payload.shape[-1]
    outputs = []
    offset = 0
    for _ in tensors:
        chunk = gathered[:, offset : offset + head_count * head_dim]
        chunk = chunk.reshape(n_needed, head_count, head_dim)
        outputs.append(chunk.permute(1, 0, 2).unsqueeze(0))
        offset += head_count * head_dim
    if offset != row_width:
        raise RuntimeError("sparse gather row width mismatch")
    return tuple(outputs)

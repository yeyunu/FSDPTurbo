# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Vocabulary-parallel embedding forward (all-to-all routing)."""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from .comm import alltoallv, alltoallv_with_gradient_divide_factor


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    while isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor


def _exchange_counts(send_counts: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return send_counts
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts.contiguous(), group=group)
    return recv_counts


def vocab_parallel_embedding_forward(
    module: torch.nn.Embedding,
    input_ids: torch.Tensor,
    vp_group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """Route embedding IDs to vocabulary owners and return results in input order."""
    vp_size = 1 if vp_group is None else dist.get_world_size(vp_group)
    vp_rank = 0 if vp_group is None else dist.get_rank(vp_group)
    local_vocab_size = module.num_embeddings // vp_size
    flat_ids = input_ids.reshape(-1).long()
    owners = torch.div(flat_ids, local_vocab_size, rounding_mode="floor")

    # NPU integer argsort falls back to AI CPU. Owner ranks are exactly
    # representable in float32 within this bound; vocabulary IDs stay int64.
    # Ties may be reordered, but the inverse permutation restores input order.
    sort_keys = owners
    if owners.device.type == "npu" and vp_size <= 2**24:
        sort_keys = owners.to(torch.float32)
    permutation = torch.argsort(sort_keys)
    sorted_owners = owners[permutation]
    sorted_local_ids = flat_ids[permutation] - sorted_owners * local_vocab_size

    # NPU-safe owner histogram: bincount/histc/scatter_add_ fall back to AI
    # CPU for integer inputs (and the AICPU ScatterElements kernel fails);
    # an eq + sum reduction stays on AI cores.
    owner_ids = torch.arange(vp_size, device=owners.device)
    send_counts_tensor = (owners.unsqueeze(-1) == owner_ids).sum(dim=0).to(dtype=torch.int64)
    recv_counts_tensor = _exchange_counts(send_counts_tensor, vp_group)
    send_counts = send_counts_tensor.cpu().tolist()
    recv_counts = recv_counts_tensor.cpu().tolist()

    received_ids = alltoallv(sorted_local_ids, vp_group, send_counts, recv_counts)
    local_weight = _local_tensor(module.weight)

    local_padding_idx = None
    if module.padding_idx is not None:
        vocab_start = vp_rank * local_vocab_size
        if vocab_start <= module.padding_idx < vocab_start + local_vocab_size:
            local_padding_idx = module.padding_idx - vocab_start

    local_embeddings = F.embedding(
        received_ids,
        local_weight,
        padding_idx=local_padding_idx,
        max_norm=module.max_norm,
        norm_type=module.norm_type,
        scale_grad_by_freq=module.scale_grad_by_freq,
        sparse=module.sparse,
    )
    output_dtype = module._vp_output_dtype
    if output_dtype is not None and local_embeddings.dtype != output_dtype:
        local_embeddings = local_embeddings.to(dtype=output_dtype)
    sorted_embeddings = alltoallv_with_gradient_divide_factor(
        local_embeddings,
        vp_group,
        recv_counts,
        send_counts,
        gradient_divide_factor=module._vp_gradient_divide_factor,
    )

    # ``inverse[perm] = arange`` is mathematically ``argsort(perm)``. Integer
    # argsort (and index_put) fall back to AI CPU on NPU, so reuse the
    # float32-key trick whenever the index range is exactly representable.
    if permutation.device.type == "npu" and permutation.numel() < 2**24:
        inverse_permutation = torch.argsort(permutation.to(torch.float32))
    else:
        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    return sorted_embeddings[inverse_permutation].view(*input_ids.shape, module.embedding_dim)

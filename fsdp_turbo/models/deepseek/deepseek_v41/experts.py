# Copyright c) 2024 NVIDIA CORPORATION.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""Packed DeepSeek-V4.1 routed experts compatible with FSDPTurbo EP."""

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .activation import limited_swiglu


def _use_fused_permute(hidden_states, global_indices):
    # Keep both local permutation pairs on the same backend. Empty send or
    # receive buffers retain eager routing, but must still enter AllToAll.
    return hidden_states.device.type == "npu" and hidden_states.numel() > 0 and global_indices.numel() > 0


def _reorder_router_weights(weights, row_map):
    # Both eager and NPU permutation return original-route -> permuted-row.
    # Scatter preserves FP32 probabilities and their autograd connection.
    return torch.empty_like(weights).scatter(0, row_map.long(), weights)


def _build_fixed_dispatch_metadata(
    num_tokens: int,
    topk: int,
    num_global_experts: int,
    ep_size: int,
    ep_rank: int,
    device: torch.device,
):
    """Build device-independent routing metadata for the debug fixed router.

    Every EP rank receives the same round-robin route in fixed-router mode, so
    its split sizes can be computed on the host without a histogram, all-gather,
    D2H copy, or Tensor.tolist() in the training step.
    """
    counts = [0] * num_global_experts
    for token in range(num_tokens):
        for choice in range(topk):
            counts[(token + choice) % num_global_experts] += 1

    num_local_experts = num_global_experts // ep_size
    send_splits = [sum(counts[rank * num_local_experts : (rank + 1) * num_local_experts]) for rank in range(ep_size)]
    recv_splits = [send_splits[ep_rank]] * ep_size
    local_start = ep_rank * num_local_experts
    local_counts_list = [counts[local_start + index] * ep_size for index in range(num_local_experts)]
    global_indices_list = []
    for _ in range(ep_size):
        for local_index in range(num_local_experts):
            global_indices_list.extend([local_index] * counts[local_start + local_index])

    local_counts = torch.tensor(local_counts_list, dtype=torch.int64, device=device)
    global_indices = torch.tensor(global_indices_list, dtype=torch.int32, device=device)
    return (local_counts, global_indices), (send_splits, recv_splits)


def dispatch_preprocess(ep_group, top_k_index, num_global_experts, expert_ids_per_ep_rank):
    from fsdp_turbo.distributed.dist_ops import gather_along_first_dim_expert_parallel

    ep_size = torch.distributed.get_world_size(ep_group)
    ep_rank = torch.distributed.get_rank(ep_group)
    num_local_experts = num_global_experts // ep_size
    local_experts_start_id = num_local_experts * ep_rank
    local_experts_end_id = local_experts_start_id + num_local_experts

    # [B*S, K] --> [E]
    num_local_tokens_per_expert = torch.histc(
        top_k_index.float(), bins=num_global_experts, min=0, max=num_global_experts
    ).long()
    # [E] --> [EP*E]
    num_global_tokens_per_expert, _ = gather_along_first_dim_expert_parallel(num_local_tokens_per_expert, ep_group)
    # [EP*E] --> [EP, local_E]
    num_global_tokens_per_local_expert = num_global_tokens_per_expert.reshape(ep_size, num_global_experts)[
        :, local_experts_start_id:local_experts_end_id
    ]
    # [EP, local_E] --> [local_E]
    num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
    # [E] --> [EP, local_E] --> [EP]
    input_split = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(axis=1)
    # [EP, local_E] --> [EP]
    output_splits = num_global_tokens_per_local_expert.sum(axis=-1)
    # all_to_all_single currently requires host-side split lists. Copy both
    # vectors together and materialize them once: the old path called
    # Tensor.tolist() again for dispatch, router weights, and combine, which
    # expanded into thousands of aten::item calls for a deep MoE model.
    split_values = torch.cat((input_split, output_splits)).to(torch.device("cpu"))
    split_values = split_values.numpy().tolist()
    input_split = split_values[:ep_size]
    output_splits = split_values[ep_size:]
    # [EP, local_E] --> [E*select]
    global_indices = torch.repeat_interleave(expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
    return (num_tokens_per_local_expert, global_indices), (input_split, output_splits)


def alltoall_dispatch(ep_group, hidden_states, top_k_index, indices, split_sizes, fused):
    from torch.distributed._functional_collectives import all_to_all_single_autograd
    from fsdp_turbo.ops.moe import permute

    local_indices, global_indices = indices
    input_split, output_splits = split_sizes

    hidden_states, unpermute_indices1 = permute(hidden_states, top_k_index, use_eager=not fused)
    hidden_states = all_to_all_single_autograd(hidden_states, output_splits, input_split, ep_group)
    hidden_states, unpermute_indices2 = permute(hidden_states, global_indices, use_eager=not fused)

    return hidden_states, (unpermute_indices1, unpermute_indices2)


def _unpermute_preweighted(hidden_states, inverse_map, routing_shape):
    """Restore expert slots and sum them in the eager weighted path's order.

    Router weights have already been applied before the down projection.
    Avoid multiplying by ones, which also saves an autograd copy of the
    restored activations. Keep the same FP32 sum as the caller's input.
    """
    num_tokens, topk = routing_shape
    restored = hidden_states.index_select(0, inverse_map.long())
    return restored.reshape(num_tokens, topk, hidden_states.shape[-1]).sum(dim=1)


def alltoall_combine(
    ep_group,
    hidden_states,
    top_k_weights,
    unpermute_indices,
    split_sizes,
    fused,
    *,
    weights_applied=False,
):
    from torch.distributed._functional_collectives import all_to_all_single_autograd
    from fsdp_turbo.ops.moe import unpermute

    if weights_applied and fused:
        raise ValueError("Preweighted combine currently supports only the eager permutation path.")
    unpermute_indices1, unpermute_indices2 = unpermute_indices
    input_split, output_splits = split_sizes
    hidden_states = unpermute(hidden_states, unpermute_indices2, use_eager=not fused)
    hidden_states = all_to_all_single_autograd(hidden_states, input_split, output_splits, ep_group)
    if weights_applied:
        hidden_states = _unpermute_preweighted(hidden_states, unpermute_indices1, top_k_weights.shape)
    else:
        hidden_states = unpermute(hidden_states, unpermute_indices1, top_k_weights, use_eager=not fused)
    return hidden_states


class DeepSeekV41Experts(nn.Module):
    """Store routed experts as two stacked linear-weight tensors.

    The layout follows FSDPTurbo's expert dispatcher convention:

    - ``gate_up_proj``: ``[num_experts, 2 * intermediate_size, hidden_size]``
    - ``down_proj``: ``[num_experts, hidden_size, intermediate_size]``
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        swiglu_limit: float = 0.0,
        use_fused_clipped_swiglu: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.intermediate_size = intermediate_size
        self.limit = swiglu_limit if swiglu_limit > 0 else None
        self.use_fused_clipped_swiglu = use_fused_clipped_swiglu
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                dtype=dtype,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size,
                dtype=dtype,
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run routed experts with grouped projections on supported NPU inputs."""
        if (
            hidden_states.device.type == "npu"
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and hidden_states.numel() > 0
            and topk_indices.numel() > 0
        ):
            return self._forward_grouped(hidden_states, topk_indices, topk_weights)
        return self._forward_eager(hidden_states, topk_indices, topk_weights)

    def _forward_eager(self, hidden_states, topk_indices, topk_weights):
        """Original per-expert arithmetic, also the accuracy reference."""
        # The per-expert grouping below (unique/where) is data-dependent: under activation
        # recompute, optimizer updates shift router outputs slightly, so the replayed forward
        # can group one token differently and the saved-tensor counts diverge (CheckpointError).
        # Identity hooks keep these data-dependent saves out of the checkpoint's accounting;
        # the graph itself is identical, so gradients are unaffected.
        with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
            return self._forward_eager_impl(hidden_states, topk_indices, topk_weights)

    def _forward_eager_impl(self, hidden_states, topk_indices, topk_weights):
        """Original per-expert arithmetic, also the accuracy reference."""
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        topk_indices = topk_indices.reshape(hidden_states.size(0), -1)
        topk_weights = topk_weights.reshape_as(topk_indices)
        output = torch.zeros_like(hidden_states, dtype=torch.float32)

        active_expert_indices = torch.unique(topk_indices).tolist()
        for expert_index in active_expert_indices:
            token_indices, topk_positions = torch.where(topk_indices == expert_index)
            selected_hidden_states = hidden_states[token_indices]
            gate_up_states = F.linear(
                selected_hidden_states,
                self.gate_up_proj[expert_index],
            ).float()
            intermediate_states = limited_swiglu(gate_up_states, self.limit, fused=self.use_fused_clipped_swiglu)
            intermediate_states = intermediate_states * topk_weights[token_indices, topk_positions, None]
            expert_output = F.linear(
                intermediate_states.to(hidden_states.dtype),
                self.down_proj[expert_index],
            )
            output = output.index_add(0, token_indices, expert_output.float())

        # Reference MoE accumulates routed experts and the shared expert in
        # FP32, then performs one final cast in MoE.forward.
        return output.view(original_shape)

    def _forward_grouped(self, hidden_states, topk_indices, topk_weights):
        """Group projections only; preserve per-expert activation and sum order.

        CPU execution of this private method is useful for structural tests;
        the public fast path is restricted to supported NPU input dtypes.
        """
        # Same data-dependent-grouping hazard as _forward_eager: isolate the saves from
        # the checkpoint's accounting so replayed groupings cannot desynchronize it.
        with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
            return self._forward_grouped_impl(hidden_states, topk_indices, topk_weights)

    def _forward_grouped_impl(self, hidden_states, topk_indices, topk_weights):
        """Group projections only; preserve per-expert activation and sum order.

        CPU execution of this private method is useful for structural tests;
        the public fast path is restricted to supported NPU input dtypes.
        """
        from fsdp_turbo.ops.moe import grouped_matmul

        if hidden_states.numel() == 0 or topk_indices.numel() == 0:
            return self._forward_eager(hidden_states, topk_indices, topk_weights)
        original_shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_dim)
        ids = topk_indices.reshape(x.size(0), -1)
        probs = topk_weights.reshape_as(ids)
        active = torch.unique(ids).tolist()
        selections = [torch.where(ids == expert) for expert in active]
        sizes = [rows.numel() for rows, _ in selections]
        counts = torch.tensor(sizes, dtype=torch.int64, device=x.device)
        expert_ids = torch.tensor(active, dtype=torch.long, device=x.device)

        def select_weights(weights):
            # unique returns sorted ids: all-active weights need no copy.
            if len(active) == self.num_experts:
                return weights
            return weights.index_select(0, expert_ids)

        # The shared dispatcher otherwise follows the current accelerator,
        # which can be NPU even for CPU tensors in mixed-device processes.
        use_eager = x.device.type == "cpu"
        packed_x = torch.cat([x[rows] for rows, _ in selections], dim=0)
        gate_up = grouped_matmul(packed_x, counts, select_weights(self.gate_up_proj), use_eager=use_eager)
        intermediates = []
        for states, (rows, slots) in zip(gate_up.split(sizes), selections):
            states = states.float()
            intermediate = limited_swiglu(states, self.limit, fused=self.use_fused_clipped_swiglu)
            # Match eager type promotion and apply probabilities before down.
            intermediate = intermediate * probs[rows, slots, None]
            intermediates.append(intermediate.to(x.dtype))
        packed_output = grouped_matmul(
            torch.cat(intermediates, dim=0),
            counts,
            select_weights(self.down_proj),
            use_eager=use_eager,
        )
        output = torch.zeros_like(x, dtype=torch.float32)
        for contribution, (rows, _) in zip(packed_output.split(sizes), selections):
            output = output.index_add(0, rows, contribution.float())
        return output.view(original_shape)

    def native_eager_forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        ep_group,
        fixed_router: bool = False,
    ) -> torch.Tensor:
        """Run EP with the released DeepSeek expert arithmetic order.

        This is selected through FSDPTurbo's existing ``custom_*`` dispatcher
        extension point. NPU token routing uses paired fused permutations,
        while probability application and FP32 combination retain the native
        arithmetic order. The two expert projections use the accelerator's
        grouped GEMM. CPU/CUDA and empty routing buffers retain eager routing.
        """
        from torch.distributed._functional_collectives import (
            all_to_all_single_autograd,
        )
        from torch.distributed.tensor import DTensor

        from fsdp_turbo.distributed.expert_parallel.utils import (
            fixed_router_for_debug,
            normalize_expert_args,
        )
        from fsdp_turbo.ops.moe import grouped_matmul

        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        topk_indices, topk_weights = normalize_expert_args(topk_indices, topk_weights)
        if fixed_router:
            topk = 1 if topk_indices.dim() == 1 else topk_indices.shape[1]
            cache_key = (
                topk_indices.shape[0],
                topk,
                topk_indices.device,
                topk_indices.dtype,
                topk_weights.dtype,
            )
            cached = getattr(self, "_fixed_router_cache", None)
            if cached is None or cached[0] != cache_key:
                fixed_indices, fixed_weights = fixed_router_for_debug(
                    topk_indices,
                    topk_weights,
                    self.num_global_experts,
                )
                metadata = _build_fixed_dispatch_metadata(
                    fixed_indices.shape[0],
                    topk,
                    self.num_global_experts,
                    torch.distributed.get_world_size(ep_group),
                    torch.distributed.get_rank(ep_group),
                    fixed_indices.device,
                )
                cached = (cache_key, fixed_indices, fixed_weights, metadata)
                self._fixed_router_cache = cached
            _, topk_indices, topk_weights, (indices, split_sizes) = cached
        else:
            indices, split_sizes = dispatch_preprocess(
                ep_group,
                topk_indices,
                self.num_global_experts,
                self.expert_ids_per_ep_rank,
            )
        local_counts, global_indices = indices
        input_split, output_splits = split_sizes
        fused_permute = _use_fused_permute(hidden_states, global_indices)

        hidden_states, unpermute_indices = alltoall_dispatch(
            ep_group,
            hidden_states,
            topk_indices,
            indices,
            split_sizes,
            fused=fused_permute,
        )
        source_order, expert_order = unpermute_indices
        router_weights = _reorder_router_weights(topk_weights.reshape(-1), source_order)
        router_weights = all_to_all_single_autograd(
            router_weights,
            output_splits,
            input_split,
            ep_group,
        )
        router_weights = _reorder_router_weights(router_weights, expert_order)
        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj

        gate_up = grouped_matmul(
            hidden_states,
            local_counts,
            gate_up_proj,
            use_eager=False,
        ).float()
        intermediate = limited_swiglu(gate_up, self.limit, fused=self.use_fused_clipped_swiglu)
        # Do not mutate an activation output that a fused backward may save.
        intermediate = intermediate * router_weights.float().unsqueeze(-1)
        expert_output = grouped_matmul(
            intermediate.to(hidden_states.dtype),
            local_counts,
            down_proj,
            use_eager=False,
        )

        # Probabilities are already applied. Fused routing needs its paired
        # inverse map and unit weights; eager routing can skip that multiply.
        # Keep return AllToAll and summation FP32 in both paths.
        combine_weights = torch.ones_like(topk_weights, dtype=torch.float32) if fused_permute else topk_weights
        output = alltoall_combine(
            ep_group,
            expert_output.float(),
            combine_weights,
            unpermute_indices,
            split_sizes,
            fused=fused_permute,
            weights_applied=not fused_permute,
        )
        return output.view(*original_shape)


class DispatchOverlapExperts(DeepSeekV41Experts):
    """Accept the EP dispatch callback, with sequential execution when EP is absent."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        dispatch_overlap: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Execute the independent branch once, then use the native local experts."""
        if dispatch_overlap is not None:
            dispatch_overlap()
        return super().forward(hidden_states, topk_indices, topk_weights)


__all__ = ["DeepSeekV41Experts", "DispatchOverlapExperts"]

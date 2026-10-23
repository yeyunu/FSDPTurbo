# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""AllGather EP dispatch with differentiable SUM ReduceScatter combination.

Communication APIs are migrated from fsdp_training's comm.py.
"""

from typing import Optional, Callable
import torch
import torch.distributed as dist
from fsdp_turbo.distributed.expert_parallel.dispatcher import experts_computation
from fsdp_turbo.distributed.expert_parallel.utils import get_num_global_tokens_per_local_expert
from fsdp_turbo.ops.moe import permute, unpermute
from fsdp_turbo.distributed.expert_parallel.ep_balance import EPBalanceWeights
from fsdp_turbo.distributed.expert_parallel.comm import (
    set_ep_rank_seq_lens,
    allgather_tokens_in_ep,
    reduce_scatter_tokens_in_ep,
)


def all_gather_experts_forward_fn(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    weights: torch.Tensor,
    num_global_experts: int,
    expert_ids_per_ep_rank: list[int],
    ep_group: torch.distributed.ProcessGroup,
    fused: bool,
    act_fn: Optional[Callable] = None,
    act_limit: Optional[float] = None,
    act_alpha: Optional[float] = None,
    ep_balance_strategy=None,
    quant_config=None,
    expert_module=None,
    **kwargs,
):
    gate_up_weights, down_weights = weights

    # --- Expert Parallelism (EP) AllGather Phase ---
    # In EP, each rank only holds a subset of experts.
    # To ensure all tokens can be processed by their selected experts (which might reside on different ranks),
    # we need to gather all tokens from the EP group.

    # """Gather routing/inputs, compute owned experts, then SUM back to owners."""
    # The operator registry selects the installed accelerator by default;
    # CPU inputs must explicitly use its eager kernels even on NPU hosts.
    fused = fused and hidden_states.device.type != 'cpu'

    # Use the actual local token count after the optional padding filter.
    # All three AllGathers and the final ReduceScatter reuse these lengths.
    # Note: This function must be placed in the modeling file;
    # otherwise, an "all-gather" operation will be triggered at every layer.
    set_ep_rank_seq_lens(hidden_states.shape[0], ep_group, hidden_states.device)

    # AllGather Operation: Collect tokens from all ranks in the EP group.
    # After this operation:
    #   - hidden_states: Contains ALL tokens from the global batch (N_total, D)
    #   - top_k_index: Contains expert indices for ALL tokens
    #   - top_k_weights: Contains weights for ALL tokens
    # This allows the current rank to process tokens that were originally on other ranks
    # but are destined for the local experts.

    if ep_balance_strategy is not None:
        top_k_index, num_experts_local = ep_balance_strategy.prepare_allgather(top_k_index)
        ep_balance = EPBalanceWeights(ep_balance_strategy, gate_up_weights)
    else:
        with torch.no_grad():
            top_k_index = allgather_tokens_in_ep(top_k_index, ep_group)
        num_experts_local = num_global_experts // dist.get_world_size(ep_group)
    top_k_weights = allgather_tokens_in_ep(top_k_weights, ep_group)
    hidden_states = allgather_tokens_in_ep(hidden_states, ep_group)

    ep_rank = dist.get_rank(ep_group)

    # Define the ID range of experts this rank is responsible for
    start_expert_id, end_expert_id = num_experts_local * ep_rank, num_experts_local * (ep_rank + 1)

    # Create a mask to identify tokens that should be processed by local experts
    # Tokens not belonging to local experts will be masked out (assigned to a dummy expert ID)
    mask = (top_k_index >= start_expert_id) & (top_k_index < end_expert_id)

    # Remap expert IDs: Subtract start_expert_id to get local indices (0 to num_experts_local-1)
    # Tokens not for local experts are mapped to a dummy ID (num_experts_local)
    top_k_index = torch.where(mask, top_k_index - start_expert_id, num_experts_local)

    # Count the number of tokens assigned to each local expert
    # This is used for the grouped matrix multiplication
    if ep_balance_strategy is not None:
        tokens_per_expert = ep_balance_strategy.planner.get_plan_item("num_global_sum_tokens_per_local_expert")
    else:
        tokens_per_expert = get_num_global_tokens_per_local_expert(top_k_index, num_global_experts)[:num_experts_local]

    # --- Token Permutation and Computation ---
    # Permute tokens based on expert assignment for efficient computation
    # This groups tokens belonging to the same expert together in memory.
    # Keep the complete native permutation and its inverse mapping paired.
    # Truncate with a regular slice so autograd restores a full-sized gradient
    # before entering the fused permute backward.
    permuted_local_hidden_states, reversed_local_input_permutation_mapping = permute(
        hidden_states, top_k_index.to(torch.int32), use_eager=not fused
    )

    num_local_tokens = int(tokens_per_expert.sum().item())
    permuted_local_hidden_states = permuted_local_hidden_states[:num_local_tokens]

    if ep_balance_strategy is not None:
        gate_up_weights = ep_balance.prepare_gate_up(gate_up_weights, down_weights)

    hidden_states = experts_computation(
        permuted_local_hidden_states,
        tokens_per_expert,
        gate_up_weights,
        down_weights,
        fused,
        act_fn=act_fn,
        act_limit=act_limit,
        act_alpha=act_alpha,
        ep_balance=ep_balance if ep_balance_strategy is not None else None,
        quant_config=quant_config,
        weight_owner=expert_module,
    )

    # --- Result Unpermutation and Reduce-Scatter ---
    # Unpermute the results back to the original token order
    # Dummy-expert routes were sorted to the end and skipped by the GEMMs.
    # Restore their zero outputs before unpermute so its forward and backward
    # both receive one row per entry in the complete inverse mapping.
    num_dropped_tokens = reversed_local_input_permutation_mapping.numel() - hidden_states.shape[0]
    if num_dropped_tokens:
        hidden_states = torch.cat(
            [
                hidden_states,
                hidden_states.new_zeros((num_dropped_tokens, hidden_states.shape[-1])),
            ],
            dim=0,
        )
    unpermuted_local_hidden_states = unpermute(
        hidden_states,
        reversed_local_input_permutation_mapping,
        probs=top_k_weights,
        use_eager=not fused,
    )

    # Reduce-Scatter Operation: Scatter the results back to the original ranks.
    # After unpermuting, we need to send the computed results back to the ranks
    # where the tokens originally came from.
    # This is the inverse communication pattern of the initial AllGather.
    hidden_states = reduce_scatter_tokens_in_ep(unpermuted_local_hidden_states, ep_group=ep_group)

    return hidden_states

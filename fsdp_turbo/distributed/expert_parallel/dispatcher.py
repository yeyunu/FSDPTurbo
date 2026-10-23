# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Callable, Optional
import torch

from torch.distributed._functional_collectives import all_to_all_single_autograd
from fsdp_turbo.distributed.expert_parallel.comm import gather_along_first_dim_expert_parallel
from fsdp_turbo.ops.moe import grouped_matmul, permute, unpermute, swiglu, clipped_swiglu, clamp_swiglu
from fsdp_turbo.quantization.utils import quant_grouped_gemm
from fsdp_turbo.distributed.expert_parallel.utils import get_num_global_tokens_per_local_expert
from fsdp_turbo.distributed.expert_parallel.ep_balance import EPBalanceWeights


def dispatch_preprocess(ep_group, top_k_index, num_global_experts, expert_ids_per_ep_rank):
    ep_size = torch.distributed.get_world_size(ep_group)
    ep_rank = torch.distributed.get_rank(ep_group)
    num_local_experts = num_global_experts // ep_size
    local_experts_start_id = num_local_experts * ep_rank
    local_experts_end_id = local_experts_start_id + num_local_experts

    # [B*S, K] --> [E]
    num_local_tokens_per_expert = get_num_global_tokens_per_local_expert(top_k_index, num_global_experts)
    # [E] --> [EP*E]
    num_global_tokens_per_expert, _ = gather_along_first_dim_expert_parallel(num_local_tokens_per_expert, ep_group)
    # [EP*E] --> [EP, local_E]
    num_global_tokens_per_local_expert = num_global_tokens_per_expert.reshape(ep_size, num_global_experts)[
        :, local_experts_start_id:local_experts_end_id
    ]
    # [EP, local_E] --> [local_E]
    num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
    # [E] --> [EP, local_E] --> [EP]
    input_split = (
        num_local_tokens_per_expert.reshape(ep_size, num_local_experts)
        .sum(axis=1)
        .to(torch.device("cpu"), non_blocking=True)
    )
    # [EP, local_E] --> [EP]
    output_splits = num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True)
    # [EP, local_E] --> [E*select]
    global_indices = torch.repeat_interleave(expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
    return (num_tokens_per_local_expert, global_indices), (input_split, output_splits)


def alltoall_dispatch(ep_group, hidden_states, top_k_index, indices, split_sizes, fused):
    local_indices, global_indices = indices
    input_split, output_splits = split_sizes

    hidden_states, unpermute_indices1 = permute(hidden_states, top_k_index, use_eager=not fused)
    hidden_states = all_to_all_single_autograd(hidden_states, output_splits.tolist(), input_split.tolist(), ep_group)
    hidden_states, unpermute_indices2 = permute(hidden_states, global_indices, use_eager=not fused)

    return hidden_states, (unpermute_indices1, unpermute_indices2)


def apply_activation(
    intermediate_hidden_states: torch.Tensor,
    dim: int = -1,
    fused: bool = True,
    swiglu_limit: float = 0.0,
    swiglu_alpha: Optional[float] = None,
    activation: Optional[Callable] = None,
) -> torch.Tensor:
    """Apply the activation to the first grouped-matmul output.

    When ``activation`` is provided it is called directly with the full
    ``gate_up`` tensor and is expected to return the activated tensor
    (e.g. Kimi's ``SituAndMul``). Otherwise the default SwiGLU logic is used.
    """
    if activation is not None:
        return activation(intermediate_hidden_states)
    if swiglu_alpha is not None:
        limit = 7.0 if swiglu_limit is None else swiglu_limit
        return clipped_swiglu(
            intermediate_hidden_states, dim=dim, fused=fused, swiglu_alpha=swiglu_alpha, swiglu_limit=limit
        )
    if swiglu_limit is not None and swiglu_limit > 0:
        return clamp_swiglu(intermediate_hidden_states, dim=dim, fused=fused, limit=swiglu_limit)
    return swiglu(intermediate_hidden_states, dim=dim, fused=fused)


def experts_computation(
    hidden_states,
    split_list,
    gate_up_weights,
    down_weights,
    fused,
    act_fn: Optional[Callable] = None,
    act_limit: Optional[float] = None,
    act_alpha: Optional[float] = None,
    ep_balance: Optional[EPBalanceWeights] = None,
    quant_config=None,
    weight_owner=None,
):
    if hidden_states.shape[0] == 0:
        if ep_balance is not None:
            down_weights = ep_balance.prepare_down(down_weights)
        # Avoid zero-sized fused kernels and keep zero gradients connected to
        # both expert parameters, including on ranks receiving no tokens.
        gate, up = torch.nn.functional.linear(hidden_states, gate_up_weights[0]).chunk(2, dim=-1)
        return torch.nn.functional.linear(gate * up, down_weights[0])

    def gemm_fn(x, w, weight_attr):
        if quant_config is not None:
            return quant_grouped_gemm(
                x,
                w,
                split_list,
                quant_config,
                grad_enabled=torch.is_grad_enabled(),
                weight_owner=weight_owner,
                weight_attr=weight_attr,
            )
        return grouped_matmul(x, split_list, w, use_eager=not fused)

    intermediate_hidden_states = gemm_fn(hidden_states, gate_up_weights, "gate_up_proj")
    intermediate_activations = apply_activation(
        intermediate_hidden_states,
        dim=-1,
        fused=fused,
        swiglu_limit=act_limit,
        swiglu_alpha=act_alpha,
        activation=act_fn,
    )

    if ep_balance is not None:
        down_weights = ep_balance.prepare_down(down_weights)

    hidden_states = gemm_fn(intermediate_activations, down_weights, "down_proj")
    return hidden_states


def alltoall_combine(ep_group, hidden_states, top_k_weights, unpermute_indices, split_sizes, fused):
    unpermute_indices1, unpermute_indices2 = unpermute_indices
    input_split, output_splits = split_sizes
    hidden_states = unpermute(hidden_states, unpermute_indices2, use_eager=not fused)
    hidden_states = all_to_all_single_autograd(hidden_states, input_split.tolist(), output_splits.tolist(), ep_group)
    hidden_states = unpermute(hidden_states, unpermute_indices1, top_k_weights, use_eager=not fused)
    return hidden_states


def all2all_experts_forward_fn(
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

    if ep_balance_strategy is not None:
        top_k_index, permute_indices, split_sizes = ep_balance_strategy.prepare_all2all(top_k_index)
        ep_balance = EPBalanceWeights(ep_balance_strategy, gate_up_weights)
    else:
        permute_indices, split_sizes = dispatch_preprocess(
            ep_group, top_k_index, num_global_experts, expert_ids_per_ep_rank
        )

    if fused and (permute_indices[0] == 0).all():
        fused = False

    # AllToAll dispatch --> MLP computation --> AllToAll combine
    hidden_states, unpermute_indices = alltoall_dispatch(
        ep_group, hidden_states, top_k_index, permute_indices, split_sizes, fused
    )

    if ep_balance_strategy is not None:
        gate_up_weights = ep_balance.prepare_gate_up(gate_up_weights, down_weights)

    hidden_states = experts_computation(
        hidden_states,
        permute_indices[0],
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

    hidden_states = alltoall_combine(ep_group, hidden_states, top_k_weights, unpermute_indices, split_sizes, fused)

    return hidden_states

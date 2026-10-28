# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional
import torch

from fsdp_turbo.distributed.expert_parallel.comm import gather_along_first_dim_expert_parallel
from fsdp_turbo.ops.moe import permute, unpermute, all2all_grouped_matmul, grouped_matmul_all2all
from fsdp_turbo.distributed.expert_parallel.dispatcher import all2all_experts_forward_fn, apply_activation
from fsdp_turbo.distributed.expert_parallel.utils import get_num_global_tokens_per_local_expert
from fsdp_turbo.ops.clipped_swiglu import clipped_swiglu

if TYPE_CHECKING:
    from fsdp_turbo.distributed.expert_parallel.dispatcher import FP4QuantConfig, FP8QuantConfig
    from fsdp_turbo.distributed.expert_parallel.ep_balance import EPBalanceStrategy


def mc2_experts_forward_fn(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor],
    num_global_experts: int,
    expert_ids_per_ep_rank: torch.Tensor,
    ep_group: torch.distributed.ProcessGroup,
    fused: bool,
    act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    act_limit: Optional[float] = None,
    act_alpha: Optional[float] = None,
    ep_balance_strategy: Optional[EPBalanceStrategy] = None,
    quant_config: Optional[FP4QuantConfig | FP8QuantConfig] = None,
    expert_module: Optional[torch.nn.Module] = None,
    **kwargs: object,
) -> torch.Tensor:
    """Run MC2 expert projections, retaining post-down routing and activation dtype.

    MC2 requires nonempty source buffers. Every rank uses the same gathered
    routing counts to select the AllToAll fallback if any source is empty.
    """
    gate_up_weights, down_weights = weights

    send_counts, recv_counts, has_empty_source = dispatch_preprocess(ep_group, top_k_index, num_global_experts)
    if has_empty_source:
        return all2all_experts_forward_fn(
            hidden_states,
            top_k_index,
            top_k_weights,
            weights,
            num_global_experts,
            expert_ids_per_ep_rank,
            ep_group,
            fused,
            act_fn=act_fn,
            act_limit=act_limit,
            act_alpha=act_alpha,
            ep_balance_strategy=ep_balance_strategy,
            quant_config=quant_config,
            expert_module=expert_module,
            **kwargs,
        )
    hidden_states, unpermute_indices1 = permute(hidden_states, top_k_index)
    hidden_states = all2all_grouped_matmul(hidden_states, gate_up_weights, ep_group, send_counts, recv_counts)

    if getattr(expert_module, "use_fused_clipped_swiglu", False) and act_limit is not None and act_limit > 0:
        hidden_states = clipped_swiglu(hidden_states, act_limit)
    else:
        hidden_states = apply_activation(
            hidden_states,
            dim=-1,
            fused=fused,
            swiglu_limit=act_limit,
            swiglu_alpha=act_alpha,
            activation=act_fn,
        )

    hidden_states = grouped_matmul_all2all(hidden_states, down_weights, ep_group, recv_counts, send_counts)
    hidden_states = unpermute(hidden_states, unpermute_indices1, top_k_weights)
    return hidden_states


def dispatch_preprocess(
    ep_group: torch.distributed.ProcessGroup,
    top_k_index: torch.Tensor,
    num_global_experts: int,
) -> tuple[list[int], list[int], bool]:
    """Gather expert counts and return MC2 splits plus a rank-consistent empty-source flag."""
    ep_size = torch.distributed.get_world_size(ep_group)
    ep_rank = torch.distributed.get_rank(ep_group)
    num_local_experts = num_global_experts // ep_size
    local_experts_start_id = num_local_experts * ep_rank
    local_experts_end_id = local_experts_start_id + num_local_experts

    # [B*S, K] --> [E]
    num_local_tokens_per_expert = get_num_global_tokens_per_local_expert(top_k_index, num_global_experts)
    # [E] --> [EP*E]
    num_global_tokens_per_expert, _ = gather_along_first_dim_expert_parallel(num_local_tokens_per_expert, ep_group)
    has_empty_source = bool(
        (num_global_tokens_per_expert.reshape(ep_size, num_global_experts).sum(-1) == 0).any().item()
    )
    send_counts = num_local_tokens_per_expert
    recv_counts = num_global_tokens_per_expert.reshape(ep_size, num_global_experts)[
        :, local_experts_start_id:local_experts_end_id
    ].reshape(-1)
    return send_counts.to(torch.int64).tolist(), recv_counts.to(torch.int64).tolist(), has_empty_source

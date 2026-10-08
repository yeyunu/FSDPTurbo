"""Packed DeepSeek-V4.1 routed experts compatible with FSDPTurbo EP."""

import torch
import torch.nn.functional as F
from torch import nn


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
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.intermediate_size = intermediate_size
        self.limit = swiglu_limit if swiglu_limit > 0 else None
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
        """Run the non-EP fallback over the selected experts."""
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
            gate_states, up_states = gate_up_states.chunk(2, dim=-1)
            if self.limit is not None:
                gate_states = gate_states.clamp(max=self.limit)
                up_states = up_states.clamp(min=-self.limit, max=self.limit)
            intermediate_states = self.act_fn(gate_states) * up_states
            intermediate_states = intermediate_states * topk_weights[token_indices, topk_positions, None]
            expert_output = F.linear(
                intermediate_states.to(hidden_states.dtype),
                self.down_proj[expert_index],
            )
            output = output.index_add(0, token_indices, expert_output.float())

        # Reference MoE accumulates routed experts and the shared expert in
        # FP32, then performs one final cast in MoE.forward.
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
        extension point. Token routing and combination retain the native eager
        ordering, while the two expert projections use the accelerator's
        grouped GEMM. Keeping this on the model avoids changing the generic
        dispatcher or grouped-matmul behavior used by other architectures.
        """
        from torch.distributed._functional_collectives import (
            all_to_all_single_autograd,
        )
        from torch.distributed.tensor import DTensor

        from fsdp_turbo.distributed.expert_parallel.dispatcher import (
            alltoall_combine,
            alltoall_dispatch,
            dispatch_preprocess,
        )
        from fsdp_turbo.distributed.expert_parallel.utils import (
            fixed_router_for_debug,
            normalize_expert_args,
        )
        from fsdp_turbo.ops.moe import grouped_matmul

        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        topk_indices, topk_weights = normalize_expert_args(topk_indices, topk_weights)
        if fixed_router:
            topk_indices, topk_weights = fixed_router_for_debug(
                topk_indices,
                topk_weights,
                self.num_global_experts,
            )

        indices, split_sizes = dispatch_preprocess(
            ep_group,
            topk_indices,
            self.num_global_experts,
            self.expert_ids_per_ep_rank,
        )
        local_counts, _ = indices
        input_split, output_splits = split_sizes

        hidden_states, unpermute_indices = alltoall_dispatch(
            ep_group,
            hidden_states,
            topk_indices,
            indices,
            split_sizes,
            fused=False,
        )
        source_order, expert_order = unpermute_indices
        router_weights = topk_weights.reshape(-1).index_select(0, source_order.long())
        router_weights = all_to_all_single_autograd(
            router_weights,
            output_splits.tolist(),
            input_split.tolist(),
            ep_group,
        )
        router_weights = router_weights.index_select(0, expert_order.long())
        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj

        gate_up = grouped_matmul(
            hidden_states,
            local_counts,
            gate_up_proj,
            use_eager=False,
        ).float()
        gate, up = gate_up.chunk(2, dim=-1)
        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        intermediate = self.act_fn(gate) * up
        intermediate *= router_weights.float().unsqueeze(-1)
        expert_output = grouped_matmul(
            intermediate.to(hidden_states.dtype),
            local_counts,
            down_proj,
            use_eager=False,
        )

        # The probabilities are already applied. Ones preserve the original
        # [tokens, top-k] combine shape. Converting before the shared combine
        # makes both its return AllToAll and final expert sum operate in FP32.
        combine_weights = torch.ones_like(topk_weights, dtype=torch.float32)
        output = alltoall_combine(
            ep_group,
            expert_output.float(),
            combine_weights,
            unpermute_indices,
            split_sizes,
            fused=False,
        )
        return output.view(*original_shape)


__all__ = ["DeepSeekV41Experts"]

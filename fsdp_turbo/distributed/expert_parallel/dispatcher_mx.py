# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MX-quantized Expert Parallel dispatch variant.

Holds the full EP forward path (AllToAll dispatch -> FP8 GMM -> AllToAll
combine) for MX-quantized experts. MXFP8GMM inherits this class, so its
instances expose ``ep_forward_eager`` / ``ep_forward_fused`` to the
``custom_ep_forward_*`` EP dispatcher. Mirrors the generic dispatcher in
``dispatcher.py`` but replaces ``grouped_matmul`` with ``mx_quant_group_gemm``
and delegates activation to ``_apply_activation`` so Kimi's SiTU (full-tensor)
is preserved instead of defaulting to standard GLU.
"""

import torch

from fsdp_turbo.ops.quantize import swiglu
from fsdp_turbo.utils.gmm_layout import to_gmm_3d


# no-member is disabled for this class: the attributes the methods below use
# (gate_up_proj, down_proj, num_global_experts, hidden_dim, intermediate_size,
# num_experts, act_fn, config, and optionally _is_situ / _quant_config /
# expert_ids_per_ep_rank) are defined on MXFP8GMM, the subclass that inherits
# this class. pylint analyses this class in isolation and cannot see them.
class MXExpertParallelForward:  # pylint: disable=no-member
    """MX-quantized EP forward methods inherited by MXFP8GMM."""

    def _apply_activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Apply the activation between the two expert GEMMs.

        For SiTU (Kimi), act_fn takes the full [tokens, 2*intermediate] tensor.
        For standard activations, uses swiglu (split gate/up, silu(gate)*up).
        """
        if getattr(self, '_is_situ', False) and self.act_fn is not None:
            return self.act_fn(gate_up)
        return swiglu(gate_up, dim=-1)

    def _get_local_weight_ep(self, weight_param: torch.Tensor) -> torch.Tensor:
        """Extract the local DTensor shard for use as the weight arg in EP GMM calls."""
        weight_tensor = weight_param
        if hasattr(weight_tensor, 'to_local'):
            weight_tensor = weight_tensor.to_local()
        return weight_tensor

    def _ep_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *,
        ep_group,
        fixed_router: bool,
        fused: bool,
    ) -> torch.Tensor:
        """Full EP forward: AllToAll dispatch -> FP8 GMM -> AllToAll combine.

        Mirrors PatchKimiMoeExperts._ep_forward but replaces grouped_matmul with
        mx_quant_group_gemm and delegates activation to _apply_activation so that
        Kimi's SiTU (full-tensor) is preserved instead of defaulting to standard GLU.
        """
        from fsdp_turbo.distributed.expert_parallel.dispatcher import (
            alltoall_combine,
            alltoall_dispatch,
            dispatch_preprocess,
        )
        from fsdp_turbo.distributed.expert_parallel.utils import fixed_router_for_debug
        from fsdp_turbo.quantization.mx_formats.mx_gmm import mx_quant_group_gemm

        if fixed_router:
            top_k_index, top_k_weights = fixed_router_for_debug(top_k_index, top_k_weights, self.num_global_experts)

        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        expert_ids_per_ep_rank = getattr(
            self,
            'expert_ids_per_ep_rank',
            torch.zeros(self.num_global_experts, dtype=torch.int32, device=top_k_index.device),
        )
        indices, split_sizes = dispatch_preprocess(
            ep_group, top_k_index, self.num_global_experts, expert_ids_per_ep_rank
        )
        if fused and (indices[0] == 0).any():
            fused = False

        hidden_states, unpermute_indices = alltoall_dispatch(
            ep_group, hidden_states, top_k_index, indices, split_sizes, fused
        )
        tokens_per_local_expert = indices[0]
        quant_config = getattr(self, '_quant_config', None) or self.config

        if hidden_states.numel() > 0:
            group_list = torch.cumsum(tokens_per_local_expert, dim=0).to(torch.int64)
            gate_up_w = self._get_local_weight_ep(self.gate_up_proj)
            down_w = self._get_local_weight_ep(self.down_proj)
            fc1_output = mx_quant_group_gemm(
                x=hidden_states,
                weight=gate_up_w,
                group_list=group_list,
                config=quant_config,
                grad_enabled=torch.is_grad_enabled(),
                weight_owner=self,
                weight_attr="gate_up_proj",
            )
            hidden_states = mx_quant_group_gemm(
                x=self._apply_activation(fc1_output),
                weight=down_w,
                group_list=group_list,
                config=quant_config,
                grad_enabled=torch.is_grad_enabled(),
                weight_owner=self,
                weight_attr="down_proj",
            )
        else:
            # Zero tokens dispatched to this rank: maintain autograd connectivity
            # through a zero-valued bf16 path to avoid unused-parameter errors.
            num_local_experts = getattr(self, '_num_local_experts', self.num_experts)
            gate_up_w = self._get_local_weight_ep(self.gate_up_proj)
            down_w = self._get_local_weight_ep(self.down_proj)
            gate_up_3d = to_gmm_3d(gate_up_w, num_local_experts, k=self.hidden_dim)
            down_3d = to_gmm_3d(down_w, num_local_experts, k=self.intermediate_size)
            fc1_output = hidden_states @ gate_up_3d.sum(dim=0)
            hidden_states = self._apply_activation(fc1_output) @ down_3d.sum(dim=0) * 0.0

        hidden_states = alltoall_combine(ep_group, hidden_states, top_k_weights, unpermute_indices, split_sizes, fused)
        return hidden_states.view(*original_shape)

    def ep_forward_eager(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        ep_group,
        fixed_router: bool,
    ) -> torch.Tensor:
        """EP forward with eager (non-fused) permutation."""
        return self._ep_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
            ep_group=ep_group,
            fixed_router=fixed_router,
            fused=False,
        )

    def ep_forward_fused(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        ep_group,
        fixed_router: bool,
    ) -> torch.Tensor:
        """EP forward with fused permutation (falls back to eager if any expert has zero tokens)."""
        return self._ep_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
            ep_group=ep_group,
            fixed_router=fixed_router,
            fused=True,
        )

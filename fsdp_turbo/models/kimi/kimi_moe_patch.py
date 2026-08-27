"""MindSpeed-MM FSDP MoE patch for Kimi-K3.

This module provides an alternative implementation of Kimi's sparse MoE block
that keeps the original routing logic, latent-MoE projection and shared-expert
behaviour, but reorganises routed-expert weights as 3-D tensors (the same layout
as Qwen3.5-MoE) so that it can be used with:

* non-EP and EP execution paths,
* NPU grouped GEMM / token permute / token unpermute kernels,
* the existing MindSpeed-MM FSDP2/EP tooling.

The patch is applied by ``apply_kimi_k3_moe_patch()`` which monkey-patches the
class reference in ``modeling_kimi``; the original source classes are left
untouched.
"""
# pylint: disable=relative-beyond-top-level,consider-using-from-import,unexpected-keyword-arg,no-name-in-module,ungrouped-imports

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

from fsdp_turbo.ops.moe import grouped_matmul as dispatch_grouped_matmul
from fsdp_turbo.ops.moe import permute as dispatch_permute
from fsdp_turbo.ops.moe import unpermute as dispatch_unpermute
from fsdp_turbo.ops.quantize import swiglu
from fsdp_turbo.utils.device import get_accelerator_name

from .configuration_kimi_k3 import KimiLinearConfig
from .modeling_kimi_linear import (
    ACT2FN,
    KimiMLP,
    KimiMoEGate,
    KimiRMSNorm,
    SituAndMul,
    _get_situ_activation_params,
)


from torch.nn.modules.module import _IncompatibleKeys
import logging

logger = logging.getLogger(__name__)


def grouped_matmul(inputs, weights, group_sizes, fused=True):
    """Dispatch Kimi's input-major expert weights through the shared MoE op."""
    linear_weights = weights.transpose(1, 2).contiguous()
    return dispatch_grouped_matmul(inputs, group_sizes, linear_weights, use_eager=not fused)


def permute(tokens, indices, fused=True):
    return dispatch_permute(tokens, indices, use_eager=not fused)


def unpermute(permuted_tokens, sorted_indices, probs=None, fused=True):
    return dispatch_unpermute(
        permuted_tokens,
        sorted_indices,
        probs=probs,
        use_eager=not fused,
    )


class _KimiProjectionStateDictAlias(nn.Module):
    """Virtual experts.<id>.w[123] module for HF loading."""

    def __init__(
        self,
        experts: "PatchKimiMoeExperts",
        expert_id: int,
        projection: str,
    ):
        super().__init__()

        if projection not in {"w1", "w2", "w3"}:
            raise ValueError(f"Unsupported projection: {projection!r}")

        object.__setattr__(self, "_experts", experts)
        object.__setattr__(self, "_expert_id", expert_id)
        object.__setattr__(self, "_projection", projection)

        if projection in {"w1", "w3"}:
            shape = (
                experts.intermediate_size,
                experts.hidden_size,
            )
            dtype = experts.gate_up_proj.dtype
        else:
            shape = (
                experts.hidden_size,
                experts.intermediate_size,
            )
            dtype = experts.down_proj.dtype

        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.empty(
                    shape,
                    dtype=dtype,
                    device="meta",
                ),
                requires_grad=False,
            ),
        )

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        tensor = state_dict.get("weight")

        if tensor is None:
            return _IncompatibleKeys(
                ["weight"] if strict else [],
                [],
            )

        experts = object.__getattribute__(
            self,
            "_experts",
        )

        experts.load_one_external_expert_weight(
            expert_id=object.__getattribute__(
                self,
                "_expert_id",
            ),
            projection=object.__getattribute__(
                self,
                "_projection",
            ),
            tensor=tensor,
        )

        return _IncompatibleKeys([], [])


class _KimiExpertStateDictAlias(nn.Module):
    """Virtual experts.<id> module."""

    def __init__(
        self,
        experts: "PatchKimiMoeExperts",
        expert_id: int,
    ):
        super().__init__()

        object.__setattr__(self, "_experts", experts)
        object.__setattr__(self, "_expert_id", expert_id)

    def __getattr__(self, name: str):
        if name in {"w1", "w2", "w3"}:
            return _KimiProjectionStateDictAlias(
                experts=object.__getattribute__(
                    self,
                    "_experts",
                ),
                expert_id=object.__getattribute__(
                    self,
                    "_expert_id",
                ),
                projection=name,
            )

        return super().__getattr__(name)


def _is_situ(config: KimiLinearConfig) -> bool:
    return config.hidden_act == "situ"


# class _KimiExpertStateDictAlias:
#     """Virtual ``experts.<id>`` object for FSDP's state-dict FQN validation."""

#     def __init__(self, experts: "PatchKimiMoeExperts"):
#         self._experts = experts

#     def __getattr__(self, name: str):
#         if name in {"w1", "w2", "w3"}:
#             return self
#         if name == "weight":
#             return self._experts.gate_up_proj
#         raise AttributeError(name)


class PatchKimiMoeExperts(nn.Module):
    """Routed experts stored as 3-D tensors, compatible with GMM/permute/unpermute and EP.

    Weight layout:
        * ``gate_up_proj``: ``[num_experts, hidden_size, 2 * intermediate_size]``
        * ``down_proj``   : ``[num_experts, intermediate_size, hidden_size]``

    The module transparently loads old per-expert checkpoints that store
    separate ``w1/w2/w3`` matrices (see ``_load_from_state_dict``).
    """

    def __init__(self, config: KimiLinearConfig, hidden_size: int | None = None, intermediate_size: int | None = None):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        # Keep the physical packed names until EP replaces these parameters
        # with DTensors.  That replacement is after verl's pre-FSDP snapshot;
        # once observed, every later export must use Kimi's w1/w2/w3 ABI.
        self._export_external_state_dict = False
        # FSDP-Turbo's eager/fused EP dispatcher reads ``hidden_dim``.
        # Keep the original Kimi name as the source of truth and expose the
        # dispatcher alias without changing the packed-weight layout.
        self.hidden_dim = self.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.moe_intermediate_size

        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.intermediate_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.hidden_size))

        if _is_situ(config):
            beta, linear_beta = _get_situ_activation_params(config)
            self.act_fn = SituAndMul(beta=beta, linear_beta=linear_beta)
        else:
            self.act_fn = ACT2FN[config.hidden_act]

        self.use_grouped_expert_matmul = getattr(config, "use_grouped_expert_matmul", False)
        self.fused = self.use_grouped_expert_matmul and get_accelerator_name() == "npu"

        self.reset_parameters()

        self._hf_loading_external_state_dict = False

        # 检查每层 896 × 3 个专家权重是否全部加载。
        object.__setattr__(
            self,
            "_hf_loaded_external_keys",
            set(),
        )

    def register_parameter(self, name: str, param: nn.Parameter | None) -> None:
        """Record the model-owned EP transition to DTensor expert weights."""
        super().register_parameter(name, param)
        if name in {"gate_up_proj", "down_proj"} and isinstance(param, DTensor):
            self._export_external_state_dict = True

    def reset_parameters(self) -> None:
        """Initialize expert weights like the original Kimi linear layers."""
        import torch.nn.init as init

        if self.gate_up_proj.device.type == "meta":
            return
        std = getattr(self.config, "initializer_range", 0.02)
        init.normal_(self.gate_up_proj, mean=0.0, std=std)
        init.normal_(self.down_proj, mean=0.0, std=std)

    def __getattr__(self, name: str):
        """Resolve exported ``experts.<id>.w[123].weight`` keys for FSDP2.

        ``get_model_state_dict`` validates every state-dict key by traversing
        module attributes.  The exported per-expert keys do not physically
        exist in the packed module, so expose non-registered aliases only for
        that traversal.  They never appear in ``named_parameters()`` and do
        not alter the 3-D parameter layout used for compute.
        """
        if name.isdecimal():
            try:
                expert_id = int(name)
                num_experts = object.__getattribute__(self, "num_experts")
            except AttributeError:
                pass
            else:
                if expert_id < num_experts:
                    # return _KimiExpertStateDictAlias(self)
                    return _KimiExpertStateDictAlias(
                        experts=self,
                        expert_id=expert_id,
                    )
        return super().__getattr__(name)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Pack original Kimi ``w1/w2/w3`` checkpoint entries into 3-D weights.

        The implementation updates only the source keys present in this call,
        so it also works when Transformers loads a safetensors checkpoint one
        shard at a time.
        """
        gate_key = f"{prefix}gate_up_proj"
        down_key = f"{prefix}down_proj"
        # A packed source is the FSDP-Turbo pre-wrap snapshot being restored
        # into the already parallelized model.  Capture that lifecycle event
        # here as well so non-EP FSDP runs use the external Kimi ABI too.
        loaded_packed_parameters = gate_key in state_dict or down_key in state_dict
        old_keys = [
            f"{prefix}{expert_id}.{projection}.weight"
            for expert_id in range(self.num_experts)
            for projection in ("w1", "w2", "w3")
        ]
        present_keys = [key for key in old_keys if key in state_dict]

        if present_keys:

            def local_tensor(value: torch.Tensor) -> torch.Tensor:
                return value.full_tensor() if isinstance(value, DTensor) else value

            source = local_tensor(state_dict[present_keys[0]])

            def full_packed_tensor(target: torch.Tensor, value: torch.Tensor | None) -> torch.Tensor:
                if value is not None:
                    return local_tensor(value).to(device=source.device, dtype=source.dtype).clone()
                if getattr(target, "is_meta", False):
                    return torch.empty(target.shape, dtype=source.dtype, device=source.device)
                return local_tensor(target.detach()).to(device=source.device, dtype=source.dtype).clone()

            gate_up = full_packed_tensor(self.gate_up_proj, state_dict.get(gate_key))
            down = full_packed_tensor(self.down_proj, state_dict.get(down_key))

            for expert_id in range(self.num_experts):
                w1_key = f"{prefix}{expert_id}.w1.weight"
                w2_key = f"{prefix}{expert_id}.w2.weight"
                w3_key = f"{prefix}{expert_id}.w3.weight"
                if w1_key in state_dict:
                    gate_up[expert_id, :, : self.intermediate_size].copy_(
                        local_tensor(state_dict.pop(w1_key)).to(gate_up).transpose(0, 1)
                    )
                if w2_key in state_dict:
                    down[expert_id].copy_(local_tensor(state_dict.pop(w2_key)).to(down).transpose(0, 1))
                if w3_key in state_dict:
                    gate_up[expert_id, :, self.intermediate_size :].copy_(
                        local_tensor(state_dict.pop(w3_key)).to(gate_up).transpose(0, 1)
                    )

            state_dict[gate_key] = (
                distribute_tensor(gate_up, self.gate_up_proj.device_mesh, self.gate_up_proj.placements)
                if isinstance(self.gate_up_proj, DTensor)
                else gate_up
            )
            state_dict[down_key] = (
                distribute_tensor(down, self.down_proj.device_mesh, self.down_proj.placements)
                if isinstance(self.down_proj, DTensor)
                else down
            )

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if loaded_packed_parameters:
            self._export_external_state_dict = True

    # def _save_to_state_dict(self, destination, prefix, keep_vars):
    #     """Expose standard Kimi checkpoint keys from packed live parameters.

    #     Generic FSDP/verl code iterates this state dict tensor by tensor, so
    #     vLLM receives the original expert keys and never sees 3-D names.
    #     """
    #     # The pre-FSDP snapshot must keep physical 3-D names so
    #     # ``set_model_state_dict`` can load the wrapped parameters.  EP's
    #     # model-owned ``register_parameter`` transition makes later exports
    #     # stable even when FSDP temporarily exposes ordinary tensors.
    #     if not self._export_external_state_dict:
    #         super()._save_to_state_dict(destination, prefix, keep_vars)
    #         return

    #     gate_up = self.gate_up_proj if keep_vars else self.gate_up_proj.detach()
    #     down = self.down_proj if keep_vars else self.down_proj.detach()
    #     for expert_id in range(self.num_experts):
    #         destination[f"{prefix}{expert_id}.w1.weight"] = gate_up[
    #             expert_id, :, : self.intermediate_size
    #         ].transpose(0, 1).contiguous()
    #         destination[f"{prefix}{expert_id}.w2.weight"] = down[expert_id].transpose(0, 1).contiguous()
    #         destination[f"{prefix}{expert_id}.w3.weight"] = gate_up[
    #             expert_id, :, self.intermediate_size :
    #         ].transpose(0, 1).contiguous()

    def _save_to_state_dict(
        self,
        destination,
        prefix,
        keep_vars,
    ):
        # Transformers from_pretrained 在正式加载前会读取
        # model.state_dict() 来计算 expected_keys。
        #
        # 此时必须让它看到原始 Kimi checkpoint ABI：
        # experts.<id>.w1/w2/w3.weight。
        #
        # 这里只需要 key、shape、dtype，不需要真实数据。
        if self._hf_loading_external_state_dict:
            gate_up_dtype = self.gate_up_proj.dtype
            down_dtype = self.down_proj.dtype

            for expert_id in range(self.num_experts):
                destination[f"{prefix}{expert_id}.w1.weight"] = torch.empty(
                    (
                        self.intermediate_size,
                        self.hidden_size,
                    ),
                    dtype=gate_up_dtype,
                    device="meta",
                )

                destination[f"{prefix}{expert_id}.w2.weight"] = torch.empty(
                    (
                        self.hidden_size,
                        self.intermediate_size,
                    ),
                    dtype=down_dtype,
                    device="meta",
                )

                destination[f"{prefix}{expert_id}.w3.weight"] = torch.empty(
                    (
                        self.intermediate_size,
                        self.hidden_size,
                    ),
                    dtype=gate_up_dtype,
                    device="meta",
                )

            return

        # FSDP/EP 之前的内部 snapshot：
        # 保留真实 packed 参数名。
        if not self._export_external_state_dict:
            super()._save_to_state_dict(
                destination,
                prefix,
                keep_vars,
            )
            return

        # 对外导出给 Kimi/vLLM 的路径。
        # 保留 transpose + contiguous。
        gate_up = self.gate_up_proj if keep_vars else self.gate_up_proj.detach()
        down = self.down_proj if keep_vars else self.down_proj.detach()

        for expert_id in range(self.num_experts):
            destination[f"{prefix}{expert_id}.w1.weight"] = (
                gate_up[
                    expert_id,
                    :,
                    : self.intermediate_size,
                ]
                .transpose(0, 1)
                .contiguous()
            )

            destination[f"{prefix}{expert_id}.w2.weight"] = down[expert_id].transpose(0, 1).contiguous()

            destination[f"{prefix}{expert_id}.w3.weight"] = (
                gate_up[
                    expert_id,
                    :,
                    self.intermediate_size :,
                ]
                .transpose(0, 1)
                .contiguous()
            )

    def rewrite_packed_state_dict_for_external_export(self, destination, prefix: str) -> None:
        """Replace this module's packed keys in an already collected state dict.

        FSDP2 can collect physical parameter names without invoking this
        submodule's ``_save_to_state_dict``.  The Kimi root model invokes this
        post-processing step after collection, before verl streams tensors to
        vLLM.
        """
        gate_key = f"{prefix}gate_up_proj"
        down_key = f"{prefix}down_proj"
        if gate_key not in destination or down_key not in destination:
            return

        gate_up = destination.pop(gate_key)
        down = destination.pop(down_key)
        for expert_id in range(self.num_experts):
            destination[f"{prefix}{expert_id}.w1.weight"] = (
                gate_up[expert_id, :, : self.intermediate_size].transpose(0, 1).contiguous()
            )
            destination[f"{prefix}{expert_id}.w2.weight"] = down[expert_id].transpose(0, 1).contiguous()
            destination[f"{prefix}{expert_id}.w3.weight"] = (
                gate_up[expert_id, :, self.intermediate_size :].transpose(0, 1).contiguous()
            )

    def _apply_activation(self, gate_up: torch.Tensor, fused: bool = False) -> torch.Tensor:
        """Apply the configured activation to the first linear output."""
        if _is_situ(self.config):
            return self.act_fn(gate_up)
        # Only use the fused npu_swiglu kernel when the configured activation is SiLU.
        if fused and get_accelerator_name() == "npu" and self.config.hidden_act == "silu":
            return swiglu(gate_up, dim=-1, fused=True)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Non-EP routed-expert forward.

        Args:
            hidden_states: ``[num_tokens, hidden_size]``.
            top_k_index: ``[num_tokens, top_k]`` selected expert ids.
            top_k_weights: ``[num_tokens, top_k]`` routing weights.

        Returns:
            ``[num_tokens, hidden_size]`` aggregated expert outputs.
        """
        if self.fused:
            return self._forward_fused(hidden_states, top_k_index, top_k_weights)
        return self._forward_eager(hidden_states, top_k_index, top_k_weights)

    def _forward_fused(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        permuted_hidden_states, row_ids_map = permute(hidden_states, top_k_index.to(torch.int32), fused=True)
        tokens_per_expert = torch.histc(top_k_index, bins=self.num_experts, min=0, max=self.num_experts)
        intermediate = grouped_matmul(permuted_hidden_states, self.gate_up_proj, tokens_per_expert, fused=True)
        intermediate_activations = self._apply_activation(intermediate, fused=True)
        output = grouped_matmul(intermediate_activations, self.down_proj, tokens_per_expert, fused=True)
        return unpermute(output, row_ids_map, probs=top_k_weights, fused=True)

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Transpose weights to F.linear layout: [num_experts, out_dim, in_dim].
        gate_up_proj = self.gate_up_proj.permute(0, 2, 1).contiguous()
        down_proj = self.down_proj.permute(0, 2, 1).contiguous()

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx.item()
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = F.linear(current_state, gate_up_proj[expert_idx])
            current_hidden_states = self._apply_activation(gate_up, fused=False)
            current_hidden_states = F.linear(current_hidden_states, down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None].to(
                current_hidden_states.dtype
            )
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )

        return final_hidden_states

    def ep_forward_eager(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        ep_group: Any,
        fixed_router: bool,
    ) -> torch.Tensor:
        """Run Kimi's custom EP path with eager permutation and GMM."""
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
        ep_group: Any,
        fixed_router: bool,
    ) -> torch.Tensor:
        """Run Kimi's custom EP path with fused permutation and GMM."""
        return self._ep_forward(
            hidden_states,
            top_k_index,
            top_k_weights,
            ep_group=ep_group,
            fixed_router=fixed_router,
            fused=True,
        )

    def _ep_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *,
        ep_group: Any,
        fixed_router: bool,
        fused: bool,
    ) -> torch.Tensor:
        """Kimi-specific EP compute with the complete gate/up activation.

        FSDP-Turbo's generic dispatcher is correct for standard GLU experts,
        but Kimi's SiTU also transforms the ``up`` branch.  Keep that model
        semantic here, while reusing the generic dispatcher for all-to-all
        transport and fused/eager token permutation.
        """
        from fsdp_turbo.distributed.expert_parallel.dispatcher import (
            alltoall_combine,
            alltoall_dispatch,
            dispatch_preprocess,
        )

        from fsdp_turbo.distributed.expert_parallel.utils import fixed_router_for_debug

        if fixed_router:
            top_k_index, top_k_weights = fixed_router_for_debug(top_k_index, top_k_weights, self.num_global_experts)

        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj

        expert_ids_per_ep_rank = getattr(
            self,
            "expert_ids_per_ep_rank",
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
        if hidden_states.numel():
            intermediate = grouped_matmul(hidden_states, gate_up_proj, indices[0], fused=fused)
            hidden_states = grouped_matmul(
                self._apply_activation(intermediate, fused=fused),
                down_proj,
                indices[0],
                fused=fused,
            )
        else:
            # Keep zero-token local experts connected to autograd.
            intermediate = hidden_states @ gate_up_proj.sum(dim=0)
            hidden_states = self._apply_activation(intermediate) @ down_proj.sum(dim=0) * 0.0

        return alltoall_combine(ep_group, hidden_states, top_k_weights, unpermute_indices, split_sizes, fused)

    @torch.no_grad()
    def load_one_external_expert_weight(
        self,
        expert_id: int,
        projection: str,
        tensor: torch.Tensor,
    ) -> None:
        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()

        if tensor.is_meta:
            return

        if projection in {"w1", "w3"}:
            parameter_name = "gate_up_proj"
            expected_shape = (
                self.intermediate_size,
                self.hidden_size,
            )
        elif projection == "w2":
            parameter_name = "down_proj"
            expected_shape = (
                self.hidden_size,
                self.intermediate_size,
            )
        else:
            raise ValueError(f"Unsupported projection: {projection!r}")

        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Shape mismatch: expert={expert_id}, "
                f"projection={projection}, "
                f"expected={expected_shape}, "
                f"actual={tuple(tensor.shape)}"
            )

        target = getattr(self, parameter_name)

        if isinstance(target, DTensor):
            raise RuntimeError("Expert checkpoint must be loaded before converting packed weights to DTensor.")

        if target.is_meta:
            new_target = torch.empty(
                target.shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )

            self.register_parameter(
                parameter_name,
                nn.Parameter(
                    new_target,
                    requires_grad=target.requires_grad,
                ),
            )

            target = getattr(self, parameter_name)

        source = tensor.to(
            device=target.device,
            dtype=target.dtype,
            non_blocking=True,
        )

        # 保留 NPU 所需的 contiguous 转置结果。
        source = source.transpose(0, 1).contiguous()

        if projection == "w1":
            target[
                expert_id,
                :,
                : self.intermediate_size,
            ].copy_(
                source,
                non_blocking=True,
            )

        elif projection == "w3":
            target[
                expert_id,
                :,
                self.intermediate_size :,
            ].copy_(
                source,
                non_blocking=True,
            )

        else:
            target[expert_id].copy_(
                source,
                non_blocking=True,
            )

        self._hf_loaded_external_keys.add((expert_id, projection))


def rewrite_kimi_k3_state_dict_for_external_export(model: nn.Module, destination, prefix: str = ""):
    """Convert active packed Kimi experts in a root-model state dict in place."""
    for module_name, module in model.named_modules():
        if not isinstance(module, PatchKimiMoeExperts) or not module._export_external_state_dict:
            continue
        module_prefix = f"{prefix}{module_name}." if module_name else prefix
        module.rewrite_packed_state_dict_for_external_export(destination, module_prefix)
    return destination


class PatchKimiSparseMoeBlock(nn.Module):
    """Patched Kimi sparse MoE block with 3-D expert tensors, GMM and EP support.

    Behavioural invariants compared with the original ``KimiSparseMoeBlock``:

    * Same gating logic (via ``KimiMoEGate``).
    * Same latent-MoE down/up projection and optional RMSNorm.
    * Same shared-expert residual addition.
    * Same output shape and dtype semantics.

    The only structural change is that routed experts are stored as 3-D tensors
    inside ``PatchKimiMoeExperts`` instead of a ``ModuleList`` of per-expert MLPs.
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config

        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)

        self.experts = PatchKimiMoeExperts(
            config,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )
        self.gate = KimiMoEGate(config)

        if config.num_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiMLP(config=config, intermediate_size=intermediate_size)

        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(config.hidden_size, self.moe_hidden_size, bias=False)
            self.routed_expert_up_proj = nn.Linear(self.moe_hidden_size, config.hidden_size, bias=False)
            if self.latent_moe_use_norm:
                self.routed_expert_norm = KimiRMSNorm(self.moe_hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if self.use_latent_moe:
            hidden_states = self.routed_expert_down_proj(hidden_states)

        y = self.experts(hidden_states, topk_idx, topk_weight)

        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        y = y.view(*orig_shape)

        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


# -----------------------------------------------------------------------------
# Patch application helpers
# -----------------------------------------------------------------------------


def apply_kimi_k3_moe_patch():
    """Replace Kimi sparse MoE classes with their patched counterparts.

    Must be called after ``modeling_kimi`` has been imported (so the original
    class objects exist) but before the Kimi model is instantiated.
    """
    from . import modeling_kimi_linear

    modeling_kimi_linear.KimiSparseMoeBlock = PatchKimiSparseMoeBlock

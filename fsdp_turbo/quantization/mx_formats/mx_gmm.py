# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from functools import partial
from typing import Optional

import torch

from fsdp_turbo.ops.quantize import (
    dynamic_mx_quant,
    dynamic_mx_quant_with_dual_axis,
    mx_grouped_matmul,
    swiglu,
)
from fsdp_turbo.quantization.core.post_quant_weight import PostQuantWeight
from fsdp_turbo.quantization.core.pre_quant_weight import PreQuantWeight
from fsdp_turbo.quantization.mxfp8_config import MXFP8LinearConfig
from fsdp_turbo.quantization.utils import view_as_n_dim


def weight_quant_gmm(weight: torch.Tensor, dst_type, new_shape: tuple) -> tuple:
    """Quantize weight for GMM, reshaping before quantization.

    The weight is reshaped to 3D [num_experts, in_dim, out_dim] before
    quantization so that dual-axis quantization produces correct per-expert
    scales.

    Returns:
        (weight_fwd, weight_scale_fwd, weight_bwd, weight_scale_bwd)
    """
    original_shape = weight.shape
    num_experts = weight.numel() // (new_shape[1] * new_shape[2])
    if weight.dim() == 2:
        rows_per_expert = weight.shape[0] // num_experts
        weight = weight.view(num_experts, rows_per_expert, -1)
    else:
        weight = weight.view(num_experts, weight.shape[1], weight.shape[2])
    weight = weight.transpose(1, 2).contiguous()  # [E, out, in] -> [E, in, out]
    weight_bwd, weight_scale_bwd, weight_fwd, weight_scale_fwd = dynamic_mx_quant_with_dual_axis(
        weight, dst_type=dst_type
    )
    weight_fwd = weight_fwd.reshape(original_shape)
    weight_bwd = weight_bwd.reshape(original_shape)
    return weight_fwd, weight_scale_fwd, weight_bwd, weight_scale_bwd


def mx_quant_group_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
    config: MXFP8LinearConfig,
    grad_enabled: bool = True,
    group_list_type: int = 0,
) -> torch.Tensor:
    """FP8 quantized grouped matrix multiplication.

    Args:
        x: Input tensor [total_tokens, in_features]
        weight: Weight tensor (PostQuantWeight, PreQuantWeight, nn.Parameter,
                or plain tensor).
        group_list: Cumulative token counts per group [num_groups].
        config: MXFP8 quantization configuration.
        grad_enabled: Whether gradients are enabled.
        group_list_type: 0=cumsum, 1=counts.
    """
    x_fp8, x_scale = dynamic_mx_quant(x, axis=-1, dst_type=config.get_key_dtype("inputs"))
    x_fp8 = view_as_n_dim(x_fp8, 2)

    weight_unwrapped = weight
    if isinstance(weight_unwrapped, torch.nn.Parameter):
        weight_unwrapped = weight_unwrapped.data
    if isinstance(weight_unwrapped, PreQuantWeight):
        weight_unwrapped = weight_unwrapped._tensor
    if hasattr(weight_unwrapped, 'to_local'):
        weight_unwrapped = weight_unwrapped.to_local()

    is_post_quant = isinstance(weight_unwrapped, PostQuantWeight)

    weight_2d_shape = weight_unwrapped.shape
    if weight_unwrapped.dtype == torch.float32:
        weight_unwrapped = weight_unwrapped.to(torch.bfloat16)

    K = x.shape[-1]
    num_experts = len(group_list)
    weight_transposed = weight_unwrapped.shape[0] // num_experts != K

    if is_post_quant:
        pq = weight_unwrapped
        weight_fwd = pq._weight_fwd
        weight_bwd = None
        weight_scale_fwd = pq._scale_fwd
        weight_scale_bwd = None
        if weight_fwd is not None:
            weight_fwd = weight_fwd.view(num_experts, K, -1).contiguous()
    else:
        if weight_transposed:
            weight_3d = weight_unwrapped.view(num_experts, -1, K).transpose(1, 2).contiguous()
        else:
            weight_3d = weight_unwrapped.view(num_experts, K, -1).contiguous()

        if grad_enabled:
            weight_bwd_raw, weight_scale_bwd_raw, weight_fwd, weight_scale_fwd = dynamic_mx_quant_with_dual_axis(
                weight_3d, dst_type=config.get_key_dtype("weight")
            )
            weight_bwd = weight_bwd_raw.transpose(1, 2).contiguous()
            weight_scale_bwd = weight_scale_bwd_raw.transpose(1, 2).contiguous()
        else:
            weight_fwd, weight_scale_fwd = dynamic_mx_quant(weight_3d, axis=-2, dst_type=config.get_key_dtype("weight"))
            weight_bwd = torch.empty(0, device=weight_unwrapped.device)
            weight_scale_bwd = torch.empty(0, device=weight_unwrapped.device)

    return mx_grouped_matmul(
        x,
        weight,
        x_fp8,
        x_scale,
        weight_fwd,
        weight_scale_fwd,
        weight_bwd,
        weight_scale_bwd,
        weight_2d_shape,
        weight_transposed,
        config.get_key_dtype("grads"),
        config.get_key_dtype("inputs"),
        x.dtype,
        group_list,
        group_list_type,
    )


def _mx_gmm_transform(module: torch.nn.Module, config: MXFP8LinearConfig):
    """Handler to convert an expert module to MXFP8GMM."""
    return MXFP8GMM.from_float(module, config=config)


class MXFP8GMM(torch.nn.Module):
    """FP8-quantized MoE expert module that replaces a standard expert block.

    Supports two modes:
    - forward(): Full MoE pipeline (permute + quant GMM + swiglu + quant GMM + unpermute)
      Used when EP is disabled.
    - ep_forward(): Per-expert FP8 GMM computation only (no permute/unpermute).
      Used when EP dispatcher handles token routing.
    """

    def __init__(
        self,
        config: Optional[MXFP8LinearConfig] = None,
        num_experts: Optional[int] = None,
        hidden_size: Optional[int] = None,
        moe_intermediate_size: Optional[int] = None,
        act_fn=None,
    ):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.intermediate_size = moe_intermediate_size
        self.act_fn = act_fn
        self.gate_up_proj = None
        self.down_proj = None
        self._name = None

    @property
    def num_global_experts(self):
        return self.num_experts

    @num_global_experts.setter
    def num_global_experts(self, value):
        self.num_experts = value

    @property
    def num_local_experts(self):
        return getattr(self, '_num_local_experts', self.num_experts)

    @num_local_experts.setter
    def num_local_experts(self, value):
        self._num_local_experts = value

    def forward(self, hidden_states, selected_experts=None, routing_weights=None):
        """Full MoE forward with FP8 quantization (non-EP path).

        Args:
            hidden_states: Input tokens [batch_size * seq_len, hidden_dim].
            routing_weights: Router probabilities for each token-expert pair.
            selected_experts: Expert indices for each token.

        Returns:
            Output after expert computation.
        """
        from fsdp_turbo.ops.moe import permute, unpermute

        permuted_hidden_states, row_ids_map = permute(hidden_states, selected_experts.to(torch.int32))
        tokens_per_expert = torch.histc(selected_experts.float(), bins=self.num_experts, min=0, max=self.num_experts)
        group_list = torch.cumsum(tokens_per_expert, dim=0).to(torch.int64)

        quant_config = getattr(self, '_quant_config', None) or self.config

        fc1_output = mx_quant_group_gemm(
            x=permuted_hidden_states,
            weight=self.gate_up_proj,
            group_list=group_list,
            config=quant_config,
            grad_enabled=torch.is_grad_enabled(),
        )

        fc1_activation = swiglu(fc1_output, dim=-1)

        fc2_out = mx_quant_group_gemm(
            x=fc1_activation,
            weight=self.down_proj,
            group_list=group_list,
            config=quant_config,
            grad_enabled=torch.is_grad_enabled(),
        )

        output = unpermute(fc2_out, row_ids_map, probs=routing_weights)
        return output

    def ep_forward(self, hidden_states, tokens_per_expert):
        """EP-aware FP8 expert computation (no permute/unpermute).

        Args:
            hidden_states: Tokens already permuted and dispatched via AllToAll.
            tokens_per_expert: Number of tokens for each local expert.

        Returns:
            Expert output before AllToAll combine.
        """
        if isinstance(tokens_per_expert, list):
            group_list = torch.tensor(tokens_per_expert, device=hidden_states.device, dtype=torch.int64)
        else:
            group_list = tokens_per_expert

        group_list_cumsum = torch.cumsum(group_list, dim=0).to(torch.int64)

        quant_config = getattr(self, '_quant_config', None) or self.config

        fc1_output = mx_quant_group_gemm(
            x=hidden_states,
            weight=self.gate_up_proj,
            group_list=group_list_cumsum,
            config=quant_config,
            grad_enabled=torch.is_grad_enabled(),
        )

        fc1_activation = swiglu(fc1_output, dim=-1)

        fc2_out = mx_quant_group_gemm(
            x=fc1_activation,
            weight=self.down_proj,
            group_list=group_list_cumsum,
            config=quant_config,
            grad_enabled=torch.is_grad_enabled(),
        )

        return fc2_out

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod: torch.nn.Module,
        config: Optional[MXFP8LinearConfig] = None,
        name: Optional[str] = None,
    ):
        """Convert a standard expert module to MXFP8GMM.

        Args:
            mod: Original expert module with gate_up_proj, down_proj, act_fn,
                 hidden_dim, num_experts, intermediate_size.
            config: MXFP8 quantization configuration.
            name: Module name for debugging.

        Returns:
            MXFP8GMM instance with weights wrapped in PreQuantWeight.
        """
        if config is None:
            config = MXFP8LinearConfig()

        num_experts = getattr(mod, 'num_experts', None)
        if num_experts is None:
            num_experts = len(mod) if hasattr(mod, '__len__') else 1
        hidden_dim = getattr(mod, 'hidden_dim', getattr(mod, 'hidden_size', None))
        intermediate_size = getattr(mod, 'intermediate_size', getattr(mod, 'moe_intermediate_size', None))
        act_fn = getattr(mod, 'act_fn', None)

        # Derive hidden_dim / intermediate_size from weight shape if not found
        gate_up = mod.gate_up_proj
        if not hidden_dim or not intermediate_size:
            dim0_per_expert = gate_up.shape[0] // num_experts
            dim1 = gate_up.shape[1]
            # Detect layout: [G*K, N] (dim0 < dim1) or [G*N, K] (dim0 > dim1)
            if dim0_per_expert > dim1:
                hidden_dim = hidden_dim or dim1
                intermediate_size = intermediate_size or (dim0_per_expert // 2)
            else:
                hidden_dim = hidden_dim or dim0_per_expert
                intermediate_size = intermediate_size or (dim1 // 2)

        if config.enable_fsdp_low_precision_all_gather:
            with torch.device('meta'):
                new_mod = cls(
                    config=config,
                    num_experts=num_experts,
                    hidden_size=hidden_dim,
                    moe_intermediate_size=intermediate_size,
                    act_fn=act_fn,
                )

            new_mod.gate_up_proj = mod.gate_up_proj
            new_mod.down_proj = mod.down_proj

            weight_dtype = config.get_key_dtype('weight')

            gate_up_quantizer = partial(
                weight_quant_gmm,
                dst_type=weight_dtype,
                new_shape=(-1, hidden_dim, intermediate_size * 2),
            )
            new_mod.gate_up_proj = torch.nn.Parameter(
                PreQuantWeight(
                    new_mod.gate_up_proj,
                    gate_up_quantizer,
                    config,
                    mod.gate_up_proj.dtype,
                    name=f"{name}.gate_up_proj" if name else None,
                ),
                requires_grad=new_mod.gate_up_proj.requires_grad,
            )

            down_quantizer = partial(
                weight_quant_gmm,
                dst_type=weight_dtype,
                new_shape=(-1, intermediate_size, hidden_dim),
            )
            new_mod.down_proj = torch.nn.Parameter(
                PreQuantWeight(
                    new_mod.down_proj,
                    down_quantizer,
                    config,
                    mod.down_proj.dtype,
                    name=f"{name}.down_proj" if name else None,
                ),
                requires_grad=new_mod.down_proj.requires_grad,
            )

            new_mod._name = name
            return new_mod

        # Non-low-precision path: just patch the class
        mod.__class__ = cls
        mod._quant_config = config
        mod._name = name
        return mod

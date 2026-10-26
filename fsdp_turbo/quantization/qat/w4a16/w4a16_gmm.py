from functools import partial
from typing import Optional, Union, List

import torch
from torch.distributed.tensor import DTensor

from einops import rearrange
from fsdp_turbo.quantization.qat.w4a16.w4a16_config import W4A16Config
from fsdp_turbo.quantization.cache import cached_weight_fakequant
from fsdp_turbo.ops.quantize import dynamic_mx_quant, w4a16_grouped_matmul, swiglu


@torch._dynamo.allow_in_graph
class gmm_with_hp_or_lp_weight(torch.autograd.Function):
    @classmethod
    def gmm_apply(cls, x, weight, bias, tokens_per_expert, config, grad_enabled, to_shape, ori_weight=None):
        if isinstance(tokens_per_expert, list):
            tokens_per_expert = torch.tensor(
                tokens_per_expert, device=torch.accelerator.current_accelerator(), dtype=torch.int64
            )
        group_list = torch.cumsum(tokens_per_expert, dim=0).to(torch.int64)
        return cls.apply(x, weight, bias, group_list, grad_enabled, config, to_shape, tokens_per_expert, ori_weight, 0)

    @classmethod
    def forward(
        cls,
        ctx,
        x,
        weight,
        bias,
        group_list,
        grad_enabled,
        config,
        to_shape,
        tokens_per_expert,
        ori_weight=None,
        group_list_type=0,
    ):
        def get_fakequant_weight(weight, grad_enabled, config, bias, block_size, key=None):
            if weight.dtype == torch.float32:
                weight = weight.to(torch.bfloat16)

            weight_fakequant = cached_weight_fakequant(
                weight,
                partial(dynamic_mx_quant, round_mode="rint", scale_alg=0, dst_type=torch.float4_e2m1fn_x2),
                bias,
                block_size,
                key=key,
            )

            return weight_fakequant

        block_size = 32

        if not grad_enabled:
            raise RuntimeError('Forward-only is currently not supported by QAT W4A16 recipe.')

        if isinstance(group_list, torch.Tensor):
            if group_list.device.type == "cpu":
                group_list = group_list.to(torch.accelerator.current_accelerator())
            group_list = group_list.to(torch.int64)
        else:
            group_list = torch.tensor(group_list, device=torch.accelerator.current_accelerator(), dtype=torch.int64)

        weight = weight.view(to_shape)

        if weight.shape[-1] != x.shape[-1]:
            weight = weight.transpose(1, 2).contiguous()

        weight_fwd = get_fakequant_weight(weight, grad_enabled, config, bias, block_size, key=ori_weight)

        num_experts = weight_fwd.shape[0]
        weights_for_matmul = [weight_fwd[i].T.contiguous() for i in range(num_experts)]

        output = w4a16_grouped_matmul(
            x,
            weights_for_matmul,
            bias=bias,
            group_list=group_list,
            group_type=0,
            output_dtype=x.dtype,
            group_list_type=group_list_type,
            split_item=2,
        )

        ctx.save_for_backward(x, weight_fwd, group_list)
        ctx.config = config
        ctx.group_list_type = group_list_type
        ctx.bias = bias
        ctx.to_shape = to_shape
        ctx.weight_ori = weight
        return output[0]

    @classmethod
    def backward(cls, ctx, grad_outputs):
        x, weight_bwd, group_list = ctx.saved_tensors
        group_list_type = ctx.group_list_type

        if isinstance(weight_bwd, DTensor):
            weight_bwd = weight_bwd.to_local()

        grad_bias = None
        if ctx.bias is not None:
            grad_bias = grad_outputs.reshape(-1, grad_outputs.shape[-1]).sum(dim=0)

        grad_x = w4a16_grouped_matmul(
            grad_outputs,
            [weight_bwd],
            group_list=group_list,
            group_type=0,
            output_dtype=grad_outputs.dtype,
            group_list_type=group_list_type,
            split_item=2,
        )[0]

        grad_weights = w4a16_grouped_matmul(
            x.t(),
            [grad_outputs],
            group_list=group_list,
            group_type=2,
            output_dtype=x.dtype,
            group_list_type=group_list_type,
            split_item=2,
        )[0]

        grad_weights = rearrange(grad_weights, "n h f -> n f h")
        return grad_x, grad_weights, grad_bias, None, None, None, None, None, None, None


def mx_quant_group_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    tokens_per_expert: Union[List[int], torch.Tensor] = None,
    grad_enabled: bool = True,
    config: W4A16Config = None,
    to_shape: list = None,
    ori_weight: torch.Tensor = None,
) -> torch.Tensor:
    return gmm_with_hp_or_lp_weight.gmm_apply(
        x=x,
        weight=weight,
        bias=bias,
        tokens_per_expert=tokens_per_expert,
        config=config,
        grad_enabled=grad_enabled,
        to_shape=to_shape,
        ori_weight=ori_weight,
    )


class W4A16GMM(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = kwargs.pop("config", None)
        self.num_experts = kwargs.pop("num_experts", None)
        self.hidden_size = kwargs.pop("hidden_size", None)
        self.intermediate_size = kwargs.pop("moe_intermediate_size", None)
        self.act_fn = kwargs.pop("act_fn", None)

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
        from fsdp_turbo.ops.moe import permute, unpermute

        permuted_hidden_states, row_ids_map = permute(hidden_states, selected_experts.to(torch.int32))
        tokens_per_expert = torch.histc(selected_experts.float(), bins=self.num_experts, min=0, max=self.num_experts)
        fc1_output = mx_quant_group_gemm(
            x=permuted_hidden_states,
            weight=self.gate_up_proj,
            bias=None,
            tokens_per_expert=tokens_per_expert,
            config=self.config,
            grad_enabled=torch.is_grad_enabled(),
            to_shape=[self.num_experts, -1, self.hidden_size],
            ori_weight=self.gate_up_proj,
        )

        fc1_activation = swiglu(fc1_output, dim=-1)

        fc2_out = mx_quant_group_gemm(
            x=fc1_activation,
            weight=self.down_proj,
            bias=None,
            tokens_per_expert=tokens_per_expert,
            config=self.config,
            grad_enabled=torch.is_grad_enabled(),
            to_shape=[self.num_experts, self.hidden_size, -1],
            ori_weight=self.down_proj,
        )

        routing_weights = routing_weights.to(torch.bfloat16)
        output = unpermute(fc2_out, row_ids_map, probs=routing_weights)
        return output

    def ep_forward(self, hidden_states, tokens_per_expert):
        gate_up_proj = self.gate_up_proj.to_local() if hasattr(self.gate_up_proj, 'to_local') else self.gate_up_proj
        down_proj = self.down_proj.to_local() if hasattr(self.down_proj, 'to_local') else self.down_proj

        fc1_output = mx_quant_group_gemm(
            x=hidden_states,
            weight=gate_up_proj,
            bias=None,
            tokens_per_expert=tokens_per_expert,
            config=self.config,
            grad_enabled=torch.is_grad_enabled(),
            to_shape=[self.num_local_experts, self.hidden_size, -1],
            ori_weight=self.gate_up_proj,
        )

        fc1_activation = swiglu(fc1_output, dim=-1)

        fc2_out = mx_quant_group_gemm(
            x=fc1_activation,
            weight=down_proj,
            bias=None,
            tokens_per_expert=tokens_per_expert,
            config=self.config,
            grad_enabled=torch.is_grad_enabled(),
            to_shape=[self.num_local_experts, -1, self.hidden_size],
            ori_weight=self.down_proj,
        )
        return fc2_out

    @classmethod
    @torch.no_grad()
    def from_float(cls, mod: torch.nn.Linear, config: Optional[W4A16Config] = None, name: Optional[str] = None):
        if config is None:
            config = W4A16Config()

        hidden_size = getattr(mod, 'hidden_dim', getattr(mod, 'hidden_size', None))
        num_experts = getattr(mod, 'num_experts', None)

        mod.__class__ = cls
        mod.config = config
        mod.num_experts = num_experts
        mod.hidden_size = hidden_size
        mod._name = name
        return mod


def _w4a16_gmm_transform(module, config):
    return W4A16GMM.from_float(module, config=config)

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import torch

from fsdp_turbo.ops.moe import grouped_matmul
from fsdp_turbo.quantization.qat.w8a16.w8a16_linear import W8A16FakeQuantization
from fsdp_turbo.quantization.qat.w8a16.w8a16_config import W8A16Config
from fsdp_turbo.quantization.utils import resolve_expert_weights


def reshape_for_gmm(w: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Reshape an expert weight to [num_experts, out_dim, in_dim] for GMM.

    Accepts both the 2D stacked layout [num_experts * out_dim, in_dim] and the
    pre-split 3D layout [num_experts, out_dim, in_dim] (e.g. Qwen3's
    ``gate_up_proj``); matches ``grouped_matmul``'s weight convention.
    """
    if w.dim() == 2:
        return w.view(num_experts, -1, w.shape[1])
    return w.contiguous()


def w8a16_experts_computation(hidden_states, split_list, gate_up_weights, down_weights, act_fn, act_limit=None):
    """Fake-quantized grouped matmul for the W8A16 MoE path.

    Used by ``W8A16GMM.forward`` and the EP path via ``quant_grouped_gemm``
    (the dispatcher's quant_config branch).
    """
    num_experts = split_list.size(0) if isinstance(split_list, torch.Tensor) else len(split_list)

    gate_up_quantized = W8A16FakeQuantization.apply(gate_up_weights)
    gate_up_3d = reshape_for_gmm(gate_up_quantized, num_experts)
    fc1 = grouped_matmul(hidden_states, split_list, gate_up_3d)

    gate, up = fc1.chunk(2, dim=-1)
    if act_limit is not None:
        gate = gate.clamp(max=act_limit)
        up = up.clamp(min=-act_limit, max=act_limit)
    act = (torch.nn.functional.silu if act_fn is None else act_fn)(gate) * up

    down_quantized = W8A16FakeQuantization.apply(down_weights)
    down_3d = reshape_for_gmm(down_quantized, num_experts)
    return grouped_matmul(act, split_list, down_3d)


class W8A16GMM(torch.nn.Module):
    def __init__(self, config=None, num_experts=None, hidden_size=None, moe_intermediate_size=None, act_fn=None):
        super().__init__()
        self.config = config or W8A16Config()
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.intermediate_size = moe_intermediate_size
        self.act_fn = act_fn

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

    def forward(self, hidden_states, routing_weights=None, selected_experts=None):
        from fsdp_turbo.ops.moe import permute, unpermute

        permuted_hidden_states, row_ids_map = permute(hidden_states, selected_experts.to(torch.int32))
        tokens_per_expert = torch.histc(selected_experts.float(), bins=self.num_experts, min=0, max=self.num_experts)

        gate_up, down = resolve_expert_weights(self)

        out = w8a16_experts_computation(permuted_hidden_states, tokens_per_expert, gate_up, down, self.act_fn)
        return unpermute(out, row_ids_map, probs=routing_weights)

    @classmethod
    @torch.no_grad()
    def from_float(cls, mod, config=None):
        if config is None:
            config = W8A16Config()

        num_experts = getattr(mod, 'num_experts', None)
        if num_experts is None:
            raise ValueError(f"Module {type(mod).__name__} missing 'num_experts' attribute")
        hidden_dim = getattr(mod, 'hidden_dim', getattr(mod, 'hidden_size', None))
        intermediate_size = getattr(mod, 'intermediate_size', getattr(mod, 'moe_intermediate_size', None))

        gate_up = mod.gate_up_proj
        if isinstance(gate_up, torch.nn.Module):
            gate_up = gate_up.weight
        if hidden_dim is None or intermediate_size is None:
            # gate_up 布局：2D 堆叠 [E * out_dim, in_dim] 或 3D [E, out_dim, in_dim]
            if gate_up.dim() == 2:
                out_dim = gate_up.shape[0] // num_experts
                in_dim = gate_up.shape[1]
            else:
                out_dim = gate_up.shape[1]
                in_dim = gate_up.shape[2]
            hidden_dim = hidden_dim or in_dim
            intermediate_size = intermediate_size or (out_dim // 2)

        mod.__class__ = cls
        mod.config = config
        mod.num_experts = num_experts
        mod.hidden_dim = hidden_dim
        mod.intermediate_size = intermediate_size
        mod.act_fn = getattr(mod, 'act_fn', None)
        return mod


def _w8a16_gmm_transform(module, config):
    return W8A16GMM.from_float(module, config=config)

from typing import Optional
from functools import partial

import torch

from fsdp_turbo.quantization.qat.w4a16.w4a16_config import W4A16Config
from fsdp_turbo.quantization.cache import cached_weight_fakequant
from fsdp_turbo.quantization.utils import register_quantize_module_handler
from fsdp_turbo.ops.quantize import dynamic_mx_quant


@torch._dynamo.allow_in_graph
class matmul_with_hp_or_lp_weight(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        config: W4A16Config,
        grad_enabled: bool,
        bias: torch.Tensor = None,
        name: str = None,
    ):
        block_size = 32

        if not grad_enabled:
            raise RuntimeError('Forward-only is currently not supported by the QAT W4A16 recipe.')

        weight_fwd = cached_weight_fakequant(
            weight,
            partial(dynamic_mx_quant, dst_type=torch.float4_e2m1fn_x2),
            bias,
            block_size,
        )

        output = torch.matmul(x, weight_fwd.t())

        ctx.save_for_backward(x, weight_fwd)
        ctx.name = name
        ctx.bias = bias

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight_bwd = ctx.saved_tensors

        grad_output_orig_shape = grad_output.shape
        grad_output_reshaped = grad_output.reshape(-1, grad_output_orig_shape[-1])

        grad_bias = None
        if ctx.bias is not None:
            grad_bias = grad_output_reshaped.sum(dim=0)

        grad_x = torch.matmul(grad_output, weight_bwd)
        x_reshaped = x.reshape(-1, x.shape[-1])
        grad_weight = torch.matmul(grad_output_reshaped.t(), x_reshaped)

        return grad_x, grad_weight, None, None, grad_bias, None


def mx_quant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    config: W4A16Config,
    grad_enabled: bool = True,
    bias: Optional[torch.Tensor] = None,
    name: Optional[str] = None,
) -> torch.Tensor:
    return matmul_with_hp_or_lp_weight.apply(x, weight, config, grad_enabled, bias, name)


@torch._dynamo.allow_in_graph
class W4A16Linear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.config: Optional[W4A16Config] = None
        self._tp_mesh = None
        self._tp_output_placements = None

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod: torch.nn.Module,
        config: Optional[W4A16Config] = None,
        name: Optional[str] = None,
    ) -> "W4A16Linear":
        if config is None:
            config = W4A16Config()

        mod.__class__ = cls
        mod.config = config
        mod._name = name
        return mod

    def forward(self, x) -> torch.Tensor:
        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_dtype())

        output = mx_quant_linear(
            x,
            self.weight,
            self.config,
            torch.is_grad_enabled(),
            None,
            self._name,
        )

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    def extra_repr(self):
        if self.config is None:
            return super().extra_repr()
        return f"{super().extra_repr()}, {str(self.config)}"


@register_quantize_module_handler(W4A16Config)
def _w4a16_linear_transform(module: torch.nn.Module, config: W4A16Config) -> W4A16Linear:
    return W4A16Linear.from_float(module, config=config)

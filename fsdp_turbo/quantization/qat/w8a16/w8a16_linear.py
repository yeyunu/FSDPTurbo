# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from typing import Optional

import torch
from torch import nn
from torch.autograd import Function
from torch.distributed.tensor import DTensor

from fsdp_turbo.ops.quantize import dynamic_mx_quant
from fsdp_turbo.quantization.qat.w8a16.w8a16_config import W8A16Config
from fsdp_turbo.quantization.utils import register_quantize_module_handler


def w8a16_fake_quant(tensor: torch.Tensor, qdim: int = -1) -> torch.Tensor:
    """FP8 e4m3 + per-32-block shared-exponent fake quantization (MX format).

    Uses the ``dynamic_mx_quant`` op so QAT numerics match the MX inference
    path bit-for-bit. Only qdim=-1 (last dim) is supported.
    """
    orig_dim_size = tensor.shape[qdim]
    num_blocks = (orig_dim_size + 31) // 32
    pad_size = (32 - orig_dim_size % 32) % 32
    if pad_size != 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_size))

    data_fp8, scale_bytes = dynamic_mx_quant(
        tensor, axis=qdim, dst_type=torch.float8_e4m3fn, scale_alg=1, block_size=32
    )

    # scale_bytes: uint8 e8m0, interleaved pack [.., nb_padded/2, 2]; unpack
    # and drop the dummy block (odd num_blocks), broadcast over [.., nb, 32].
    scale_bf16 = torch.pow(2.0, scale_bytes.to(torch.float32) - 127.0).to(tensor.dtype)
    scale_bf16 = scale_bf16.reshape(*scale_bf16.shape[:-2], -1)[..., :num_blocks]
    scale_blocks = scale_bf16.view(*scale_bf16.shape[:-1], num_blocks, 1)
    data_blocks = data_fp8.to(tensor.dtype).view(*data_fp8.shape[:-1], num_blocks, 32)
    dequantized = (data_blocks * scale_blocks).flatten(-2, -1)

    if pad_size != 0:
        dequantized = dequantized.narrow(qdim, 0, orig_dim_size)
    return dequantized


@torch._dynamo.allow_in_graph
class W8A16FakeQuantization(Function):
    @staticmethod
    def forward(ctx, high_precision_tensor):
        return w8a16_fake_quant(high_precision_tensor)

    @staticmethod
    def backward(ctx, output_grad):
        return (output_grad,)


class W8A16Linear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.config: Optional[W8A16Config] = None
        self._tp_mesh = None
        self._tp_output_placements = None

    @classmethod
    @torch.no_grad()
    def from_float(cls, mod: nn.Module, config: Optional[W8A16Config] = None) -> "W8A16Linear":
        if config is None:
            config = W8A16Config()
        mod.__class__ = cls
        mod.config = config
        mod._tp_mesh = None
        mod._tp_output_placements = None
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        x = x.to_local() if hasattr(x, 'to_local') else x
        w = w.to_local() if hasattr(w, 'to_local') else w

        w_q = W8A16FakeQuantization.apply(w).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None

        y = nn.functional.linear(x, w_q, bias)

        if self._tp_mesh is not None:
            y = DTensor.from_local(y, self._tp_mesh, self._tp_output_placements)
        return y


@register_quantize_module_handler(W8A16Config)
def _w8a16_linear_transform(module: torch.nn.Module, config: W8A16Config) -> W8A16Linear:
    return W8A16Linear.from_float(module, config=config)

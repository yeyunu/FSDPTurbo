# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


@register_op('rotary_mul', 'npu')
def rotary_mul_npu(x, cos, sin):
    return torch_npu.npu_rotary_mul(x, cos, sin)

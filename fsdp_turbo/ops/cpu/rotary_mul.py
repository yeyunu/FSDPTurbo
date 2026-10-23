# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from fsdp_turbo.ops.registry import register_op


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@register_op('rotary_mul', 'cpu')
def rotary_mul_cpu(x, cos, sin):
    return (x * cos) + (_rotate_half(x) * sin)

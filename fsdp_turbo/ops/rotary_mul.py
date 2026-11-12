# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from fsdp_turbo.ops import dispatch_op


def rotary_mul(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding multiply with automatic device dispatch.

    Computes ``x * cos + rotate_half(x) * sin`` using the fused NPU
    ``npu_rotary_mul`` op when available, or the PyTorch eager fallback
    otherwise.

    Args:
        x: Query or key tensor.
        cos: Cosine component of rotary embedding.
        sin: Sine component of rotary embedding.

    Returns:
        Rotated tensor.
    """
    return dispatch_op('rotary_mul', x, cos, sin, device_type=x.device.type)

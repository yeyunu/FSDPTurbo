# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch

from fsdp_turbo.ops.registry import dispatch_op


def fused_situ_glu(
    x: torch.Tensor,
    *,
    dim: int = -1,
    beta: float = 1.0,
    linear_beta: float = 0.0,
    activate_left: bool = True,
    use_eager: bool = False,
) -> torch.Tensor:
    """Apply the SiTU-GLU activation.

    When ``use_eager`` is ``False`` (default), dispatches to the fused
    operator registered for the current accelerator (e.g.
    ``cann_ops_nn.situ_glu`` on NPU); the per-backend implementation falls
    back to eager when the fused operator is not available. When
    ``use_eager`` is ``True``, forces the eager PyTorch implementation.

    Args:
        x: Input tensor whose dimension ``dim`` is split into gate and up halves.
        dim: Dimension along which to split. Defaults to ``-1``.
        beta: Scaling factor for the SiTU activation. Defaults to ``1.0``.
        linear_beta: If non-zero, applies ``linear_beta * tanh(up / linear_beta)``
            to the up half. Defaults to ``0.0`` (no bounded activation).
        activate_left: Whether the gate half is the left slice. Defaults to ``True``.
        use_eager: If ``False``, use the fused implementation registered for the
            current device (falling back to eager when unavailable). If ``True``,
            force the eager implementation. Defaults to ``False``.

    Returns:
        Tensor of the same shape as ``x`` with the last dimension halved.
    """
    device_type = 'cpu' if use_eager else None
    return dispatch_op(
        'situ_glu',
        x,
        dim,
        beta,
        linear_beta,
        activate_left,
        device_type=device_type,
    )


__all__ = ['fused_situ_glu']

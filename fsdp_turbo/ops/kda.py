# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch

from fsdp_turbo.ops.registry import dispatch_op


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *args: object,
    disable_recompute: bool | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Apply chunked Kimi Delta Attention using the active device backend.

    When disable_recompute is None, use the backend default. A boolean
    controls reconstruction of KDA intermediates during the kernel backward.
    """
    if disable_recompute is not None:
        kwargs["disable_recompute"] = disable_recompute
    from fsdp_turbo.ops.npu.kimi_kda_alignment import enabled, training_kda
    if enabled() and q.device.type == 'npu':
        def native(q, k, v, g, beta, **options):
            return dispatch_op('chunk_kda', q, k, v, g, beta, device_type=q.device.type, **options)
        return training_kda(native, q, k, v, g, beta, args, kwargs)
    return dispatch_op('chunk_kda', q, k, v, g, beta, *args, device_type=q.device.type, **kwargs)


__all__ = ['chunk_kda']

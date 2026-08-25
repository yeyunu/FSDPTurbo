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
    *args,
    **kwargs,
):
    """Apply chunked Kimi Delta Attention using the active device backend."""
    return dispatch_op('chunk_kda', q, k, v, g, beta, *args, **kwargs)


__all__ = ['chunk_kda']

# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

"""NPU registration for the Engram gated-memory operator.

Selects the vendored PyPTO-Pro kernels when ``pypto_pro`` is importable,
otherwise falls back to the eager CPU implementation (mirrors chunk_kda).
The packed-wkv block glue (``EngramWkvFunction``) is device-neutral and lives
in the backend-free semantic layer :mod:`fsdp_turbo.ops.engram`; it is
registered for NPU here so its internal dispatch picks the fused kernels
without a fallback warning.
"""

import logging

from fsdp_turbo.ops.engram import EngramWkvFunction
from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)

try:
    from fsdp_turbo.ops.npu.engram_pypto import (
        engram_backward_wrapper as _engram_backward_impl,
        engram_forward_wrapper as _engram_forward_impl,
    )
except ImportError:
    from fsdp_turbo.ops.cpu.engram import (
        engram_backward_cpu as _engram_backward_impl,
        engram_forward_cpu as _engram_forward_impl,
    )

    print_rank(
        logger.warning,
        "pypto_pro Engram kernels are unavailable; falling back to the eager implementation.",
    )


@register_op("engram_forward", "npu")
def engram_forward_npu(
    hidden_states,
    embeddings,
    key_proj_weights,
    value_proj_weights,
    key_gamma,
    query_gamma,
    clamp_value: float = 1e-6,
    eps: float = 1e-6,
):
    return _engram_forward_impl(
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
        clamp_value=clamp_value,
        eps=eps,
    )


@register_op("engram_backward", "npu")
def engram_backward_npu(
    grad_output,
    hidden_states,
    embeddings,
    key_proj_weights,
    value_proj_weights,
    key_gamma,
    query_gamma,
    scores,
    gates,
    keys,
    value,
    clamp_value: float = 1e-6,
    eps: float = 1e-6,
    pretransposed_weights: bool = False,
):
    return _engram_backward_impl(
        grad_output,
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
        scores,
        gates,
        keys,
        value,
        clamp_value=clamp_value,
        eps=eps,
        pretransposed_weights=pretransposed_weights,
    )


@register_op("engram_wkv", "npu")
def engram_wkv_npu(
    hidden_states,
    memory,
    wkv_weight,
    key_gamma,
    query_gamma,
    clamp_value: float = 1e-6,
    eps: float = 1e-6,
    token_mask=None,
):
    return EngramWkvFunction.apply(
        hidden_states,
        memory,
        wkv_weight,
        key_gamma,
        query_gamma,
        eps,
        clamp_value,
        token_mask,
    )


__all__ = [
    "engram_forward_npu",
    "engram_backward_npu",
    "engram_wkv_npu",
]

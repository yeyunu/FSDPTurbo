# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from typing import Any, Tuple

from fsdp_turbo.ops.registry import dispatch_op


def chunk_gdn(*args: Any, **kwargs: Any) -> Tuple[Any, ...]:
    """Compute chunked gated delta rule via the device-dispatched operator.

    Thin wrapper over :func:`fsdp_turbo.ops.registry.dispatch_op` for the
    ``chunk_gated_delta_rule`` operator. On NPU the AscendC / FLA backend is
    used; on CUDA the FLA triton backend is used; on other devices the call
    falls back to the registered CPU implementation if one exists.
    """
    return dispatch_op('chunk_gated_delta_rule', *args, **kwargs)


def fused_recurrent_gdn(*args: Any, **kwargs: Any) -> Tuple[Any, ...]:
    """Compute fused recurrent gated delta rule via the device-dispatched operator.

    Thin wrapper over :func:`fsdp_turbo.ops.registry.dispatch_op` for the
    ``fused_recurrent_gated_delta_rule`` operator. Used during single-token
    cached decode where the recurrent path is preferred over the chunk path.
    """
    return dispatch_op('fused_recurrent_gated_delta_rule', *args, **kwargs)


__all__ = [
    "chunk_gdn",
    "fused_recurrent_gdn",
]

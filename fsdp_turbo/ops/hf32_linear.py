# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Backend-dispatched entry point for HF32 head projection."""

from fsdp_turbo.ops.registry import dispatch_op


def hf32_linear(*args, **kwargs):
    return dispatch_op("hf32_linear", *args, **kwargs)


__all__ = ["hf32_linear"]

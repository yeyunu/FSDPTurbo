# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

from fsdp_turbo.ops.registry import dispatch_op


def attn_res(prefix_sum, block_residual, proj_weight, norm_weight, variance_epsilon):
    """Apply attention residuals using the active device backend."""
    return dispatch_op('attn_res', prefix_sum, block_residual, proj_weight, norm_weight, variance_epsilon)


__all__ = ['attn_res']

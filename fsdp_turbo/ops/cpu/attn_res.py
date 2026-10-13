# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch

from fsdp_turbo.ops.registry import register_op


@register_op('attn_res', 'cpu')
def attn_res_cpu(prefix_sum, block_residual, proj_weight, norm_weight, variance_epsilon):
    """Portable attention-residual implementation."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_float = values.float()
    variance = values_float.pow(2).mean(-1, keepdim=True)
    normalized = values_float * torch.rsqrt(variance + variance_epsilon)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scores = (normalized * score_weight).sum(-1)
    probabilities = scores.softmax(-1).unsqueeze(1)
    output = torch.matmul(probabilities, values_float).squeeze(1)
    return output.to(values.dtype)


__all__ = ['attn_res_cpu']

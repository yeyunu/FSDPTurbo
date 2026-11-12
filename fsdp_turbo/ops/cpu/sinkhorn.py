# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2026, The HuggingFace Inc. Team. All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION AFFILIATES. All rights reserved.

import torch

from fsdp_turbo.ops.registry import register_op


@register_op('hc_split_sinkhorn', 'cpu')
def hc_split_sinkhorn_cpu(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mHC coefficients on the input device with the fused dtype contract.

    FP16/BF16 inputs compute in FP32; FP32 and FP64 inputs retain their precision.
    All coefficients return in ``mixes.dtype``. Autograd preserves each input's
    gradient dtype through the differentiable conversions.
    """
    output_dtype = mixes.dtype
    compute_dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype
    mixes = mixes.to(compute_dtype)
    hc_scale = hc_scale.to(compute_dtype)
    hc_base = hc_base.to(compute_dtype)
    hc = hc_mult
    pre_w, post_w, comb_w = mixes.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = hc_base.split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = hc_scale.unbind(0)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)

    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre.to(output_dtype), post.to(output_dtype), comb.to(output_dtype)

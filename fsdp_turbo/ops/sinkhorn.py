# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from fsdp_turbo.ops import dispatch_op


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    use_eager: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    HC-Split Sinkhorn algorithm for MoE routing with automatic device dispatch.

    Args:
        mixes: Input tensor [batch_size, seq_len, (2+hc_mult)*hc_mult]
        hc_scale: Scale parameters [3] (pre, post, comb)
        hc_base: Base parameters [(2+hc_mult)*hc_mult]
        hc_mult: HC multiplier dimension (default=4)
        sinkhorn_iters: Sinkhorn normalization iterations (default=20)
        eps: Epsilon for numerical stability (default=1e-6)
        use_eager: If False, dispatch using the input tensor's device.
                   If True, use the PyTorch reference on that same device.

    Returns:
        tuple: (pre, post, comb), all on mixes.device and in mixes.dtype.
            FP16/BF16 inputs compute in FP32 in both eager and fused paths.
            - pre: [batch_size, seq_len, hc_mult]
            - post: [batch_size, seq_len, hc_mult]
            - comb: [batch_size, seq_len, hc_mult, hc_mult]
    """
    device_type = 'cpu' if use_eager else mixes.device.type
    return dispatch_op(
        'hc_split_sinkhorn', mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps, device_type=device_type
    )

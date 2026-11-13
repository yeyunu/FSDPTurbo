# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

from importlib import import_module
import torch

from fsdp_turbo.ops.registry import dispatch_op


def attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    variance_epsilon: float,
    *,
    device_type: str | None = None,
    implementation: str = "auto",
) -> torch.Tensor:
    """Apply attention residuals with automatic, eager or explicit AscendC dispatch.

    Eager runs PyTorch operations on the input device. On NPU, selecting
    AscendC requires the fused implementation. Other devices run eager.
    """
    args = (prefix_sum, block_residual, proj_weight, norm_weight, variance_epsilon)
    if implementation == "ascendc" and prefix_sum.device.type == "npu":
        try:
            npu_attn_res = import_module("fsdp_turbo.ops.npu.attn_res")
        except ImportError as exc:
            raise RuntimeError("The NPU attention residual operator is unavailable.") from exc
        if npu_attn_res.ATTN_RES_BACKEND != "ascendc":
            raise RuntimeError("AscendC attention residuals require cann_ops_transformer.block_attention_residuals.")
        device_type = "npu"
    elif implementation in ("eager", "ascendc"):
        device_type = "cpu"
    elif implementation != "auto":
        raise ValueError(f"Unsupported attention residual implementation: {implementation}")

    return dispatch_op("attn_res", *args, device_type=device_type)


__all__ = ['attn_res']

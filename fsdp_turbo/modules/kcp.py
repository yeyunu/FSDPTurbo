# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.
"""Operators for Kimi context parallelism (KCP)."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from triton_ascend_kernels.attention.fla.kda.cp import FLACPContext


@lru_cache(maxsize=1)
def _ensure_triton_driver() -> None:
    """Select the Triton driver that matches the active PyTorch accelerator."""
    from triton.backends import backends
    from triton.runtime import driver

    device_type = torch.accelerator.current_accelerator().type
    if device_type == "npu":
        backend_name = "ascend"
    elif device_type == "cuda":
        backend_name = "amd" if torch.version.hip is not None else "nvidia"
    else:
        raise RuntimeError(f"Triton is not supported on accelerator {device_type!r}.")

    triton_backend = backends.get(backend_name)
    if triton_backend is None:
        raise RuntimeError(f"Triton backend {backend_name!r} is not registered.")

    # Do not access driver.active before this call: doing so triggers
    # automatic detection and may see both NPUDriver and CudaDriver after
    # transfer_to_npu aliases torch.cuda to torch.npu.
    driver.set_active(triton_backend.driver())


def _normalize_output(result, operator_name: str) -> torch.Tensor:
    output = result[0] if isinstance(result, tuple) else result
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"{operator_name} must return a tensor or a tuple beginning with one.")
    return output


def build_kcp_context(
    global_cu_seqlens: torch.Tensor,
    group,
    conv1d_kernel_size: int,
) -> FLACPContext:
    """Validate global sequence metadata and build a KDA CP context."""

    if global_cu_seqlens.dim() != 1 or global_cu_seqlens.numel() < 2:
        raise ValueError("global_cu_seqlens must be one-dimensional with at least two elements.")
    if global_cu_seqlens[0].item() != 0:
        raise ValueError("global_cu_seqlens must start at zero.")
    if not torch.all(global_cu_seqlens[1:] > global_cu_seqlens[:-1]).item():
        raise ValueError("global_cu_seqlens boundaries must be strictly increasing.")

    from triton_ascend_kernels.attention.fla.kda.cp import build_cp_context

    return build_cp_context(
        cu_seqlens=global_cu_seqlens,
        group=group,
        conv1d_kernel_size=conv1d_kernel_size,
    )


def kcp_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    context: FLACPContext,
) -> torch.Tensor:
    """Run FLA KCP-aware causal depthwise convolution."""

    from fla.modules.conv.causal_conv1d import causal_conv1d

    result = causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        activation=activation,
        initial_state=None,
        output_final_state=False,
        cp_context=context,
    )
    return _normalize_output(result, "kcp_causal_conv1d")


def kcp_chunk_kda(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    disable_recompute: bool,
    context: FLACPContext,
) -> torch.Tensor:
    """Run Triton Ascend KCP-aware chunked Kimi Delta Attention."""

    if q.dim() == 4 and q.size(-1) > 256:
        raise ValueError(f"KCP chunk KDA key head dimension must be at most 256; got {q.size(-1)}.")
    if lower_bound is not None and not -5 <= lower_bound < 0:
        raise ValueError(f"KCP chunk KDA lower_bound must be in [-5, 0) when safe gate is enabled; got {lower_bound}.")

    from triton_ascend_kernels.attention.fla.kda import chunk_kda

    result = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=False,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound,
        state_v_first=False,
        disable_recompute=disable_recompute,
        cp_context=context,
    )
    return _normalize_output(result, "kcp_chunk_kda")

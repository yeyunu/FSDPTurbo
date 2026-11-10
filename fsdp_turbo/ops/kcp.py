# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.
"""Operators for Kimi context parallelism (KCP)."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F


if TYPE_CHECKING:
    from triton_ascend_kernels.attention.fla.kda.cp import FLACPContext


KCPKDAKernel = Callable[..., tuple[torch.Tensor | None, ...]]


def _default_kcp_kda_kernel(*args: object, **kwargs: object) -> tuple[torch.Tensor | None, ...]:
    from fsdp_turbo.ops.kda import chunk_kda

    return chunk_kda(*args, **kwargs)


_KCP_KDA_BACKENDS: dict[str, KCPKDAKernel] = {"kcp": _default_kcp_kda_kernel}


def register_kcp_kda_backend(name: str, kernel: KCPKDAKernel) -> None:
    """Register a chunk-KDA-compatible adapter before constructing the config.

    Adapters translate the normalized KDA flags and CP context to their kernel's
    interface. Device-specific fused kernels must still use the operator registry.
    Registering an existing name with a different callable is rejected.
    """
    if not isinstance(name, str) or not name or not callable(kernel):
        raise ValueError("A KCP KDA backend requires a non-empty name and callable kernel.")
    if name in _KCP_KDA_BACKENDS and _KCP_KDA_BACKENDS[name] is not kernel:
        raise ValueError(f"KCP KDA backend {name!r} is already registered.")
    _KCP_KDA_BACKENDS[name] = kernel


def get_kcp_kda_backend(name: str) -> KCPKDAKernel:
    """Resolve a registered state backend, rejecting unavailable P2P adapters."""
    if not isinstance(name, str) or name not in _KCP_KDA_BACKENDS:
        available = ", ".join(sorted(_KCP_KDA_BACKENDS))
        raise ValueError(f"KCP KDA backend {name!r} is unavailable; registered backends: {available}.")
    return _KCP_KDA_BACKENDS[name]


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

    # Select explicitly before driver.active triggers ambiguous automatic detection.
    driver.set_active(triton_backend.driver())


def _normalize_output(
    result: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None], operator_name: str
) -> torch.Tensor:
    output = result[0] if isinstance(result, tuple) else result
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"{operator_name} must return a tensor or a tuple beginning with one.")
    return output


def build_kcp_context(
    global_cu_seqlens: torch.Tensor,
    group: dist.ProcessGroup,
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

    total_tokens = global_cu_seqlens[-1].item()
    world_size = dist.get_world_size(group)
    if total_tokens < world_size or total_tokens % world_size:
        raise ValueError("Global sequence length must be positive and divisible by the KCP group size.")
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
    """Convolve contiguous CP shards without importing the external FLA package.

    Exchange at most W-1 tail tokens per rank. Autograd returns halo gradients
    to their source ranks. Packed sequence boundaries reset the history.
    The existing PyTorch convolution implementation also runs on NPU, avoiding
    the non-CP Triton convolution backend that rejects A5.
    """
    from fsdp_turbo.ops.cpu.causal_conv1d import causal_conv1d_cpu

    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError("KCP convolution expects [1, local_sequence, channels].")
    history_size = weight.shape[-1] - 1
    if history_size:
        local_length = x.shape[1]
        tail_length = min(local_length, history_size)
        tails = x[:, -tail_length:].contiguous()
        gathered = gather_cp_tensor(tails, context.group)
        rank = dist.get_rank(context.group)
        first_rank = max(0, rank - (history_size + local_length - 1) // local_length)
        previous = gathered[first_rank:rank].reshape(1, -1, x.shape[-1])
        history = F.pad(previous[:, -history_size:], (0, 0, max(0, history_size - previous.shape[1]), 0))
        valid_history = min(context.pre_num_conv_tokens, history_size)
        mask = torch.arange(history_size, device=x.device) >= history_size - valid_history
        history = history * mask[None, :, None]
        # Rank zero must also participate in the gather's backward collective.
        history = history + gathered.sum() * 0
        first_state = F.pad(history.transpose(1, 2), (1, 0))
        initial_state = torch.cat(
            (first_state, x.new_zeros(context.num_seqs - 1, x.shape[-1], history_size + 1)),
            dim=0,
        )
    else:
        initial_state = None
    output, _ = causal_conv1d_cpu(
        x=x,
        weight=weight.transpose(0, 1),
        bias=bias,
        initial_state=initial_state,
        activation=activation,
        cu_seqlens=context.cu_seqlens_cpu,
    )
    return output


class _GatherCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        outputs = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(outputs, tensor.contiguous(), group=group)
        return torch.stack(outputs)

    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.contiguous().clone()
        dist.all_reduce(gradient, group=ctx.group)
        return gradient[ctx.rank], None


def gather_cp_tensor(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Gather equal-shaped shards with a summed backward on Gloo and HCCL."""
    return _GatherCP.apply(tensor, group)


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
    backend: str = "kcp",
) -> torch.Tensor:
    """Run a registered KCP state adapter; the default uses device dispatch."""

    if q.dim() == 4 and q.size(-1) > 256:
        raise ValueError(f"KCP chunk KDA key head dimension must be at most 256; got {q.size(-1)}.")
    if lower_bound is not None and not -5 <= lower_bound < 0:
        raise ValueError(f"KCP chunk KDA lower_bound must be in [-5, 0) when safe gate is enabled; got {lower_bound}.")

    result = get_kcp_kda_backend(backend)(
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

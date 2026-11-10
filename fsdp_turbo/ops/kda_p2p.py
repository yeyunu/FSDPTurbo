# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Semantic KDA interface with an injected state context."""

from typing import Protocol

import torch

from fsdp_turbo.ops.registry import get_op


class KdaStateContext(Protocol):
    """Local interval metadata and ordered KDA boundary-state computations.

    Summaries have shape ``[intervals, heads, dim, 2 * dim]`` and contain
    ``[E | M]``, mapping a state H to ``M @ H + E``. Both computations return
    ``[intervals, heads, dim, dim]`` in the same local interval order, on the
    summaries' device and with their dtype. Summaries use FP32 in the current
    CPU/NPU implementations. The first global interval has a zero initial
    state; the last has a zero terminal-state gradient.

    Implementations own topology and ordered merging. All state ranks must
    participate, including boundary ranks. A singleton axis stays local.
    This protocol carries no saved activations and requires no inheritance
    or runtime type checks from concrete communication contexts.
    """

    @property
    def cu_seqlens(self) -> tuple[int, ...]:
        """Local interval offsets, starting at zero and ending at local length.

        Offsets describe concatenated chronological intervals, including the
        separated front/back intervals; they are not packed sample boundaries.
        """

    def compute_initial_states(self, summaries: torch.Tensor) -> torch.Tensor:
        """Compute exclusive forward states from forward affine summaries.

        Preserve autograd through summaries and communication for the CPU
        reference, whose backward follows this computation graph.
        """

    def compute_terminal_gradients(self, summaries: torch.Tensor) -> torch.Tensor:
        """Compute exclusive terminal gradients from backward affine summaries.

        Used explicitly by staged NPU backward; CPU backward uses autograd
        through compute_initial_states and does not call this method.
        """


def chunk_kda_p2p(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_context: KdaStateContext,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """KDA on local intervals described and connected by one state context.

    Q/K/V/g use BSND, beta is effective (already sigmoid) BSN. Shared parameter
    gradients remain local contributions; the enclosing FSDP performs reduction.
    NPU uses saved staged kernels (BF16 QKV, head dimensions 128, chunk 64),
    requiring the model's existing safe gate with -5 <= lower_bound < 0.
    """
    from fsdp_turbo.ops.cpu.kda_p2p import chunk_kda_p2p_cpu  # noqa: F401

    if q.device.type == "npu":
        from fsdp_turbo.ops.npu.kda_p2p import chunk_kda_p2p_npu  # noqa: F401
    return get_op("chunk_kda_p2p", q.device.type)(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        state_context,
        lower_bound,
    )

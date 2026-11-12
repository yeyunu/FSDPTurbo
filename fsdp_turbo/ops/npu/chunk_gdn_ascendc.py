# Copyright (c) 2026 Tianjin University Ltd.
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""AscendC GDN (Gated Delta Net) fused operators for NPU."""

import logging
from typing import Optional

import torch

from fsdp_turbo.utils.log import log_warning_once

from fla_npu.ops.ascendc import (
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule_fwd,
)

from mindspeed_ops.api.triton.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
)


logger = logging.getLogger(__name__)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _prepare_chunk_indices(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    """Generate flattened chunk indices ``[s0, c0, s1, c1, ...]`` for varlen."""
    indices: list[int] = []
    for i in range(len(cu_seqlens) - 1):
        length = cu_seqlens[i + 1] - cu_seqlens[i]
        if length <= 0:
            continue
        for chunk_id in range((length + chunk_size - 1) // chunk_size):
            indices.append(i)
            indices.append(chunk_id)
    return indices


def _flash_gdn_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens_list=None,
    chunk_indices=None,
    chunk_size=64,
    use_qk_l2norm_in_kernel=False,
):
    o, final_state, g, A, _, _, q_hat, k_hat, q_rstd_out, k_rstd_out = chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
        scale=scale,
        disable_recompute=True,
        return_intermediate_states=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        layout="BSND",
    )
    if not output_final_state:
        final_state = None
    return g, o, A, final_state, q_rstd_out, k_rstd_out, q_hat, k_hat


def _flash_gdn_bwd(
    v,
    g,
    beta,
    A,
    scale,
    initial_state,
    do,
    dht,
    cu_seqlens_list=None,
    chunk_indices=None,
    chunk_size=64,
    use_qk_l2norm_in_kernel=False,
    q_rstd=None,
    k_rstd=None,
    q_hat=None,
    k_hat=None,
):
    if dht is not None:
        raise NotImplementedError(
            "The installed AscendC state-backward kernel ignores final-state gradients (dht); "
            "use implementation='eager' when the final state participates in autograd."
        )
    dq, dk, dv, db, dg, dh0, _, _ = chunk_gated_delta_rule_bwd(
        q_hat,
        k_hat,
        v,
        g,
        beta,
        A,
        do,
        scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
        initial_state=initial_state,
        dht=dht,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        layout="BSND",
    )
    if dg.dtype != torch.float32:
        raise ValueError(f"dg current type is {dg.dtype}, should be float32")
    return dq, dk, dv, db, dg, dh0


class _ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens_list=None,
        chunk_indices=None,
        use_qk_l2norm_in_kernel=False,
        chunk_size=64,
    ):
        g, o, A, final_state, q_rstd, k_rstd, q_hat, k_hat = _flash_gdn_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(q, k, v, g, beta, A)
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.q_hat = q_hat
        ctx.k_hat = k_hat
        ctx.initial_state = initial_state
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, g, beta, A = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = _flash_gdn_bwd(
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=ctx.initial_state,
            do=do,
            dht=dht,
            cu_seqlens_list=ctx.cu_seqlens_list,
            chunk_indices=ctx.chunk_indices,
            chunk_size=ctx.chunk_size,
            use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
            q_rstd=ctx.q_rstd,
            k_rstd=ctx.k_rstd,
            q_hat=ctx.q_hat,
            k_hat=ctx.k_hat,
        )
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def chunk_gdn_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    head_first: bool = False,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""AscendC GDN entry point.

    Expects q/k/v in ``[B, T, H, D]`` (time-first) and g/beta in ``[B, T, H]``.
    Returns ``(o, final_state)`` where o is ``[B, T, H, V]``.
    Like the MindSpeed-MM AscendC core, final state uses the initial state's
    dtype when supplied, otherwise FP32.
    The Ascend 950 composite backward requires a BF16 initial state matching q.
    Final-state gradients are rejected:
    the currently supported state-backward kernel ignores its dht argument.
    The fused ``chunk_gated_delta_rule_fwd``/``chunk_gated_delta_rule_bwd``
    AscendC composite-core kernels with the ``BSND`` layout are always used.
    """
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(f"q={q.dtype}, k={k.dtype}, v={v.dtype}; they should be equal")
    if q.dtype == torch.float32:
        raise ValueError("AscendC GDN does not support float32. Please use bfloat16/float16.")
    if beta.ndim != 3:
        raise ValueError("beta must be rank-3 with shape [B, T, H].")
    if head_first:
        log_warning_once(logger, "head_first is deprecated. Inputs must be [B, T, H, D].")
    if not head_first and q.shape[1] < q.shape[2]:
        log_warning_once(
            logger,
            f"seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}); "
            "inputs may be in head-first format [B, H, T, D] instead of [B, T, H, D].",
        )
    if not _is_power_of_two(chunk_size):
        raise ValueError(f"chunk_size must be a power of 2, got {chunk_size}.")

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"batch size must be 1 when using cu_seqlens, got {q.shape[0]}.")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError("initial_state count must match number of sequences.")
        cu_seqlens_list = cu_seqlens.tolist()
        chunk_indices = _prepare_chunk_indices(cu_seqlens_list, chunk_size)
    else:
        cu_seqlens_list = None
        chunk_indices = None

    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = _ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens_list,
        chunk_indices,
        use_qk_l2norm_in_kernel,
        chunk_size,
    )
    return o, final_state

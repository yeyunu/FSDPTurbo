# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Saved staged AscendC KDA, including reverse P2P boundary gradients.

Derived from fla-npu-private/examples/cp/chunk_kda_cp.py. The state
context is injected; this module never imports the distributed business layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fsdp_turbo.ops.registry import register_op

if TYPE_CHECKING:
    from fsdp_turbo.ops.kda_p2p import KdaStateContext


class _ChunkKdaP2P(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, A_log, dt_bias, state_context, lower_bound):
        from fla_npu.ops import ascendc as op

        ctx.options = dict(
            safe_gate=True,
            lower_bound=lower_bound,
            use_gate_in_kernel=True,
            use_exp2=True,
            use_beta_sigmoid_in_kernel=False,
        )
        ctx.cu = list(state_context.cu_seqlens)
        ctx.state_context, ctx.scale = state_context, 128**-0.5
        ctx.input_dtypes = tuple(tensor.dtype for tensor in (q, k, v, g, beta, A_log, dt_bias))
        g, beta = (tensor.float().contiguous() for tensor in (g, beta))
        # A contiguous FP32 head slice may retain its parent's NPU storage shape.
        # The staged ABI checks storage shapes, so materialize exact-sized copies.
        A_log, dt_bias = (tensor.float().clone(memory_format=torch.contiguous_format) for tensor in (A_log, dt_bias))
        q, k, v, g = (tensor.transpose(1, 2).contiguous() for tensor in (q, k, v, g))
        beta = beta.transpose(1, 2).contiguous()
        saved = op.chunk_kda_fwd_prepare(
            q,
            k,
            v,
            g,
            beta,
            ctx.scale,
            layout="BNSD",
            cu_seqlens=ctx.cu,
            backward_mode="save",
            A_log=A_log,
            dt_bias=dt_bias.flatten(),
            use_qk_l2norm_in_kernel=True,
            **ctx.options,
        )
        gk, aqk, akk, w, u, qg, kg, qgs, qhat, khat, qr, kr, beta_eff = saved
        summaries = torch.stack(
            [
                op.pre_process_fwd_kernel_merged(
                    kg,
                    w,
                    u,
                    gk=gk,
                    cu_seqlens=ctx.cu[index : index + 2],
                    chunk_size=64,
                )[0]
                for index in range(len(ctx.cu) - 1)
            ]
        )
        initial_states = state_context.compute_initial_states(summaries)
        h, vn, _ = op.chunk_fwd_h(
            kg,
            w,
            u,
            gk=gk,
            initial_state=initial_states,
            cu_seqlens=ctx.cu,
            save_new_value=True,
            output_final_state=False,
            use_exp2=True,
        )
        output = op.chunk_kda_fwd_finalize(qgs, aqk, vn, h, cu_seqlens=ctx.cu, output_layout="BSND")
        ctx.save_for_backward(
            v,
            g,
            A_log,
            dt_bias,
            gk,
            aqk,
            akk,
            w,
            qg,
            kg,
            qhat,
            khat,
            qr,
            kr,
            beta_eff,
            h,
            vn,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from fla_npu.ops import ascendc as op

        (
            v,
            g,
            A_log,
            dt_bias,
            gk,
            aqk,
            akk,
            w,
            qg,
            kg,
            qhat,
            khat,
            qr,
            kr,
            beta_eff,
            h,
            vn,
        ) = ctx.saved_tensors
        grad_head_major = grad_output.transpose(1, 2).contiguous()
        da, dv_local, dq_raw = op.chunk_kda_bwd_prepare(
            aqk.squeeze(0),
            vn.squeeze(0),
            grad_head_major.squeeze(0),
            h.squeeze(0),
            scale=ctx.scale,
            layout="NTD",
            cu_seqlens=ctx.cu,
        )
        da, dv_local, dq_raw = (tensor.unsqueeze(0) for tensor in (da, dv_local, dq_raw))
        summaries = torch.stack(
            [
                op.pre_process_bwd_kernel_merged(
                    qg,
                    kg,
                    w,
                    grad_head_major,
                    dv_local,
                    ctx.scale,
                    64,
                    gk=gk.to(qhat.dtype),
                    cu_seqlens=ctx.cu[index : index + 2],
                )
                for index in range(len(ctx.cu) - 1)
            ]
        )
        terminal_gradients = ctx.state_context.compute_terminal_gradients(summaries).to(qhat.dtype)
        indices = [
            value
            for seq, (start, end) in enumerate(zip(ctx.cu[:-1], ctx.cu[1:]))
            for chunk in range((end - start + 63) // 64)
            for value in (seq, chunk)
        ]
        dh, _, dv_scan = op.chunk_gated_delta_rule_bwd_dhu(
            qg,
            kg,
            w,
            grad_head_major,
            dv_local,
            ctx.scale,
            64,
            gK=gk,
            dht=terminal_gradients,
            cu_seqlens=ctx.cu,
            chunk_indices=indices,
            use_exp2=True,
        )
        dq, dk, dv, dbeta, dg, dA, dbias = op.chunk_kda_bwd_finalize(
            qhat,
            khat,
            v,
            gk,
            g,
            beta_eff,
            A_log,
            dt_bias,
            akk,
            vn,
            h,
            dh,
            dv_scan,
            da,
            dq_raw,
            scale=ctx.scale,
            q_rstd=qr,
            k_rstd=kr,
            cu_seqlens=ctx.cu,
            output_layout="BSND",
            **ctx.options,
        )
        gradients = (dq, dk, dv, dg, dbeta, dA, dbias)
        return (
            *(gradient.to(dtype) for gradient, dtype in zip(gradients, ctx.input_dtypes)),
            None,
            None,
        )


@register_op("chunk_kda_p2p", "npu")
def chunk_kda_p2p_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_context: KdaStateContext,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run BF16, B=1, equal 128-dimensional heads through the saved AscendC ABI."""
    if lower_bound is None or not -5.0 <= lower_bound < 0.0:
        raise ValueError(
            "NPU KDA P2P staged backward requires a trained safe gate with -5 <= gate_lower_bound < 0; unbounded gates are unsupported."
        )
    if q.shape[0] != 1 or q.shape[-1] != 128 or v.shape != q.shape or k.shape != q.shape or q.dtype != torch.bfloat16:
        raise ValueError("NPU KDA P2P requires B=1, BF16 QKV, equal heads and head_dim=128.")
    return _ChunkKdaP2P.apply(q, k, v, g, beta, A_log, dt_bias, state_context, lower_bound)

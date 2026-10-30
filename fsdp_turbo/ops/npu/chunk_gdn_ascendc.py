# Copyright (c) 2026 Tianjin University Ltd.
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""AscendC GDN (Gated Delta Net) fused operators for NPU."""

import logging
from typing import Optional

import torch

from fsdp_turbo.ops.triton.utils import is_arch35
from fsdp_turbo.utils.log import log_warning_once

from fla_npu.ops.ascendc import (
    chunk_bwd_dqkwg,
    chunk_bwd_dv_local,
    chunk_fwd_o,
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule_bwd_dhu,
    chunk_gated_delta_rule_fwd,
    chunk_gated_delta_rule_fwd_h,
    chunk_local_cumsum as ascendc_chunk_local_cumsum,
    chunk_scaled_dot_kkt as ascendc_chunk_scaled_dot_kkt,
    prepare_wy_repr_bwd_da,
    prepare_wy_repr_bwd_full,
    recompute_w_u_fwd,
    solve_tri as ascendc_solve_tri,
)

from mindspeed_ops.api.triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from mindspeed_ops.api.triton.cumsum import chunk_local_cumsum
from mindspeed_ops.api.triton.l2norm import l2norm_bwd, l2norm_fwd
from mindspeed_ops.api.triton.solve_tril import solve_tril
from mindspeed_ops.api.triton.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
)


logger = logging.getLogger(__name__)

_SOLVE_TRI_ASCENDC_AVAILABLE: Optional[bool] = None
_SOLVE_TRI_ASCENDC_UNAVAILABLE_REASON = ""


def _as_int_list(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    return [int(x) for x in value]


def _probe_solve_tri_ascendc(device: torch.device, dtype: torch.dtype) -> bool:
    global _SOLVE_TRI_ASCENDC_AVAILABLE, _SOLVE_TRI_ASCENDC_UNAVAILABLE_REASON

    if _SOLVE_TRI_ASCENDC_AVAILABLE is not None:
        return _SOLVE_TRI_ASCENDC_AVAILABLE

    probe_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16
    try:
        probe = torch.zeros((1, 64, 1, 128), dtype=probe_dtype, device=device)
        ascendc_solve_tri(probe, layout="bsnd")
        torch.accelerator.synchronize()
    except Exception as exc:
        _SOLVE_TRI_ASCENDC_AVAILABLE = False
        _SOLVE_TRI_ASCENDC_UNAVAILABLE_REASON = str(exc).splitlines()[0]
        try:
            torch.accelerator.synchronize()
        except Exception:  # nosec B110
            pass
        return False

    _SOLVE_TRI_ASCENDC_AVAILABLE = True
    _SOLVE_TRI_ASCENDC_UNAVAILABLE_REASON = ""
    return True


def solve_tri_auto(
    A: torch.Tensor,
    *,
    cu_seqlens=None,
    cu_seqlens_list=None,
    chunk_indices=None,
    output_dtype: torch.dtype = torch.float,
    head_first: bool = False,
) -> torch.Tensor:
    """Solve triangular system using AscendC when available, falling back to Triton."""
    global _SOLVE_TRI_ASCENDC_AVAILABLE, _SOLVE_TRI_ASCENDC_UNAVAILABLE_REASON

    if head_first:
        return _solve_tri_head_first(
            A,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            output_dtype=output_dtype,
        )

    if _probe_solve_tri_ascendc(A.device, output_dtype):
        cu_list = _as_int_list(cu_seqlens_list)
        chunk_list = _as_int_list(chunk_indices)
        A_in = A.to(output_dtype).contiguous()
        try:
            if cu_list is None:
                return ascendc_solve_tri(A_in, layout="bsnd")
            if A_in.ndim != 4 or A_in.shape[0] != 1:
                raise ValueError(f"solve_tri varlen path expects A with shape [1, T, H, BT], got {tuple(A_in.shape)}.")
            if chunk_list is None:
                raise ValueError("solve_tri varlen path requires chunk_indices.")
            out = ascendc_solve_tri(
                A_in.squeeze(0),
                cu_seqlens=cu_list,
                chunk_indices=chunk_list,
                layout="tnd",
            )
            return out.unsqueeze(0)
        except Exception as exc:
            _SOLVE_TRI_ASCENDC_AVAILABLE = False
            _SOLVE_TRI_ASCENDC_UNAVAILABLE_REASON = str(exc).splitlines()[0]
            try:
                torch.accelerator.synchronize()
            except Exception:  # nosec B110
                pass

    log_warning_once(
        logger,
        "AscendC npu_solve_tri is unavailable; falling back to Triton solve_tril_npu. "
        f"Reason: {_SOLVE_TRI_ASCENDC_UNAVAILABLE_REASON}",
    )
    return solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=output_dtype)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _should_use_ascendc_cumsum(g: torch.Tensor, *, chunk_size: int, output_dtype: torch.dtype) -> bool:
    if not _is_power_of_two(int(chunk_size)):
        return False
    if g.dim() != 3:
        return False
    if output_dtype not in (torch.float, torch.float32):
        return False
    return True


def _should_use_ascendc_kkt(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
    output_dtype: torch.dtype,
) -> bool:
    if chunk_size not in (16, 32, 64, 128):
        return False
    if k.dim() != 4 or g.dim() != 3 or beta.dim() != 3:
        return False
    if k.dtype not in (torch.float16, torch.bfloat16):
        return False
    if output_dtype not in (torch.float, torch.float32):
        return False
    if g.shape != beta.shape:
        return False
    B, Hk, T, _ = k.shape
    if g.shape[0] != B or g.shape[2] != T:
        return False
    Hv = g.shape[1]
    return Hk > 0 and Hv > 0 and Hv % Hk == 0


def chunk_local_cumsum_auto(
    g: torch.Tensor,
    *,
    cu_seqlens=None,
    cu_seqlens_list=None,
    chunk_indices=None,
    chunk_size: int,
    output_dtype: torch.dtype,
    reverse: bool = False,
    head_first: bool = False,
) -> torch.Tensor:
    """Prefer AscendC chunk_local_cumsum, fall back to Triton."""
    if _should_use_ascendc_cumsum(g, chunk_size=chunk_size, output_dtype=output_dtype):
        g_ascendc = g.contiguous() if head_first else g.transpose(1, 2).contiguous()
        cu_list = _as_int_list(cu_seqlens_list)
        chunk_list = _as_int_list(chunk_indices)
        op_kwargs = {
            "reverse": reverse,
            "scale": 1.0,
            "head_first": True,
            "output_dtype": "float32",
        }
        if cu_list is not None or chunk_list is not None:
            op_kwargs["cu_seqlens"] = cu_list
            op_kwargs["chunk_indices_out"] = chunk_list
        out = ascendc_chunk_local_cumsum(g_ascendc.float(), chunk_size, **op_kwargs)
        return out if head_first else out.transpose(1, 2).contiguous()

    return chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
    )


def chunk_scaled_dot_kkt_fwd_auto(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    cu_seqlens=None,
    cu_seqlens_list=None,
    chunk_indices=None,
    chunk_size: int,
    output_dtype: torch.dtype,
    head_first: bool = False,
) -> torch.Tensor:
    """Prefer AscendC chunk_scaled_dot_kkt, fall back to Triton."""
    k_h, g_h, beta_h = (k, g, beta) if head_first else (k.transpose(1, 2), g.transpose(1, 2), beta.transpose(1, 2))
    if _should_use_ascendc_kkt(k_h, g_h, beta_h, chunk_size=chunk_size, output_dtype=output_dtype):
        cu_list = _as_int_list(cu_seqlens_list)
        chunk_list = _as_int_list(chunk_indices)
        op_kwargs = {"chunk_size": chunk_size}
        if cu_list is not None or chunk_list is not None:
            op_kwargs["cu_seqlens"] = cu_list
            op_kwargs["chunk_indices"] = chunk_list
        out = ascendc_chunk_scaled_dot_kkt(
            k_h.contiguous(),
            g_h.contiguous().float(),
            beta_h.contiguous().float(),
            **op_kwargs,
        )
        return out if head_first else out.transpose(1, 2).contiguous()

    # MindSpeed's existing call uses time-first tensors.
    if head_first:
        k, g, beta = (x.transpose(1, 2).contiguous() for x in (k, g, beta))
    out = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=output_dtype,
    )

    return out.transpose(1, 2).contiguous() if head_first else out


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


def _solve_tri_head_first(A, *, cu_seqlens, cu_seqlens_list, chunk_indices, output_dtype):
    """BNSD in/out. Packed solve uses the established Triton fallback.

    Do not send packed sequences through dense bnsd: chunk boundaries differ.
    The installed stable solve wrapper also documents unsafe packed layouts.
    """
    if cu_seqlens_list is None:
        try:
            return ascendc_solve_tri(A.to(output_dtype).contiguous(), layout="bnsd")
        except Exception as exc:
            log_warning_once(logger, f"AscendC BNSD solve unavailable; using Triton: {exc}")
    A_time = A.transpose(1, 2).contiguous()
    result = solve_tril(A=A_time, cu_seqlens=cu_seqlens, output_dtype=output_dtype)
    return result.transpose(1, 2).contiguous()


def _flash_gdn_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens=None,
    cu_seqlens_list=None,
    chunk_indices=None,
    chunk_size=64,
    use_composite_core=True,
    use_qk_l2norm_in_kernel=False,
):
    if use_composite_core:
        if is_arch35():
            layout = "BSND"
        else:
            layout = "BNSD"
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
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
            layout=layout,
        )
        if not output_final_state:
            final_state = None
        return g, o, A, final_state, q_rstd_out, k_rstd_out, q_hat, k_hat

    # One boundary conversion; every AscendC intermediate stays head-first.
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous().float()
    beta = beta.transpose(1, 2).contiguous().float()
    g = chunk_local_cumsum_auto(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        head_first=True,
        output_dtype=torch.float32,
    )
    A = chunk_scaled_dot_kkt_fwd_auto(
        k,
        g,
        beta,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        head_first=True,
        output_dtype=torch.float32,
    )
    A = solve_tri_auto(
        A,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        output_dtype=k.dtype,
        head_first=True,
    )

    w, u = recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g,
        gk=None,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )
    if is_arch35():
        output_layout = "BSND"
    else:
        output_layout = "BNSD"
    o = chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        scale,
        g=g,
        g_gamma=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        output_layout=output_layout,
    )
    if output_layout == "BNSD":
        o = o.transpose(1, 2).contiguous()

    # AscendC [B, H, T, D] -> Triton [B, T, H, D]
    g = g.transpose(1, 2).contiguous()
    return g, o, A, final_state, None, None, None, None


def _flash_gdn_bwd(
    q,
    k,
    v,
    g,
    beta,
    A,
    scale,
    initial_state,
    do,
    dht,
    cu_seqlens=None,
    cu_seqlens_list=None,
    chunk_indices=None,
    chunk_size=64,
    use_composite_core=True,
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
    if use_composite_core and is_arch35():
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

    if initial_state is not None:
        raise NotImplementedError("Initial-state gradients require the Ascend 950 composite core.")

    # Step-by-step path (non-A5 or non-composite fallback)
    # Triton [B, T, H, D] -> AscendC [B, H, T, D]
    v = v.transpose(1, 2).contiguous()
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    do = do.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()

    w, u = recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g,
        gk=None,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
        output_final_state=False,
        chunk_size=chunk_size,
    )
    dv = chunk_bwd_dv_local(
        q,
        k,
        do,
        g,
        scale,
        chunk_size,
        g_gamma=None,
        A=A,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do,
        dv,
        scale,
        chunk_size,
        g=g,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
        use_exp2=False,
        transpose_state_layout=False,
    )
    dh0 = None
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q,
        k,
        v_new,
        g,
        h,
        do,
        dh,
        dv,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
        w=None,
        g_gamma=None,
        scale=scale,
        use_exp2=False,
        transpose_state_layout=False,
    )

    dA = prepare_wy_repr_bwd_da(
        k,
        v,
        beta,
        A,
        dw,
        dv,
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd_full(
        k,
        v,
        beta,
        A,
        dA,
        dw,
        dv,
        g,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
    )

    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(f"dg current type is {dg.dtype}, should be float32")
    dg = chunk_local_cumsum_auto(
        dg,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        reverse=True,
        head_first=True,
        output_dtype=torch.float32,
    )
    # Autograd gradients must match the time-first public inputs.
    dq, dk, dv, db, dg = (x.transpose(1, 2).contiguous() for x in (dq, dk, dv, db, dg))
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
        cu_seqlens=None,
        cu_seqlens_list=None,
        chunk_indices=None,
        use_qk_l2norm_in_kernel=False,
        chunk_size=64,
        use_composite_core=True,
    ):
        use_internal_l2norm = use_composite_core and is_arch35()
        if use_qk_l2norm_in_kernel and not use_internal_l2norm:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A, final_state, q_rstd_out, k_rstd_out, q_hat, k_hat = _flash_gdn_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            use_composite_core=use_composite_core,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel and use_internal_l2norm,
        )
        if q_rstd is None and q_rstd_out is not None:
            q_rstd = q_rstd_out
        if k_rstd is None and k_rstd_out is not None:
            k_rstd = k_rstd_out
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(q, k, v, g, beta, A)
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.q_hat = q_hat
        ctx.k_hat = k_hat
        ctx.initial_state = initial_state
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_internal_l2norm = use_internal_l2norm
        ctx.chunk_size = chunk_size
        ctx.use_composite_core = use_composite_core
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, g, beta, A = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = _flash_gdn_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=ctx.initial_state,
            do=do,
            dht=dht,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_list=ctx.cu_seqlens_list,
            chunk_indices=ctx.chunk_indices,
            chunk_size=ctx.chunk_size,
            use_composite_core=ctx.use_composite_core,
            use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel and ctx.use_internal_l2norm,
            q_rstd=ctx.q_rstd,
            k_rstd=ctx.k_rstd,
            q_hat=ctx.q_hat,
            k_hat=ctx.k_hat,
        )
        if ctx.use_qk_l2norm_in_kernel and not ctx.use_internal_l2norm:
            dq = l2norm_bwd(q, ctx.q_rstd, dq)
            dk = l2norm_bwd(k, ctx.k_rstd, dk)
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
    use_composite_core: bool = True,
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

    When ``use_composite_core=True`` (default), the fused
    ``chunk_gated_delta_rule_fwd`` AscendC operator is used for the forward
    pass.  When ``False``, a step-by-step path composed of triton cumsum/kkt/
    solve_tril followed by AscendC w-u/h/o kernels is used instead.
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
        cu_seqlens,
        cu_seqlens_list,
        chunk_indices,
        use_qk_l2norm_in_kernel,
        chunk_size,
        use_composite_core,
    )
    return o, final_state

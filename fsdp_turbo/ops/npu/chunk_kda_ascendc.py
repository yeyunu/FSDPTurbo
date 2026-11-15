# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li


import warnings
import inspect
from functools import lru_cache

import torch

from fla_npu.ops.ascendc import chunk_kda_fwd as ascendc_chunk_kda_fwd
from fla_npu.ops.ascendc import chunk_kda_bwd as ascendc_chunk_kda_bwd

from triton_ascend_kernels.attention.fla.kda.chunk import (
    l2norm_fwd,
    l2norm_bwd,
    chunk_kda_bwd as triton_chunk_kda_bwd,
    chunk_kda_fwd,
    prepare_chunk_indices,
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
)


def _platform_from_name(name):
    if name.startswith("Ascend950"):
        return "a5"
    if name.startswith(("Ascend910B", "Ascend910_93")):
        return "a2a3"
    raise RuntimeError(f"Unsupported KDA NPU platform: {name}")


@lru_cache(maxsize=None)
def _ascend_platform(device):
    # Key by the input device, not the process's current device.
    return _platform_from_name(torch.npu.get_device_name(device))


@lru_cache(maxsize=None)
def _legacy_bwd_kwargs():
    # Older A2/A3 wheels expose only the original fused backward and have no
    # implementation argument. New wheels require explicit routing to legacy.
    parameters = inspect.signature(ascendc_chunk_kda_bwd).parameters
    return {"implementation": "legacy"} if "implementation" in parameters else {}


def _a2a3_recompute(q, k, v, g, beta, akk, *, A_log, dt_bias, cu_seqlens, use_gate_in_kernel, lower_bound):
    # A2/A3 wheels do not ship an executable AscendC KDA recompute. Use
    # the model's Triton small operators, then the AscendC state recurrence.
    # Imports are lazy so the A5 dependency surface remains unchanged.
    from fla_npu.ops.ascendc import chunk_fwd_h
    from triton_ascend_kernels.attention.fla.kda.gate import kda_gate_chunk_cumsum
    from triton_ascend_kernels.attention.fla.kda.cumsum_kda import chunk_local_cumsum
    from triton_ascend_kernels.attention.fla.kda.wy_fast import recompute_w_u_fwd

    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int64, device=q.device)
    indices = (
        None
        if cu_seqlens is None
        else torch.tensor(
            [
                (sequence, chunk)
                for sequence, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:]))
                for chunk in range((end - start + 63) // 64)
            ],
            dtype=torch.int64,
            device=q.device,
        )
    )
    # Triton uses BSND, whereas AscendC FwdH uses BNSD even for packed input.
    g_raw = _bnsd_to_bsnd(g)
    gate_kwargs = dict(chunk_size=64, scale=1.4426950408889634, cu_seqlens=cu, chunk_indices=indices)
    if use_gate_in_kernel:
        gk = kda_gate_chunk_cumsum(
            g=g_raw,
            A_log=A_log,
            dt_bias=None if dt_bias is None else dt_bias.reshape(-1),
            lower_bound=lower_bound,
            **gate_kwargs,
        )
    else:
        gk = chunk_local_cumsum(g=g_raw, **gate_kwargs)
    w, u, qg, kg = recompute_w_u_fwd(
        q=_bnsd_to_bsnd(q),
        k=_bnsd_to_bsnd(k),
        v=_bnsd_to_bsnd(v),
        beta=_bhs_to_bsh(beta),
        A=_bnsd_to_bsnd(akk),
        gk=gk,
        cu_seqlens=cu,
        chunk_indices=indices,
    )
    gk, w, u, qg, kg = map(_bsnd_to_bnsd, (gk, w, u, qg, kg))
    h, v_new, _ = chunk_fwd_h(
        kg,
        w,
        u,
        gk=gk,
        chunk_size=64,
        cu_seqlens=cu_seqlens,
        use_exp2=True,
        state_v_first=False,
    )
    return gk, w, qg, kg, v_new, h


def fused_beta_sigmoid_fwd(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return (torch.sigmoid(x.float()) * scale).to(x.dtype)


def fused_beta_sigmoid_bwd(x: torch.Tensor, dy: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    x = x.float()
    s = torch.sigmoid(x)
    return (dy.float() * scale * s * (1.0 - s)).to(x.dtype)


def fused_beta_sigmoid(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Compute ``scale * sigmoid(x)``.

    Used to map raw beta logits into ``[0, scale]`` before launching the chunk
    kernel when ``use_beta_sigmoid_in_kernel=True``.
    """
    return fused_beta_sigmoid_fwd(x, scale)


def _host_int_tuple(value):
    """Convert tensor metadata to one flat Host tuple with at most one D2H."""

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(int(item) for item in value)


def _bnsd_to_bsnd(tensor):
    """Restore the sequence-major layout consumed by the Triton backward."""

    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise RuntimeError(
            f"AscendC KDA saved tensors must be rank-4 BNSD tensors, but received shape {tuple(tensor.shape)}."
        )
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bsnd_to_bnsd(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise RuntimeError(f"Expected one rank-4 BSND tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bsh_to_bhs(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected one rank-3 BSH tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1).contiguous()


def _bhs_to_bsh(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected one rank-3 BHS tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1).contiguous()


def _bsnd_to_ntd(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 4 or tensor.shape[0] != 1:
        raise RuntimeError(
            f"Varlen KDA tensors must use packed BSND with B=1, but received shape {tuple(tensor.shape)}."
        )
    return tensor.squeeze(0).permute(1, 0, 2).contiguous()


def _bsh_to_nt(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 3 or tensor.shape[0] != 1:
        raise RuntimeError(
            f"Varlen KDA tensors must use packed BSH with B=1, but received shape {tuple(tensor.shape)}."
        )
    return tensor.squeeze(0).transpose(0, 1).contiguous()


def _ntd_to_bsnd(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected one rank-3 NTD tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(1, 0, 2).unsqueeze(0).contiguous()


def _nt_to_bsh(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 2:
        raise RuntimeError(f"Expected one rank-2 NT tensor, got shape {tuple(tensor.shape)}.")
    return tensor.transpose(0, 1).unsqueeze(0).contiguous()


def _varlen_saved_tensor(tensor):
    if tensor is None:
        return None
    if tensor.shape[0] != 1:
        raise RuntimeError(
            "Varlen AscendC intermediates must have a leading singleton batch dimension, "
            f"but received shape {tuple(tensor.shape)}."
        )
    return tensor.squeeze(0)


class ChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        chunk_size: int = 64,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        transpose_state_layout: bool = False,
    ):
        platform = _ascend_platform(q.device) if chunk_size == 64 else "triton"
        external_norm = platform != "a5"
        if platform == "a2a3":
            invalid_inputs = (
                q.dtype not in (torch.bfloat16, torch.float16),
                k.dtype != q.dtype,
                v.dtype != q.dtype,
                q.shape[-1] != 128,
                v.shape[2:] != (q.shape[2], 128),
                min(q.shape[:3]) <= 0,
                not safe_gate,
            )
            if any(invalid_inputs):
                raise ValueError(
                    "A2/A3 fused KDA requires matching FP16/BF16 q/k/v, "
                    "K=V=128, Hq=Hv, positive B/T/H and safe_gate=True."
                )
            if not disable_recompute and q.dtype != torch.bfloat16:
                raise ValueError("A2/A3 KDA recompute + fwd_h requires BF16 q/k/v.")
        # A2/A3 legacy backward does not fuse L2Norm backward.
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel and external_norm:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        beta_raw = beta
        # The public A5 V2 backward reserves fused beta sigmoid, so beta stays
        # external on both platforms even when A5 handles Q/K normalization.
        if use_beta_sigmoid_in_kernel:
            beta = fused_beta_sigmoid(beta_raw, scale=2.0 if allow_neg_eigval else 1.0)

        chunk_indices = None
        g_input = g
        use_fused_kda_bwd = False
        cu_seqlens_host = None

        use_triton = chunk_size != 64
        v2_aux = (None,) * 5

        if use_triton:
            if cu_seqlens is not None:
                chunk_indices = prepare_chunk_indices(
                    cu_seqlens,
                    chunk_size,
                    cu_seqlens_cpu=cu_seqlens_cpu,
                )
            (o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state) = chunk_kda_fwd(
                q=q,
                k=k,
                v=v,
                g=g_input,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                chunk_indices=chunk_indices,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                use_gate_in_kernel=use_gate_in_kernel,
                A_log=A_log,
                dt_bias=dt_bias,
                chunk_size=chunk_size,
                disable_recompute=disable_recompute,
                return_intermediate_states=return_intermediate_states,
                # state_v_first=state_v_first,
            )
        else:
            # The AscendC Python entry consumes Host metadata. Prefer the CPU
            # copy already prepared by the model to avoid another D2H.
            cu_seqlens_host = _host_int_tuple(cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens)
            if cu_seqlens_host:
                # Empty sequences contain no tokens or chunks. With recurrent
                # states disabled, compacting them preserves the packed layout
                # and lets forward use V2's fused normalization exports.
                cu_seqlens_host = (cu_seqlens_host[0],) + tuple(
                    end for start, end in zip(cu_seqlens_host, cu_seqlens_host[1:]) if end != start
                )
            if platform == "a5" and not disable_recompute and not return_intermediate_states:
                # The A5 V2 backward supports both dense and packed recompute.
                # Check its narrower contract before spending a forward launch.
                heads = q.shape[2]
                invalid_inputs = (
                    q.dtype != torch.bfloat16,
                    k.dtype != torch.bfloat16,
                    v.dtype != torch.bfloat16,
                    q.shape[-1] != 128,
                    v.shape[2:] != (heads, 128),
                    min(q.shape[:3]) <= 0,
                    heads > 256,
                    heads % 8 != 0,
                    not safe_gate,
                    not use_gate_in_kernel,
                )
                if any(invalid_inputs):
                    raise ValueError(
                        "AscendC KDA disable_recompute=False requires the A5 optimized "
                        "backward: BF16 q/k/v, K=V=128, Hq=Hv, positive B/T/H, "
                        "H<=256 divisible by 8, safe_gate=True and use_gate_in_kernel=True."
                    )
            use_fused_kda_bwd = True

            # Export head-major normalized inputs for the existing backward.
            batch, tokens, heads, dim = q.shape
            head_shape = (batch, heads, tokens, dim)
            scalar_shape = (batch, heads, tokens)
            qh = (
                torch.empty(head_shape, dtype=q.dtype, device=q.device)
                if use_qk_l2norm_in_kernel and platform == "a5"
                else None
            )
            kh = (
                torch.empty(head_shape, dtype=k.dtype, device=k.device)
                if use_qk_l2norm_in_kernel and platform == "a5"
                else None
            )
            qr = (
                torch.empty(scalar_shape, dtype=torch.float32, device=q.device)
                if use_qk_l2norm_in_kernel and platform == "a5"
                else None
            )
            kr = (
                torch.empty(scalar_shape, dtype=torch.float32, device=k.device)
                if use_qk_l2norm_in_kernel and platform == "a5"
                else None
            )
            be = None
            v2_aux = (qh, kh, qr, kr, be)
            fwd_norm_kwargs = (
                dict(
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    use_beta_sigmoid_in_kernel=False,
                    allow_neg_eigval=False,
                    q_hat_out=qh,
                    k_hat_out=kh,
                    q_rstd_out=qr,
                    k_rstd_out=kr,
                    beta_eff_out=be,
                )
                if platform == "a5"
                else {}
            )

            forward_outputs = ascendc_chunk_kda_fwd(
                q,
                k,
                v,
                g_input,
                beta,
                float(scale),
                int(chunk_size),
                layout="BSND",
                initial_state=initial_state,
                output_final_state=bool(output_final_state),
                cu_seqlens=cu_seqlens_host,
                chunk_indices=None,
                safe_gate=bool(safe_gate),
                lower_bound=lower_bound,
                use_gate_in_kernel=bool(use_gate_in_kernel),
                A_log=A_log if use_gate_in_kernel else None,
                dt_bias=dt_bias if use_gate_in_kernel else None,
                disable_recompute=bool(disable_recompute),
                return_intermediate_states=bool(return_intermediate_states),
                state_v_first=bool(state_v_first),
                **fwd_norm_kwargs,
            )

            (o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state) = forward_outputs[:12]

            # Both fused backward paths consume head-major intermediates.
            u = None
            if not disable_recompute and not return_intermediate_states:
                # A5 reconstructs these inside V2; A2/A3 reconstructs them
                # with standalone recompute + FwdH before the fused backward.
                # Do not retain these large caches across forward/backward.
                g_cumsum = w = qg = kg = v_new = h = None

        if (
            not use_fused_kda_bwd
            and not return_intermediate_states
            and cu_seqlens is not None
            and chunk_indices is None
        ):
            chunk_indices = prepare_chunk_indices(
                cu_seqlens,
                chunk_size,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )

        if return_intermediate_states:
            assert torch.is_inference_mode_enabled(), "return_intermediate_states is only allowed in inference mode"
            assert disable_recompute is False, "return_intermediate_states must be used with disable_recompute=False"
            return o.type_as(q), final_state, h

        saved_cu_seqlens = None if use_fused_kda_bwd else cu_seqlens
        saved_chunk_indices = None if use_fused_kda_bwd else chunk_indices
        ctx.save_for_backward(
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g_cumsum,
            g_input,
            beta_raw,
            beta,
            A_log,
            dt_bias,
            Aqk,
            Akk,
            w,
            u,
            qg,
            kg,
            v_new,
            h,
            initial_state,
            saved_cu_seqlens,
            saved_chunk_indices,
            *v2_aux,
        )
        ctx.chunk_size = chunk_size
        ctx.platform = platform
        ctx.external_norm = external_norm
        ctx.safe_gate = safe_gate
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.allow_neg_eigval = allow_neg_eigval
        ctx.disable_recompute = disable_recompute
        ctx.state_v_first = state_v_first
        ctx.use_fused_kda_bwd = use_fused_kda_bwd
        ctx.cu_seqlens_host = cu_seqlens_host if use_fused_kda_bwd else None
        return o.type_as(q), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        (
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g_cumsum,
            g_input,
            beta_raw,
            beta,
            A_log,
            dt_bias,
            Aqk,
            Akk,
            w,
            u,
            qg,
            kg,
            v_new,
            h,
            initial_state,
            cu_seqlens,
            chunk_indices,
            v2_qh,
            v2_kh,
            v2_qr,
            v2_kr,
            _v2_be,
        ) = ctx.saved_tensors

        if ctx.use_fused_kda_bwd:
            if dht is not None:
                raise RuntimeError("Fused AscendC KDA backward does not support final-state gradients.")

            is_varlen = ctx.cu_seqlens_host is not None
            if is_varlen:
                q_head = None if ctx.use_qk_l2norm_in_kernel else _bsnd_to_ntd(q)
                k_head = None if ctx.use_qk_l2norm_in_kernel else _bsnd_to_ntd(k)
                v_head = _bsnd_to_ntd(v)
                beta_head = beta.squeeze(0).contiguous()
                do_head = do.squeeze(0).contiguous()
                raw_g_head = g_input.squeeze(0).contiguous() if ctx.use_gate_in_kernel else None
                gk_head = _varlen_saved_tensor(g_cumsum)
                Aqk_head = _varlen_saved_tensor(Aqk)
                Akk_head = _varlen_saved_tensor(Akk)
                w_head = _varlen_saved_tensor(w)
                qg_head = _varlen_saved_tensor(qg)
                kg_head = _varlen_saved_tensor(kg)
                v_new_head = _varlen_saved_tensor(v_new)
                h_head = _varlen_saved_tensor(h)
            else:
                q_head = None if ctx.use_qk_l2norm_in_kernel else _bsnd_to_bnsd(q)
                k_head = None if ctx.use_qk_l2norm_in_kernel else _bsnd_to_bnsd(k)
                v_head = _bsnd_to_bnsd(v)
                beta_head = beta.contiguous()
                do_head = do.contiguous()
                raw_g_head = g_input.contiguous() if ctx.use_gate_in_kernel else None
                gk_head, Aqk_head, Akk_head = g_cumsum, Aqk, Akk
                w_head, qg_head, kg_head = w, qg, kg
                v_new_head, h_head = v_new, h

            norm_kwargs = {}

            def saved_head(x):
                return x.squeeze(0) if is_varlen else x

            if ctx.use_qk_l2norm_in_kernel and ctx.platform == "a5":
                q_head, k_head = saved_head(v2_qh), saved_head(v2_kh)
                norm_kwargs = dict(q_rstd=saved_head(v2_qr), k_rstd=saved_head(v2_kr))
            gate_A_log = A_log if ctx.use_gate_in_kernel else None
            gate_dt_bias = None
            if ctx.use_gate_in_kernel and dt_bias is not None:
                gate_dt_bias = dt_bias.reshape(q.shape[2], q.shape[3]).contiguous()

            if ctx.platform == "a2a3":
                # q/k/beta already contain the external preprocessing results.
                q_head = _bsnd_to_ntd(q) if is_varlen else _bsnd_to_bnsd(q)
                k_head = _bsnd_to_ntd(k) if is_varlen else _bsnd_to_bnsd(k)
                if not ctx.disable_recompute:
                    # Small operators take rank-4 BNSD even for packed inputs;
                    # the fused backward takes rank-3 HTD for packed inputs.
                    caches = _a2a3_recompute(
                        _bsnd_to_bnsd(q),
                        _bsnd_to_bnsd(k),
                        _bsnd_to_bnsd(v),
                        _bsnd_to_bnsd(g_input),
                        _bsh_to_bhs(beta),
                        Akk,
                        A_log=gate_A_log,
                        dt_bias=gate_dt_bias,
                        cu_seqlens=ctx.cu_seqlens_host,
                        use_gate_in_kernel=ctx.use_gate_in_kernel,
                        lower_bound=ctx.lower_bound,
                    )
                    if is_varlen:
                        caches = tuple(_varlen_saved_tensor(x) for x in caches)
                    gk_head, w_head, qg_head, kg_head, v_new_head, h_head = caches
                bwd_kwargs = _legacy_bwd_kwargs()
            else:
                bwd_kwargs = {"implementation": "optimized"}

            dq_head, dk_head, dv_head, db_head, dg_head, dh0, dA, dbias = ascendc_chunk_kda_bwd(
                q_head,
                k_head,
                v_head,
                beta_head,
                gk_head,
                Aqk_head,
                Akk_head,
                w_head,
                qg_head,
                kg_head,
                v_new_head,
                h_head,
                do_head,
                float(ctx.scale),
                raw_g=raw_g_head,
                A_log=gate_A_log,
                dt_bias=gate_dt_bias,
                initial_state=None,
                dht=None,
                cu_seqlens=ctx.cu_seqlens_host,
                chunk_indices=None,
                chunk_size=ctx.chunk_size,
                safe_gate=ctx.safe_gate,
                lower_bound=ctx.lower_bound,
                use_gate_in_kernel=ctx.use_gate_in_kernel,
                disable_recompute=True if ctx.platform == "a2a3" else ctx.disable_recompute,
                use_exp2=True,
                state_v_first=False,
                layout="NTD" if is_varlen else "BNSD",
                **bwd_kwargs,
                **norm_kwargs,
            )

            if is_varlen:
                dq, dk, dv = map(_ntd_to_bsnd, (dq_head, dk_head, dv_head))
                db, dg = db_head.unsqueeze(0), dg_head.unsqueeze(0)
            else:
                dq, dk, dv = map(_bnsd_to_bsnd, (dq_head, dk_head, dv_head))
                db, dg = db_head, dg_head
            if dbias is not None:
                dbias = dbias.reshape(dt_bias.shape)
        else:
            dq, dk, dv, db, dg, dh0, dA, dbias = triton_chunk_kda_bwd(
                q=q,
                k=k,
                v=v,
                beta=beta,
                Aqk=Aqk,
                Akk=Akk,
                scale=ctx.scale,
                initial_state=initial_state,
                do=do,
                dht=dht,
                g=g_cumsum,
                g_org=g_input if ctx.use_gate_in_kernel else None,
                state_v_first=ctx.state_v_first,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=ctx.chunk_size,
                safe_gate=ctx.safe_gate,
                lower_bound=ctx.lower_bound,
                use_gate_in_kernel=ctx.use_gate_in_kernel,
                A_log=A_log,
                dt_bias=dt_bias,
                disable_recompute=ctx.disable_recompute,
                w=w,
                u=u,
                qg=qg,
                kg=kg,
                v_new=v_new,
                h=h,
            )

        if ctx.use_qk_l2norm_in_kernel and ctx.external_norm:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        if ctx.use_beta_sigmoid_in_kernel:
            db = fused_beta_sigmoid_bwd(beta_raw, db, scale=2.0 if ctx.allow_neg_eigval else 1.0)

        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g_input),
            db.to(beta_raw),
            dA,
            dbias,
            None,
            dh0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def chunk_kda_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = True,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
):
    """Run KDA training with platform-specific AscendC backward paths.

    With ``chunk_size=64`` and ``disable_recompute=False``, retain Aqk/Akk
    and necessary inputs/normalization auxiliaries, but reconstruct
    gk/w/u/qg/kg/h/v_new during backward. A5 uses the existing optimized V2
    implementation. A2/A3 uses Triton gate/WY recompute + AscendC FwdH followed by the
    original fused backward with disable_recompute=True (caches now supplied).
    A2/A3 applies L2Norm and beta sigmoid outside the fused KDA operators.
    Dense and packed inputs are supported; no full-forward replay is used.
    A2/A3 recompute requires BF16 K=V=128 and matching q/value head counts.
    It requires the model Triton KDA small operators and an AscendC FwdH wheel; older
    wheel wrapper bugs may require upgrading the wheel.
    ``chunk_size=32`` continues to use the Triton recompute implementation.
    """
    if 'skip_recompute' in kwargs:
        disable_recompute = bool(kwargs.pop('skip_recompute'))
    if kwargs.pop('use_cache', False):
        raise ValueError("The AscendC KDA implementation currently supports training without KV cache only.")
    if initial_state is not None:
        raise ValueError("The AscendC KDA implementation does not support recurrent state inputs.")
    initial_state = None
    output_final_state = False

    if 'transpose_state_layout' in kwargs:
        if state_v_first:
            raise ValueError("Cannot pass both `state_v_first` and the deprecated `transpose_state_layout`.")
        warnings.warn(
            "`transpose_state_layout` is deprecated and renamed to `state_v_first`.",
            DeprecationWarning,
            stacklevel=2,
        )
        state_v_first = kwargs.pop('transpose_state_layout')

    # state_v_first is deprecated in favor of the explicit argument, but we keep
    # the value as False for the AscendC implementation.
    state_v_first = False

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")
        if A_log.dtype != torch.float32:
            A_log = A_log.float()
        if dt_bias is not None and dt_bias.dtype != torch.float32:
            dt_bias = dt_bias.float()

    chunk_size = kwargs.pop("chunk_size", 64)
    if chunk_size not in (32, 64):
        raise ValueError(f"`chunk_size` must be either 32 or 64 for KDA, got {chunk_size}.")

    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")

    # Validate head dimensions for GVA
    B, T, H, K, HV = *q.shape, v.shape[2]
    assert q.shape == k.shape, f"q and k must have the same shape, got q={q.shape} vs k={k.shape}"
    assert K <= 256, f"Currently we only support key headdim <=256 for KDA, got {K}."
    assert HV % H == 0, (
        f"For GVA, num_v_heads (HV={HV}) must be evenly divisible by num_qk_heads (H={H}), but got HV % H = {HV % H}"
    )
    assert g.shape == (B, T, HV, K), f"g must have shape [B, T, HV, K]={[B, T, HV, K]}, got {list(g.shape)}"
    assert beta.shape == (B, T, HV), f"beta must have shape [B, T, HV]={[B, T, HV]}, got {list(beta.shape)}"

    if scale is None:
        scale = K**-0.5
    # AscendC short tails (<16 rows) are not repeatable. Pad each document
    # to a complete chunk; future zero rows cannot affect valid causal outputs.
    # Padding outside the autograd Function also crops input gradients naturally.
    boundaries = _host_int_tuple(cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens)
    boundaries = (0, T) if boundaries is None else boundaries
    lengths = [end - begin for begin, end in zip(boundaries, boundaries[1:])]
    padded_lengths = [((length + chunk_size - 1) // chunk_size) * chunk_size for length in lengths]
    crop_segments = []
    if chunk_size == 64 and lengths != padded_lengths:
        padded_boundaries = [0]
        for length, padded_length in zip(lengths, padded_lengths):
            crop_segments.append((padded_boundaries[-1], length))
            padded_boundaries.append(padded_boundaries[-1] + padded_length)
        def pad_documents(tensor):
            segments = []
            for begin, end, padded_length in zip(boundaries, boundaries[1:], padded_lengths):
                segment = tensor[:, begin:end]
                padding_shape = (tensor.shape[0], padded_length - (end - begin), *tensor.shape[2:])
                segments.extend((segment, tensor.new_zeros(padding_shape)))
            return torch.cat(segments, dim=1)
        q, k, v, g, beta = (pad_documents(tensor) for tensor in (q, k, v, g, beta))
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.new_tensor(padded_boundaries)
            cu_seqlens_cpu = torch.tensor(padded_boundaries, dtype=cu_seqlens.dtype)
    result = ChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
        safe_gate,
        lower_bound,
        chunk_size,
        disable_recompute,
        return_intermediate_states,
    )

    if crop_segments:
        output = torch.cat([result[0][:, begin:begin + length] for begin, length in crop_segments], dim=1)
        return (output, *result[1:])
    return result

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Fused partial RoPE on Ascend950DT, via CANN ``inplace_partial_rotary_mul``.

Provider for the registered ``partial_rope`` operator: it wraps the CANN call and
does not register itself, mirroring ``ops/npu/causal_conv1d_ascendc.py``.  It is
used for the two *large* per-layer calls only (``q`` and the inverse rotation on
``o``); the small ``kv``/``latent`` tensors keep the reference implementation,
because the operator carries a fixed 16 MiB workspace, which is a net loss for a
3-4 MiB slice.

Semantics are identical to the reference: adjacent element pairs
(``rotary_mode="interleave"``), only the trailing ``partial_slice`` channels are
touched, and the write is in place.  ``inverse=True`` is expressed as
``r2 = -sin``, which is exactly what ``freqs_cis.conj()`` does.

Measured at the real V4.1-Flash shapes (1x4096x64x512, rope dim 64, bf16, A5):

    forward    0.139 ms / +128 MiB  ->  0.086 ms / +16 MiB
    backward   1.562 ms / +1152 MiB ->  0.293 ms / +528 MiB

Numerics against the complex reference: 0.6% relative, and the unrotated
channels are bit-exact.  The residual is inherent to the operator: it requires
``r1``/``r2`` in the *input* dtype (bf16), whereas the reference rotates with
fp32 cos/sin.  Verify the loss curve before enabling this by default.
"""

from __future__ import annotations

import torch

__all__ = ["cann_partial_rotary"]

# The operator's gradient kernel writes into the incoming grad_output, which
# saves a 268 MiB copy per call at these shapes.  It is only safe when autograd
# handed us an owned, contiguous tensor, so a non-contiguous gradient is cloned.
# Set to False to always clone (slower, never mutates an incoming gradient).
FUSED_ROTARY_INPLACE_BWD = True

_OPS = None
_OPS_PROBED = False


def _cann_ops():
    """The CANN operator namespace, or None when it is unavailable."""
    global _OPS, _OPS_PROBED
    if not _OPS_PROBED:
        _OPS_PROBED = True
        try:
            from cann_ops_transformer import ops

            _OPS = ops
        except ImportError:
            _OPS = None
    return _OPS


class _CannPartialRotary(torch.autograd.Function):
    """In-place partial RoPE with an explicit fused backward.

    The operator has no autograd formula, so the backward is dispatched to
    ``inplace_partial_rotary_mul_backward``.  That kernel needs only cos/sin and
    the incoming gradient, never the forward input, so nothing else is saved.
    """

    @staticmethod
    def forward(ctx, x, r1, r2, rd):
        ops = _cann_ops()
        ctx.rd = rd
        ctx.save_for_backward(r1, r2)
        # the operator requires 4-D; a 3-D [b, s, d] input rotates through a view
        xin = x if x.dim() == 4 else x.unsqueeze(2)
        ops.inplace_partial_rotary_mul(
            xin,
            r1,
            r2,
            rotary_mode="interleave",
            partial_slice=[x.shape[-1] - rd, x.shape[-1]],
        )
        ctx.mark_dirty(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        r1, r2 = ctx.saved_tensors
        ops = _cann_ops()
        dim = grad_output.shape[-1]
        if grad_output.is_contiguous() and FUSED_ROTARY_INPLACE_BWD:
            grad_x = grad_output
        else:
            grad_x = grad_output.contiguous()
        gin = grad_x if grad_x.dim() == 4 else grad_x.unsqueeze(2)
        # this kernel mutates grad_output in place and returns nothing
        ops.inplace_partial_rotary_mul_backward(
            gin,
            r1,
            r2,
            rotary_mode="interleave",
            partial_slice=[dim - ctx.rd, dim],
        )
        return grad_output if grad_x is grad_output else grad_x, None, None, None


def cann_partial_rotary(
    x: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    rd: int,
) -> torch.Tensor:
    """Rotate the trailing ``rd`` channels of ``x`` in place and return ``x``.

    ``r1``/``r2`` are the interleave-packed ``[1, S, 1, rd]`` coefficients built
    in ``x.dtype``: a dtype mismatch faults on the device instead of raising, and
    a slice wider than the tensor makes ``partial_slice`` negative, so both are
    rejected here rather than left to the operator.
    """
    if rd % 2 or rd > x.shape[-1]:
        raise ValueError(f"invalid rope dim rd={rd} for a tensor with last dim {x.shape[-1]}.")
    if x.shape[1] != r1.shape[1]:
        raise ValueError(f"cos/sin cover {r1.shape[1]} positions but x has {x.shape[1]}.")
    return _CannPartialRotary.apply(x, r1, r2, rd)

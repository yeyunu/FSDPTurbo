# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""RotaryMul with native interleave mode and bounded coefficient caching.

Retain 128-wide kernel alignment without padding when 32/64-wide interleaved
heads (or single-head tokens) can be grouped by reshape. Other layouts retain
padding. Model frequencies/dimensions are unchanged.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.npu.partial_rotary import cann_partial_rotary
from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.ops.cpu.rope import rope_torch
from fsdp_turbo.utils.device import IS_CANN_OPS_TRANSFORMER_AVAILABLE

try:
    import torch_npu
except ImportError:
    torch_npu = None

try:
    from mindspeed_ops.api.triton.inplace_rope import inplace_rope as triton_inplace_rope
except ImportError:
    triton_inplace_rope = None


# Source views are retained to prevent allocator-address reuse giving false
# hits. The byte limit includes retained source storages (conservatively).
_COEFFICIENT_CACHE = OrderedDict()
_CACHE_MAX_ENTRIES = 32
_CACHE_MAX_BYTES = 64 * 1024 * 1024
_cache_bytes = 0


def clear_rope_cache():
    """Release cached coefficients, e.g. when unloading a model."""
    global _cache_bytes
    _COEFFICIENT_CACHE.clear()
    _cache_bytes = 0


def _cache_store(key, sources, built):
    """Store ``built`` under ``key``, evicting LRU entries to respect the budget."""
    global _cache_bytes
    if key is None:
        return
    size = sum(t.numel() * t.element_size() for t in built if t is not None)
    size += sum(t.untyped_storage().nbytes() for t in sources)
    if size > _CACHE_MAX_BYTES:
        return
    while _COEFFICIENT_CACHE and (
        len(_COEFFICIENT_CACHE) >= _CACHE_MAX_ENTRIES or _cache_bytes + size > _CACHE_MAX_BYTES
    ):
        _cache_bytes -= _COEFFICIENT_CACHE.popitem(last=False)[1][5]
    _COEFFICIENT_CACHE[key] = (*sources, *built, size)
    _cache_bytes += size


def _coefficient_key(tensor):
    return (
        tensor.device,
        tensor.dtype,
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor._version,
    )


@torch.inference_mode(False)
@torch.no_grad()
def _build_coefficients(cos, sin, *, interleaved, inverse, grouping=None, dtype=None):
    # Cache entries first built in inference must still be usable by training
    # autograd, which cannot save inference tensors for backward.
    cos, sin = cos.float(), sin.float()
    if inverse:
        sin = -sin
    factor = 64 // cos.shape[-1] if grouping else 1
    if grouping == "sequence":
        cos = cos.reshape(1, cos.shape[1] // factor, 1, 64)
        sin = sin.reshape(1, sin.shape[1] // factor, 1, 64)
    padding = 0 if grouping else (-cos.shape[-1]) % 64
    if padding:
        cos, sin = F.pad(cos, (0, padding), value=1), F.pad(sin, (0, padding))
    if interleaved:
        cos, sin = cos.repeat_interleave(2, -1), sin.repeat_interleave(2, -1)
        if grouping == "heads":
            cos, sin = torch.cat([cos] * factor, -1), torch.cat([sin] * factor, -1)
    else:
        cos, sin = torch.cat((cos, cos), -1), torch.cat((sin, sin), -1)
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos.contiguous(), sin.contiguous(), (-sin).contiguous()


def _pack_coefficients(cos, sin, *, interleaved, inverse, grouping=None, dtype=None):
    key = None
    # No Python cache mutation in compiled graphs. Inference tensors lack a
    # version counter; bypass caching to avoid stale values after mutation.
    if not torch.compiler.is_compiling():
        try:
            key = (
                _coefficient_key(cos),
                _coefficient_key(sin),
                interleaved,
                inverse,
                grouping,
                dtype,
            )
        except RuntimeError:
            pass
    if key is not None and key in _COEFFICIENT_CACHE:
        _COEFFICIENT_CACHE.move_to_end(key)
        return _COEFFICIENT_CACHE[key][2:5]
    packed_cos, packed_sin, backward_sin = _build_coefficients(
        cos,
        sin,
        interleaved=interleaved,
        inverse=inverse,
        grouping=grouping,
        dtype=dtype,
    )
    _cache_store(key, (cos, sin), (packed_cos, packed_sin, backward_sin))
    return packed_cos, packed_sin, backward_sin


class _FixedRotaryMul(torch.autograd.Function):
    """Save only fixed coefficients, including cached transpose-rotation sin."""

    @staticmethod
    def forward(ctx, value, cos, sin, backward_sin, interleaved):
        ctx.save_for_backward(cos, sin, backward_sin)
        ctx.interleaved = interleaved
        return torch_npu.npu_rotary_mul(
            value,
            cos,
            sin,
            rotary_mode="interleave" if interleaved else "half",
        )

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin, backward_sin = ctx.saved_tensors
        # The transpose of the rotation negates sin, even for non-unit
        # coefficients. Reuse this Function to support higher-order gradients.
        grad = _FixedRotaryMul.apply(
            grad_output.contiguous(),
            cos,
            backward_sin,
            sin,
            ctx.interleaved,
        )
        return grad, None, None, None, None


def _supports_fused_layout(shape, coeff_shape, interleaved):
    """Conservative RotaryMul limits for the normalized BSND layout."""
    batch, sequence, heads, dim = shape
    padded_dim = ((dim + 127) // 128) * 128
    if min(shape) == 0 or padded_dim >= 896 or batch >= 1000 or heads >= 1000:
        return False
    if interleaved:
        return batch * heads < 1000 and coeff_shape == (1, sequence, 1, dim // 2)
    return coeff_shape in (
        (1, sequence, 1, dim // 2),
        (batch, sequence, 1, dim // 2),
        (batch, sequence, heads, dim // 2),
    )


def _rotary_mul_adapter(x, cos, sin, *, interleaved, inverse):
    original_cos, original_sin = cos, sin
    value = x
    if x.ndim == 3:
        value = value.unsqueeze(2)
        cos = cos.reshape((1,) * (3 - cos.ndim) + cos.shape).unsqueeze(-2)
        sin = sin.reshape((1,) * (3 - sin.ndim) + sin.shape).unsqueeze(-2)
    else:
        cos = cos.reshape((1,) * (4 - cos.ndim) + cos.shape)
        sin = sin.reshape((1,) * (4 - sin.ndim) + sin.shape)
    if not _supports_fused_layout(tuple(value.shape), tuple(cos.shape), interleaved):
        return rope_torch(
            x,
            original_cos,
            original_sin,
            interleaved=interleaved,
            inverse=inverse,
        )
    value = value.float()
    half = x.shape[-1] // 2
    padding = (-half) % 64
    grouping = None
    factor = 128 // x.shape[-1] if x.shape[-1] in (32, 64) else 1
    if interleaved and factor > 1:
        if value.shape[2] % factor == 0:
            grouping = "heads"
        elif value.shape[2] == 1 and value.shape[1] % factor == 0:
            grouping = "sequence"
    cos, sin, backward_sin = _pack_coefficients(
        cos,
        sin,
        interleaved=interleaved,
        inverse=inverse,
        grouping=grouping,
    )
    if interleaved:
        # Pair adjacent heads (same position) or single-head tokens (with
        # individually packed frequencies). Independent rotary pairs never
        # cross a head/token boundary. No padding or gradient slicing needed.
        if grouping == "heads":
            packed = value.reshape(value.shape[0], value.shape[1], value.shape[2] // factor, 128)
        elif grouping == "sequence":
            packed = value.reshape(value.shape[0], value.shape[1] // factor, 1, 128)
        else:
            packed = F.pad(value, (0, 2 * padding)) if padding else value.contiguous()
        result = _FixedRotaryMul.apply(packed.contiguous(), cos, sin, backward_sin, True)
        result = result.reshape(value.shape) if grouping else result[..., : x.shape[-1]]
    else:
        if padding:
            left, right = value.chunk(2, -1)
            packed = torch.cat((F.pad(left, (0, padding)), F.pad(right, (0, padding))), -1)
        else:
            packed = value.contiguous()
        result = _FixedRotaryMul.apply(packed, cos, sin, backward_sin, False)
        if padding:
            left, right = result.chunk(2, -1)
            result = torch.cat((left[..., :half], right[..., :half]), -1)
    if x.ndim == 3:
        result = result.squeeze(2)
    return result.to(x.dtype)


@register_op("rope", "npu")
def rope_npu(x, cos, sin, *, interleaved=False, inverse=False):
    if x.device.type != "npu":
        raise ValueError("Fused RoPE requires NPU tensors")
    if torch_npu is None or not hasattr(torch_npu, "npu_rotary_mul"):
        raise RuntimeError("Fused RoPE requires torch_npu.npu_rotary_mul")
    return _rotary_mul_adapter(x, cos, sin, interleaved=interleaved, inverse=inverse)


class _FixedCoefficientInplaceRope(torch.autograd.Function):
    """Use the public Triton rotation without retaining its rotary input."""

    @staticmethod
    def forward(ctx, value, cos, sin):
        ctx.save_for_backward(cos, sin)
        return triton_inplace_rope(value, cos, sin)

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        # The kernel mutates its input. Never overwrite a gradient shared with
        # another branch, including a contiguous upstream gradient.
        private_grad = grad_output.clone(memory_format=torch.contiguous_format)
        return _FixedCoefficientInplaceRope.apply(private_grad, cos, -sin), None, None


@torch.inference_mode(False)
@torch.no_grad()
def _pack_partial_coefficients(cos, sin, *, inverse, dtype):
    """``(r1, r2)`` for ``inplace_partial_rotary_mul``, from ``[S, rd/2]`` rows.

    That operator's D axis must equal the rope width, so this layout gets none of
    the 128-wide alignment padding ``_build_coefficients`` applies for RotaryMul:
    adjacent pairs only, duplicated to cover each pair. It shares
    ``_COEFFICIENT_CACHE`` -- same key function (so storage reuse and ``_version``
    staleness behave identically) and the same byte budget.

    As in ``_build_coefficients``, the cached entries must be ordinary tensors:
    coefficients first built during inference are later saved for backward when
    the same model resumes training.
    """
    key = None
    if not torch.compiler.is_compiling():
        try:
            key = (_coefficient_key(cos), _coefficient_key(sin), 'partial', inverse, dtype)
        except RuntimeError:
            pass
    if key is not None and key in _COEFFICIENT_CACHE:
        _COEFFICIENT_CACHE.move_to_end(key)
        return _COEFFICIENT_CACHE[key][2:4]

    sequence, half = cos.shape
    cos, sin = cos.float(), sin.float()
    if inverse:
        sin = -sin
    width = 2 * half
    r1 = cos.repeat_interleave(2, -1).reshape(1, sequence, 1, width).to(dtype).contiguous()
    r2 = sin.repeat_interleave(2, -1).reshape(1, sequence, 1, width).to(dtype).contiguous()
    _cache_store(key, (cos, sin), (r1, r2, None))
    return r1, r2


def _broadcast_coefficients(cos, sin, rope_x):
    """Give ``rope_npu``/``rope_torch`` the broadcast shape the ``rope`` wrapper produces."""
    if cos.ndim != 2:
        return cos, sin
    shape = (1, cos.shape[0], *([1] if rope_x.ndim == 4 else []), cos.shape[-1])
    return cos.reshape(shape), sin.reshape(shape)


def _partial_coefficient_pairs(cos, sin, sequence):
    """Flatten a supported coefficient layout to one ``[S, rd/2]`` row per position.

    ``partial_rope`` accepts what ``inplace_rope`` accepts: the documented
    ``[S, rd/2]`` rows and the canonical ``[1, S, 1, rd/2]`` broadcast the ``rope``
    wrapper builds from them. The fused operators index coefficients by position,
    so any other broadcastable layout returns None and the caller falls back to
    the reference, which broadcasts on its own.
    """
    if cos.ndim == 2:
        pairs = cos, sin
    elif cos.ndim == 4 and cos.shape[0] == 1 and cos.shape[2] == 1:
        pairs = cos.reshape(cos.shape[1], cos.shape[-1]), sin.reshape(sin.shape[1], sin.shape[-1])
    else:
        return None
    positions = pairs[0].shape[0]
    if positions == sequence:
        return pairs
    if positions == 1:
        return None
    raise ValueError(f"cos/sin cover {positions} positions but x has {sequence}.")


@register_op("partial_rope", "npu")
def partial_rope_npu(x, cos, sin, *, inverse=False):
    """In-place RoPE over the trailing block of q/o, best backend first.

    Order: CANN ``inplace_partial_rotary_mul`` (fused backward, fixed workspace),
    then RotaryMul, then the Triton inplace kernel, then the eager reference.
    Kept separate from ``inplace_rope`` on purpose: reordering that one would move
    every kv/latent/indexer/context-parallel caller with it.  The fused kernels
    take one ``[S, rd/2]`` coefficient row per position, so a layout they cannot
    index by position (see ``_partial_coefficient_pairs``) skips them and is
    broadcast by the reference instead.
    """
    rotary_dim = 2 * cos.shape[-1]
    rope_x = x[..., -rotary_dim:]
    pairs = _partial_coefficient_pairs(cos, sin, rope_x.shape[1])

    if pairs is not None and IS_CANN_OPS_TRANSFORMER_AVAILABLE and x.is_contiguous():
        r1, r2 = _pack_partial_coefficients(*pairs, inverse=inverse, dtype=x.dtype)
        return cann_partial_rotary(x, r1, r2, rotary_dim)

    if torch_npu is not None and hasattr(torch_npu, "npu_rotary_mul"):
        coeff_cos, coeff_sin = _broadcast_coefficients(cos, sin, rope_x)
        x[..., -rotary_dim:] = rope_npu(rope_x, coeff_cos, coeff_sin, interleaved=True, inverse=inverse)
        return x

    if pairs is not None and triton_inplace_rope is not None and x.is_contiguous():
        rows_cos, rows_sin = pairs
        if inverse:
            rows_sin = -rows_sin
        x4 = x if x.ndim == 4 else x.unsqueeze(-2)
        cos4 = rows_cos.expand(x4.size(0), x4.size(1), rows_cos.size(-1)).contiguous()
        sin4 = rows_sin.expand(x4.size(0), x4.size(1), rows_sin.size(-1)).contiguous()
        rotated = triton_inplace_rope(x4, cos4, sin4)
        return rotated if x.ndim == 4 else rotated.squeeze(-2)

    coeff_cos, coeff_sin = _broadcast_coefficients(cos, sin, rope_x)
    x[..., -rotary_dim:] = rope_torch(rope_x, coeff_cos, coeff_sin, interleaved=True, inverse=inverse)
    return x


@register_op("inplace_rope", "npu")
def inplace_rope_npu(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
    inverse: bool = False,
    inplace: bool = False,
) -> torch.Tensor:
    """Triton inplace RoPE over the trailing block of a full [nope | rope] head.

    Only the last ``2 * cos.shape[-1]`` columns rotate, written in place by the
    triton kernel so no ViewCopy/Slice is needed. The triton path mutates x's
    storage but returns a fresh autograd-tracked tensor, so callers must use the
    return value (``x = inplace_rope(...)``) for gradients to flow through
    `inplace_rope_bwd`. When the Triton kernel cannot be used, the eager CANN
    `rope` op runs over a view of the trailing block.
    """
    rotary_dim = 2 * cos.shape[-1]
    if inplace and interleaved and triton_inplace_rope is not None and x.is_contiguous():
        if inverse:
            sin = -sin
        x4 = x if x.ndim == 4 else x.unsqueeze(-2)
        cos4 = cos.expand(x4.size(0), x4.size(1), cos.size(-1)).contiguous()
        sin4 = sin.expand(x4.size(0), x4.size(1), sin.size(-1)).contiguous()
        if torch.is_grad_enabled() and x4.requires_grad and not (cos4.requires_grad or sin4.requires_grad):
            rotated = _FixedCoefficientInplaceRope.apply(x4, cos4, sin4)
        else:
            rotated = triton_inplace_rope(x4, cos4, sin4)
        return rotated if x.ndim == 4 else rotated.squeeze(-2)
    rope_x = x[..., -rotary_dim:]
    if cos.ndim == 2:
        # rope_npu expects the broadcast shape produced by the `rope` wrapper.
        coeff_shape = (1, cos.shape[0], *([1] if rope_x.ndim == 4 else []), cos.shape[-1])
        cos = cos.reshape(coeff_shape)
        sin = sin.reshape(coeff_shape)
    rotated = rope_npu(rope_x, cos, sin, interleaved=interleaved, inverse=inverse)
    if inplace:
        x[..., -rotary_dim:] = rotated
        return x
    out = x.clone()
    out[..., -rotary_dim:] = rotated
    return out

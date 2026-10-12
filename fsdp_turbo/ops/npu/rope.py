"""RotaryMul with native interleave mode and bounded coefficient caching.

Retain 128-wide kernel alignment without padding when 32/64-wide interleaved
heads (or single-head tokens) can be grouped by reshape. Other layouts retain
padding. Model frequencies/dimensions are unchanged.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.ops.cpu.rope import rope_torch

try:
    import torch_npu
except ImportError:
    torch_npu = None


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
def _build_coefficients(cos, sin, *, interleaved, inverse, grouping=None):
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
    return cos.contiguous(), sin.contiguous(), (-sin).contiguous()


def _pack_coefficients(cos, sin, *, interleaved, inverse, grouping=None):
    global _cache_bytes
    key = None
    # No Python cache mutation in compiled graphs. Inference tensors lack a
    # version counter; bypass caching to avoid stale values after mutation.
    if not torch.compiler.is_compiling():
        try:
            key = (_coefficient_key(cos), _coefficient_key(sin), interleaved, inverse, grouping)
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
    )
    if key is not None:
        size = sum(t.numel() * t.element_size() for t in (packed_cos, packed_sin, backward_sin))
        size += cos.untyped_storage().nbytes() + sin.untyped_storage().nbytes()
        if size <= _CACHE_MAX_BYTES:
            while _COEFFICIENT_CACHE and (
                len(_COEFFICIENT_CACHE) >= _CACHE_MAX_ENTRIES or _cache_bytes + size > _CACHE_MAX_BYTES
            ):
                _cache_bytes -= _COEFFICIENT_CACHE.popitem(last=False)[1][5]
            _COEFFICIENT_CACHE[key] = (cos, sin, packed_cos, packed_sin, backward_sin, size)
            _cache_bytes += size
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

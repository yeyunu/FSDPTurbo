# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import math

import torch

from fsdp_turbo.ops.registry import register_op
from functools import lru_cache

FP8_E4M3_MAX = 448.0
_LOG2_FP8_E4M3_MAX = math.log2(FP8_E4M3_MAX)


def _e4m3fn_quantize(tensor: torch.Tensor) -> torch.Tensor:
    """Round a tensor onto the e4m3 grid (3-bit mantissa, max 448)."""
    tensor_abs = tensor.abs()
    tensor_sign = torch.sign(tensor)
    tensor_abs = tensor_abs.clamp(min=1e-10)  # guard log2(0)
    private_exp = torch.floor(torch.log2(tensor_abs)).clamp(min=-6.0)  # E4M3 min exponent
    quant_step = 2.0 ** (private_exp - 3.0)
    quantized = torch.round(tensor_abs / quant_step) * quant_step
    return tensor_sign * quantized.clamp(0.0, FP8_E4M3_MAX)


@lru_cache(maxsize=None)
def get_lut_256(device):
    e2m1 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float16, device=device
    )
    indices = torch.arange(256, device=device, dtype=torch.int32)
    low = indices & 0x0F
    high = (indices >> 4) & 0x0F
    lut = torch.stack([e2m1[low.long()], e2m1[high.long()]], dim=-1)
    return lut


def _mx_quant_single_axis(x: torch.Tensor, dim: int, block_size: int) -> tuple:
    """e4m3 + per-block shared-exponent quantization along ``dim``.

    Every ``block_size`` elements along ``dim`` share one exponent. Returns
    ``(data, scale_bytes)``: data on the e4m3 grid (in the input dtype) and
    scale_bytes as uint8 e8m0 bytes in interleaved pack [.., ceil(nb/2), 2]
    (bytes hold b == 2^(b-127)). Supports ``dim`` in (-1, 0).
    """
    if dim not in (-1, 0):
        raise NotImplementedError("_mx_quant_single_axis supports only dim=-1 or dim=0")

    transposed = dim != -1
    if transposed:
        x = x.transpose(dim, -1)

    orig_size = x.shape[-1]
    pad_size = (block_size - orig_size % block_size) % block_size
    if pad_size != 0:
        x = torch.nn.functional.pad(x, (0, pad_size))
    x_blocks = x.unflatten(-1, (-1, block_size))

    # shared exponent per block: ceil(log2(amax) - log2(448)), zero for empty
    block_abs_max = x_blocks.abs().amax(dim=-1)
    empty = block_abs_max == 0.0
    safe_amax = block_abs_max + empty.to(x.dtype) * 1e-10
    shared_exp = torch.ceil(torch.log2(safe_amax) - _LOG2_FP8_E4M3_MAX).clamp(-127.0, 127.0)
    shared_exp = torch.where(empty, torch.zeros_like(shared_exp), shared_exp)

    scale = (2.0 ** shared_exp.float()).to(x.dtype).unsqueeze(-1)
    scaled = (x_blocks / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)

    data = _e4m3fn_quantize(scaled).flatten(-2, -1)
    if pad_size != 0:
        data = data.narrow(-1, 0, orig_size)

    # e8m0 bytes (byte b == 2^(b-127)), interleaved pack [.., num_blocks/2, 2]
    num_blocks = shared_exp.shape[-1]
    if num_blocks % 2 != 0:
        shared_exp = torch.nn.functional.pad(shared_exp, (0, 1))  # dummy block
    scale_bytes = (shared_exp + 127.0).to(torch.uint8).reshape(*shared_exp.shape[:-1], -1, 2)

    if transposed:
        data = data.transpose(dim, -1)
        scale_bytes = scale_bytes.transpose(dim, -1)
    return data, scale_bytes


@register_op('dynamic_mx_quant', 'cpu')
def dynamic_mx_quant_cpu(x, axis=-1, dst_type=None, scale_alg=1, block_size=32):
    """Pure-torch e4m3 + per-block shared-exponent quantization (MX format).

    Bit-identical to ``npu_dynamic_mx_quant`` (scale_alg=1, block_size=32)
    on normal ranges (validated by bench_w8a16_fq_vs_npu_mx.py); returns
    ``(data, mxscale)`` with data on the e4m3 grid (in the input dtype) and
    mxscale as uint8 e8m0 bytes in interleaved pack [.., ceil(nb/2), 2].
    """
    if axis != -1:
        raise NotImplementedError("dynamic_mx_quant CPU fallback supports only axis=-1")

    return _mx_quant_single_axis(x, dim=axis, block_size=block_size)


def transform_mxfp4_to_mxfp8(quant_fp4_packed, scale_fp4_packed, mx4_block_size=32, mx8_block_h=128, mx8_block_w=128):
    B, H, W_packed = quant_fp4_packed.shape
    original_W = W_packed * 2
    device = quant_fp4_packed.device

    lut = get_lut_256(device)
    quant_fp4 = lut[quant_fp4_packed.long()].view(B, H, -1)

    total_groups = B * H * (original_W // mx4_block_size)
    scale_flat = scale_fp4_packed.view(-1, 2)
    scale_fp4 = scale_flat.view(-1)[:total_groups].view(B, H, -1)

    num_h_blocks = H // mx8_block_h
    num_w_blocks = original_W // mx8_block_w
    groups_per_block = mx8_block_w // mx4_block_size

    scale_blocks = scale_fp4.view(B, num_h_blocks, mx8_block_h, num_w_blocks, groups_per_block)
    s4_blocks = scale_blocks.unsqueeze(-1).expand(-1, -1, -1, -1, -1, mx4_block_size)
    s4_blocks = s4_blocks.reshape(B, num_h_blocks, mx8_block_h, num_w_blocks, mx8_block_w)
    s4_blocks = s4_blocks.permute(0, 1, 3, 2, 4)

    quant_blocks = quant_fp4.view(B, num_h_blocks, mx8_block_h, num_w_blocks, mx8_block_w)
    quant_blocks = quant_blocks.permute(0, 1, 3, 2, 4)

    s4_max = s4_blocks.amax(dim=(-2, -1))
    s8 = (s4_max - 6).clamp(0, 255).to(torch.uint8)

    exp_diff = s4_blocks - s8[:, :, :, None, None]
    scaled = quant_blocks * torch.exp2(exp_diff.to(torch.float32))
    scaled = scaled.clamp(-448.0, 448.0)
    quant_fp8_blocks = scaled.to(torch.float8_e4m3fn)

    quant_fp8 = quant_fp8_blocks.permute(0, 1, 3, 2, 4).reshape(B, H, original_W)
    return quant_fp8, s8


def expand_scale_to_per_row(scale_2d, block_size=32):
    B, Hb, Wb = scale_2d.shape
    H = Hb * block_size
    scale_expanded = scale_2d[:, :, None, :].expand(-1, -1, block_size, -1)
    scale_per_row = scale_expanded.reshape(B, H, Wb)
    scale_final = scale_per_row.view(B, H, Wb // 2, 2)
    return scale_final


def convert_mx_scale_to_axis2(weight_scale_mx, weight_shape, block_size_old=32):
    B, H_blocks_old, W_blocks_old = weight_scale_mx.shape
    _, H, W = weight_shape
    W = W * 2

    ratio = 2
    H_blocks_new = H_blocks_old // ratio

    reshaped = weight_scale_mx.float().view(B, H_blocks_new, ratio, W_blocks_old)

    scale_top = reshaped[:, :, 0, :]
    scale_bottom = reshaped[:, :, 1, :]

    def expand_width(scale_2d):
        expanded = scale_2d[:, :, :, None].expand(-1, -1, -1, block_size_old)
        return expanded.reshape(B, H_blocks_new, W)

    top_expanded = expand_width(scale_top)
    bottom_expanded = expand_width(scale_bottom)

    scale_final = torch.stack([top_expanded, bottom_expanded], dim=-1)
    return scale_final


@register_op('dynamic_mx_quant_with_dual_axis', 'cpu')
def dynamic_mx_quant_with_dual_axis_cpu(tensor, dst_type=None, block_size=32):
    """Dual-axis (column + row) MX quantization, pure-torch fallback.

    Quantizes along the last dim (columns) and the first dim (rows)
    independently; returns ``(col_data, col_scale, row_data, row_scale)``
    matching the NPU kernel's order.
    """
    col_data, col_scale = _mx_quant_single_axis(tensor, dim=-1, block_size=block_size)
    row_data, row_scale = _mx_quant_single_axis(tensor, dim=0, block_size=block_size)
    return col_data, col_scale, row_data, row_scale


@register_op('grouped_dynamic_mx_quant', 'cpu')
def grouped_dynamic_mx_quant_cpu(tensor, group_list, round_mode='rint', dst_type=None, blocksize=32):
    """Grouped dynamic MX quantization, pure-torch fallback.

    Splits the leading dim into groups per the cumulative sums of
    ``group_list``, quantizing each group independently. Approximate vs the
    NPU kernel's exact rounding and packing.
    """
    boundaries = [0] + list(torch.cumsum(torch.as_tensor(group_list, device=tensor.device), dim=0).tolist())
    group_data, group_scales = [], []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        data, scale = _mx_quant_single_axis(tensor[start:end], dim=-1, block_size=blocksize)
        group_data.append(data)
        group_scales.append(scale)
    return torch.cat(group_data, dim=0), torch.cat(group_scales, dim=0)


@register_op('swiglu', 'cpu')
def swiglu_cpu(x, dim=-1):
    """SiGLU activation: silu(x[..., :D]) * x[..., D:] split along ``dim``."""
    gate, up = x.chunk(2, dim=dim)
    return torch.nn.functional.silu(gate) * up


@register_op('mx_to_block_mx_quant', 'cpu')
def mx_to_block_mx_quant_cpu(weight_mxfp4, weight_mxscale_mxfp4, dst_type, x_type):
    weight_mxfp4_3D = weight_mxfp4
    if weight_mxfp4.dim() == 2:
        weight_mxfp4_3D = weight_mxfp4_3D.unsqueeze(0)

    weight_fwd_fp8, weight_scale_fp8 = transform_mxfp4_to_mxfp8(
        weight_mxfp4_3D, weight_mxscale_mxfp4, mx4_block_size=32, mx8_block_h=32, mx8_block_w=32
    )

    weight_fwd_scale_mxfp8 = expand_scale_to_per_row(weight_scale_fp8, block_size=32).to(torch.uint8)
    weight_bwd_scale_mxfp8 = convert_mx_scale_to_axis2(weight_scale_fp8, weight_mxfp4_3D.shape).to(torch.uint8)

    if weight_mxfp4.dim() == 2:
        weight_fwd_fp8 = weight_fwd_fp8.squeeze(0)
        weight_fwd_scale_mxfp8 = weight_fwd_scale_mxfp8.squeeze(0)
        weight_bwd_scale_mxfp8 = weight_bwd_scale_mxfp8.squeeze(0)

    return weight_fwd_fp8, weight_fwd_scale_mxfp8, weight_bwd_scale_mxfp8

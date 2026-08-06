# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


def get_hifloat8_dtype():
    """Return the NPU hifloat8 dtype, or None when torch_npu is unavailable."""
    if torch_npu is None:
        return None
    return torch_npu.hifloat8


@register_op('dynamic_mx_quant', 'npu')
def dynamic_mx_quant_npu(x, axis=-1, dst_type=None, scale_alg=0, block_size=32):
    """FP8 e4m3 + per-block shared-exponent quantization (MX format).

    Returns ``(data_fp8, mxscale)``: data_fp8 in float8_e4m3fn, mxscale as
    uint8 e8m0 bytes in interleaved pack [.., ceil(nb/2), 2].
    """
    return torch_npu.npu_dynamic_mx_quant(x, axis=axis, dst_type=dst_type, scale_alg=scale_alg, block_size=block_size)


@register_op('dynamic_mx_quant_with_dual_axis', 'npu')
def dynamic_mx_quant_with_dual_axis_npu(tensor, dst_type=None):
    """Dual-axis (column + row) MX quantization: (col_data, col_scale, row_data, row_scale)."""
    return torch_npu.npu_dynamic_mx_quant_with_dual_axis(tensor, dst_type=dst_type)


@register_op('grouped_dynamic_mx_quant', 'npu')
def grouped_dynamic_mx_quant_npu(tensor, group_list, round_mode='rint', dst_type=None, blocksize=32):
    """Grouped dynamic MX quantization for MoE token groups."""
    return torch_npu.npu_grouped_dynamic_mx_quant(
        tensor, group_list, round_mode=round_mode, dst_type=dst_type, blocksize=blocksize
    )


@register_op('swiglu', 'npu')
def swiglu_npu(x, dim=-1):
    """Fused SiGLU activation: silu(x[..., :D]) * x[..., D:] along ``dim``."""
    return torch_npu.npu_swiglu(x, dim=dim)

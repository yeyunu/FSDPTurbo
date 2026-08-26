# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from fsdp_turbo.ops import dispatch_op


def dynamic_mx_quant(x, axis=-1, dst_type=None, scale_alg=0, block_size=32):
    """Dynamic MX quantization (e4m3 data + per-block e8m0 shared exponent).

    scale_alg: 0 = kernel default (MX path), 1 = FP8 block-scaling (W8A16).
    Returns (data, scale_bytes).
    """
    return dispatch_op('dynamic_mx_quant', x, axis=axis, dst_type=dst_type, scale_alg=scale_alg, block_size=block_size)


def dynamic_mx_quant_with_dual_axis(tensor, dst_type=None):
    """Dual-axis (column + row) MX quantization: (col_data, col_scale, row_data, row_scale)."""
    return dispatch_op('dynamic_mx_quant_with_dual_axis', tensor, dst_type=dst_type)


def grouped_dynamic_mx_quant(tensor, group_list, round_mode="rint", dst_type=None, blocksize=32):
    """Grouped dynamic MX quantization for MoE token groups."""
    return dispatch_op(
        'grouped_dynamic_mx_quant',
        tensor,
        group_list,
        round_mode=round_mode,
        dst_type=dst_type,
        blocksize=blocksize,
    )


def swiglu(x, dim=-1):
    """SiGLU activation: silu(x[..., :D]) * x[..., D:] split along ``dim``."""
    return dispatch_op('swiglu', x, dim=dim)


def mx_quant_matmul(input_tensor, weight, scale, pertoken_scale=None, output_dtype=None, group_sizes=None):
    """FP8 (e4m3 data + e8m0 scales) quantized matmul.

    input [M, K] x weight [K, N], scale [num_k_blocks, N] and pertoken_scale
    [M, num_k_blocks] as e8m0 bytes.
    """
    return dispatch_op(
        'mx_quant_matmul',
        input_tensor,
        weight,
        scale,
        pertoken_scale=pertoken_scale,
        output_dtype=output_dtype,
        group_sizes=group_sizes,
    )


def mx_grouped_matmul(
    x,
    weight,
    x_fp8,
    x_scale,
    weight_fwd,
    weight_scale_fwd,
    weight_bwd,
    weight_scale_bwd,
    weight_2d_shape,
    weight_transposed,
    grad_dtype,
    input_dtype,
    x_dtype,
    group_list,
    group_list_type,
    weight_owner=None,
    weight_attr=None,
):
    """FP8 quantized grouped matmul for the MX MoE path."""
    return dispatch_op(
        'mx_grouped_matmul',
        x,
        weight,
        x_fp8,
        x_scale,
        weight_fwd,
        weight_scale_fwd,
        weight_bwd,
        weight_scale_bwd,
        weight_2d_shape,
        weight_transposed,
        grad_dtype,
        input_dtype,
        x_dtype,
        group_list,
        group_list_type,
        weight_owner,
        weight_attr,
    )


def mx_to_block_mx_quant(
    weight_mxfp4,
    weight_mxscale_mxfp4,
    dst_type,
    x_type,
):
    return dispatch_op(
        'mx_to_block_mx_quant',
        weight_mxfp4,
        weight_mxscale_mxfp4,
        dst_type,
        x_type,
    )

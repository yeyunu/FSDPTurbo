# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


@register_op('mx_quant_matmul', 'npu')
def mx_quant_matmul_npu(
    input_tensor,
    weight,
    scale,
    pertoken_scale=None,
    output_dtype=None,
    scale_dtype=None,
    pertoken_scale_dtype=None,
    group_sizes=None,
):
    """FP8 (e4m3 data + e8m0 scales) quantized matmul on Ascend.

    Wraps ``torch_npu.npu_quant_matmul``: input_tensor [M, K] fp8, weight
    [K, N] fp8, scale [num_k_blocks, N] e8m0 bytes, pertoken_scale
    [M, num_k_blocks] e8m0 bytes, ``group_sizes`` per-block group sizes
    (default [1, 1, 32]).
    """
    if scale_dtype is None:
        scale_dtype = torch_npu.float8_e8m0fnu
    if pertoken_scale_dtype is None:
        pertoken_scale_dtype = torch_npu.float8_e8m0fnu
    return torch_npu.npu_quant_matmul(
        input_tensor,
        weight,
        scale,
        pertoken_scale=pertoken_scale,
        output_dtype=output_dtype,
        scale_dtype=scale_dtype,
        pertoken_scale_dtype=pertoken_scale_dtype,
        group_sizes=group_sizes,
    )

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import torch

from fsdp_turbo.ops.registry import register_op


def _unpack_e8m0(scale_bytes: torch.Tensor) -> torch.Tensor:
    """e8m0 字节反量化：byte b == 2^(b-127)。

    兼容 CPU 量化路径的 interleaved pack 布局 [.., nb/2, 2]（解包为 [.., nb]）
    以及非 pack 布局（原样保留）。
    """
    factor = 2.0 ** (scale_bytes.to(torch.float32) - 127.0)
    if factor.shape[-1] == 2 and factor.dim() >= 2:
        factor = factor.reshape(*factor.shape[:-2], -1)
    return factor


@register_op('mx_quant_matmul', 'cpu')
def mx_quant_matmul_cpu(
    input_tensor,
    weight,
    scale,
    pertoken_scale=None,
    output_dtype=None,
    scale_dtype=None,
    pertoken_scale_dtype=None,
    group_sizes=None,
):
    """FP8 (e4m3 data + e8m0 scales) quantized matmul, pure-torch fallback.

    CPU fallback for ``npu_quant_matmul``:
    ``output[m, n] = sum_k input[m, k] * weight[k, n] * scale[k//blk, n]
    * pertoken_scale[m, k//blk]``, ``blk`` from ``group_sizes[-1]`` (default
    32). Layout matches the MX quantizers in ``ops/cpu/mx_quant.py``:
    input_tensor [M, K], weight [K, N], scale [nb, N] e8m0 bytes,
    pertoken_scale [M, nb] e8m0 bytes; pack layouts are unpacked
    transparently.
    """
    block_size = int(group_sizes[-1]) if group_sizes else 32

    input_f = input_tensor.to(torch.float32)
    weight_f = weight.to(torch.float32)

    scale_f = _unpack_e8m0(scale).repeat_interleave(block_size, dim=0)
    if scale_f.shape[0] != weight_f.shape[0]:
        # 量化端奇数块时 pack 会补 dummy 块，截断到真实 K
        scale_f = scale_f[: weight_f.shape[0]]
    weight_f = weight_f * scale_f

    if pertoken_scale is not None:
        pertoken_f = _unpack_e8m0(pertoken_scale).repeat_interleave(block_size, dim=-1)
        if pertoken_f.shape[-1] != input_f.shape[-1]:
            pertoken_f = pertoken_f[..., : input_f.shape[-1]]
        input_f = input_f * pertoken_f

    output = input_f @ weight_f
    if output_dtype is not None:
        output = output.to(output_dtype)
    return output


@register_op('w4a16_quant_matmul', 'cpu')
def w4a16_quant_matmul_cpu(
    x1,
    x2,
    scale,
    pertoken_scale=None,
    bias=None,
    scale_dtype=None,
    pertoken_scale_dtype=None,
    output_dtype=None,
    x1_dtype=None,
    x2_dtype=None,
    group_sizes=None,
):
    """Low-precision (W4A16) quantized matmul CPU fallback.

    W4A16 relies on the NPU low-precision quantized matmul kernel and the
    NPU float4 dtype, so there is no meaningful CPU implementation. This
    fallback keeps the module importable on non-NPU devices and raises a
    clear error if the W4A16 path is actually exercised on CPU.
    """
    raise NotImplementedError(
        "W4A16 low-precision quantized matmul is only supported on NPU. No CPU fallback is available."
    )

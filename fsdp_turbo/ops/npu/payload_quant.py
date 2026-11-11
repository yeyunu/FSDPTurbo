# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Native NPU kernels for swap payload quantization and reconstruction."""

import torch
import torch_npu

from fsdp_turbo.ops.npu.int8_dequant import dequant_int8_per_row
from fsdp_turbo.ops.registry import register_op


@register_op('dynamic_quant', 'npu')
def dynamic_quant_npu(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run NPU per-token int8 dynamic quantization on the caller's stream."""
    return torch_npu.npu_dynamic_quant(tensor)


@register_op('dynamic_block_quant', 'npu')
def dynamic_block_quant_npu(tensor: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Run NPU e4m3fn quantization with 1xN blocks and fp32 scales."""
    return torch_npu.npu_dynamic_block_quant(
        tensor, dst_type=torch.float8_e4m3fn, row_block_size=1, col_block_size=block_size
    )


@register_op('dequant_int8_per_row', 'npu')
def dequant_int8_per_row_npu(payload: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Restore rows through native anti-quant, Triton, then torch fallback."""
    return dequant_int8_per_row(payload, scale, dtype)


@register_op('dequant_fp8_blockwise', 'npu')
def dequant_fp8_blockwise_npu(
    payload: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, block_size: int = 128
) -> torch.Tensor:
    """Restore FP8 payloads through the native per-row anti-quant kernel."""
    output = torch_npu.npu_anti_quant(payload.reshape(-1, block_size), scale.reshape(-1, 1), dst_dtype=dtype)
    return output.reshape(payload.shape)


@register_op('fp8_payload_quant_supported', 'npu')
def fp8_payload_quant_supported_npu() -> bool:
    """Check for Ascend 950PR&950DT products, following MindSpeed-MM."""
    device_name = torch_npu.npu.get_device_name()
    return 'Ascend910_95' in device_name or 'Ascend950' in device_name

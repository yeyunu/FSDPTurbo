# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

from fsdp_turbo.ops.quantize import dynamic_mx_quant_with_dual_axis, mx_to_block_mx_quant
from fsdp_turbo.quantization.utils import TensorWithTranspose, view_as_n_dim


class MXTensor:
    def __init__(self, config):
        self.config = config
        self.fp8_format_dtype = None

    def to_mxfp8(self, data_hf, key):
        if data_hf is None:
            return data_hf
        try:
            data_hf = data_hf.to_local()
        except (AttributeError, TypeError):
            pass
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is not None and data_hf.device.type != accelerator.type:
            data_hf = data_hf.to(accelerator)
        ori_dtype = data_hf.dtype

        if data_hf.dtype == torch.float32:
            data_hf = data_hf.to(torch.bfloat16)

        self.fp8_format_dtype = self.config.get_key_dtype(key)
        tensor_2d = view_as_n_dim(data_hf)

        if key == 'weight' and self.config.quant_format == 'W4A8':
            weight_mxfp4, weight_mxscale_mxfp4 = torch_npu.npu_dynamic_mx_quant(
                tensor_2d, axis=-1, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2, block_size=32, scale_alg=0
            )
            weight_fwd_fp8, weight_fwd_scale_mxfp8, weight_bwd_scale_mxfp8 = mx_to_block_mx_quant(
                weight_mxfp4, weight_mxscale_mxfp4, dst_type=torch_npu.float8_e4m3fn, x_type=torch_npu.float4_e2m1fn_x2
            )
            weight_bwd = weight_fwd_fp8.clone()
            return TensorWithTranspose(
                self.fp8_format_dtype,
                weight_fwd_fp8,
                weight_fwd_scale_mxfp8,
                weight_bwd,
                weight_bwd_scale_mxfp8,
                ori_dtype,
            )
        else:
            col_data, col_scale, row_data, row_scale = dynamic_mx_quant_with_dual_axis(
                tensor_2d, dst_type=self.fp8_format_dtype
            )

        return TensorWithTranspose(
            self.fp8_format_dtype,
            col_data,
            col_scale,
            row_data,
            row_scale,
            ori_dtype,
        )

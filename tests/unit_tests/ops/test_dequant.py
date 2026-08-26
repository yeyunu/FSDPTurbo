from fsdp_turbo.ops.registry import get_op

import torch
import torch_npu


def checkMSE(a, b, name):
    a = a.float().cpu()
    b = b.float().cpu()

    mse = torch.mean((a - b) ** 2)
    print(f"{name} MSE: ", mse.item())


def checkShape(a, b, name):
    if a.shape == b.shape:
        print(f"{name}: shape matching")
    else:
        print(f"{name}: shape NOT matching ({a.shape} vs {b.shape})")


x = torch.empty((1024, 4096), dtype=torch.bfloat16).uniform_(-0.03, 0.04)

x_npu = x.npu()

cpu_mx_to_block_mx_quant = get_op("mx_to_block_mx_quant", "cpu")
npu_mx_to_block_mx_quant = get_op("mx_to_block_mx_quant", "npu")


weight_mxfp4, weight_mxscale_mxfp4 = torch_npu.npu_dynamic_mx_quant(
    x_npu, axis=-1, round_mode="rint", dst_type=torch_npu.float4_e2m1fn_x2, block_size=32, scale_alg=0
)

weight_fwd_fp8, weight_fwd_scale_mxfp8, weight_bwd_scale_mxfp8 = npu_mx_to_block_mx_quant(
    weight_mxfp4, weight_mxscale_mxfp4, dst_type=torch_npu.float8_e4m3fn, x_type=torch_npu.float4_e2m1fn_x2
)

weight_mxfp4_cpu = weight_mxfp4.clone().cpu()
weight_mxscale_mxfp4_cpu = weight_mxscale_mxfp4.clone().cpu()

weight_fwd_fp8_cpu, weight_fwd_scale_mxfp8_cpu, weight_bwd_scale_mxfp8_cpu = cpu_mx_to_block_mx_quant(
    weight_mxfp4_cpu.clone(),
    weight_mxscale_mxfp4_cpu.clone(),
    dst_type=torch_npu.float8_e4m3fn,
    x_type=torch_npu.float4_e2m1fn_x2,
)

checkShape(weight_fwd_fp8, weight_fwd_fp8_cpu, "Tensor")
checkShape(weight_fwd_scale_mxfp8, weight_fwd_scale_mxfp8_cpu, "FWD scale")
checkShape(weight_bwd_scale_mxfp8, weight_bwd_scale_mxfp8_cpu, "BWD scale")
checkMSE(weight_fwd_fp8, weight_fwd_fp8_cpu, "Tensor")
checkMSE(weight_fwd_scale_mxfp8, weight_fwd_scale_mxfp8_cpu, "FWD scale")
checkMSE(weight_bwd_scale_mxfp8, weight_bwd_scale_mxfp8_cpu, "BWD scale")

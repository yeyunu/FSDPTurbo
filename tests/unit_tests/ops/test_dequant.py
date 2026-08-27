import pytest
import torch

from fsdp_turbo.ops.registry import get_op

torch_npu = pytest.importorskip("torch_npu")
try:
    import cann_ops_nn
except ImportError:
    cann_ops_nn = None
if cann_ops_nn is None or not hasattr(torch.ops.cann_ops_nn, "mx_to_block_mx_quant"):
    pytest.skip(
        "cann_ops_nn lacks mx_to_block_mx_quant; W4A8 unavailable on this stack",
        allow_module_level=True,
    )


def _rel_mse(a, b):
    err = (a.float().cpu() - b.float().cpu()) ** 2
    scale = a.float().mean().abs() ** 2 + 1e-12
    return torch.mean(err) / scale


def test_mx_to_block_mx_quant_npu_matches_cpu():
    """W4A8 dequant: the NPU and CPU mx_to_block_mx_quant implementations must
    produce the same shapes and near-identical values (MSE-validated).
    """
    x = torch.empty((1024, 4096), dtype=torch.bfloat16).uniform_(-0.03, 0.04).to("npu")

    weight_mxfp4, weight_mxscale_mxfp4 = torch_npu.npu_dynamic_mx_quant(
        x,
        axis=-1,
        round_mode="rint",
        dst_type=torch_npu.float4_e2m1fn_x2,
        block_size=32,
        scale_alg=0,
    )

    npu_out = get_op("mx_to_block_mx_quant", "npu")(
        weight_mxfp4,
        weight_mxscale_mxfp4,
        dst_type=torch.float8_e4m3fn,
        x_type=torch_npu.float4_e2m1fn_x2,
    )
    cpu_out = get_op("mx_to_block_mx_quant", "cpu")(
        weight_mxfp4.clone().cpu(),
        weight_mxscale_mxfp4.clone().cpu(),
        dst_type=torch.float8_e4m3fn,
        x_type=torch_npu.float4_e2m1fn_x2,
    )

    for npu_tensor, cpu_tensor, name in zip(npu_out, cpu_out, ("tensor", "fwd scale", "bwd scale")):
        assert npu_tensor.shape == cpu_tensor.shape, f"{name} shape mismatch"
        assert _rel_mse(npu_tensor, cpu_tensor) < 1e-4, f"{name} diverges between npu and cpu"

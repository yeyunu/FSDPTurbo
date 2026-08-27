import pytest
import torch

import fsdp_turbo.ops.cpu.mx_quant
import fsdp_turbo.ops.cpu.mx_quant_matmul  # noqa: F401

# Dispatch through the public API: on accelerator machines these tests guard
# the NPU/CUDA implementations; without an accelerator the registry falls
# back to the pure-torch CPU implementations (registered below).
from fsdp_turbo.ops.quantize import (
    dynamic_mx_quant,
    dynamic_mx_quant_with_dual_axis,
    grouped_dynamic_mx_quant,
    mx_quant_matmul,
)


def _device():
    """Run inputs on the active accelerator so NPU/CUDA implementations get
    device tensors; plain CPU hosts fall back to CPU tensors.
    """
    accelerator = torch.accelerator.current_accelerator()
    return accelerator.type if accelerator is not None else "cpu"


def _unpack_e8m0(scale_bytes):
    """Interleaved e8m0 pack [.., nb/2, 2] -> [.., nb]; other layouts pass
    through unchanged.
    """
    factor = 2.0 ** (scale_bytes.to(torch.float32) - 127.0)
    if factor.shape[-1] == 2 and factor.dim() >= 2:
        factor = factor.reshape(*factor.shape[:-2], -1)
    return factor


def _dequant(data, scale_bytes, block_size=32):
    scale = _unpack_e8m0(scale_bytes).repeat_interleave(block_size, dim=-1)
    if scale.shape[-1] != data.shape[-1]:
        scale = scale[..., : data.shape[-1]]
    return data.to(torch.float32) * scale


def _matmul_pair(x, w):
    """Build (x_q, x_scale, w_q, w_scale) per backend matmul contract. NPU
    follows the production path verbatim (MXTensor.to_mxfp8 + the mx_linear
    call shape); the CPU fallback uses its unpacked 2D byte layout.
    """
    if _device() != "cpu":
        from fsdp_turbo.quantization.mx_formats.mx_tensor import MXTensor
        from fsdp_turbo.quantization.mxfp8_config import MXFP8LinearConfig

        mx = MXTensor(MXFP8LinearConfig())
        x_mx = mx.to_mxfp8(x, "inputs")
        w_mx = mx.to_mxfp8(w.t().contiguous(), "weight")
        return x_mx.data, x_mx.scale, w_mx.data.t(), w_mx.scale.transpose(0, 1)

    x_q, x_scale = dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
    q = dynamic_mx_quant(w.t().contiguous(), dst_type=torch.float8_e4m3fn)
    return (
        x_q,
        x_scale,
        q[0].t().contiguous(),
        q[1].reshape(w.shape[1], -1).t().contiguous(),
    )


def _max_block_error_bound(x, block_size=32):
    """Per-element e4m3 round-trip error bound: half the block-shared
    quantization step, <= block_abs_max / 16; doubled to /8 so bf16 rounding
    at exponent boundaries (which can shift the step by one band) stays in.
    Pads non-multiple columns to the block grid, then trims back.
    """
    n = x.shape[-1]
    pad = (-n) % block_size
    x_padded = torch.nn.functional.pad(x.abs(), (0, pad))
    bound = x_padded.unflatten(-1, (-1, block_size)).amax(dim=-1).repeat_interleave(block_size, dim=-1) / 8.0
    return bound[..., :n]


def test_dynamic_mx_quant_round_trip_error_bounded():
    torch.manual_seed(0)
    x = (torch.randn(64, 100, device=_device()) * 8).to(torch.bfloat16)  # 100 not a block multiple

    data, scale = dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)

    assert data.shape == x.shape
    # NPU keeps true fp8 data, the CPU fallback stores the e4m3 grid in the
    # input dtype; both round-trip to the same grid values.
    assert data.dtype in (torch.bfloat16, torch.float8_e4m3fn)
    assert scale.dtype == torch.uint8
    assert data.to(torch.float32).abs().max() <= 448.0  # e4m3 max

    x_hat = _dequant(data, scale)
    err = (x_hat - x.to(torch.float32)).abs()
    bound = _max_block_error_bound(x.to(torch.float32))
    assert torch.all(err <= bound * 1.01 + 1e-3)


def test_dual_axis_col_round_trip_error_bounded():
    torch.manual_seed(1)
    x = (torch.randn(16, 48, device=_device()) * 4).to(torch.bfloat16)

    col_data, col_scale, _, _ = dynamic_mx_quant_with_dual_axis(x, dst_type=torch.float8_e4m3fn)

    col_hat = _dequant(col_data, col_scale)
    assert torch.all((col_hat - x.to(torch.float32)).abs() <= _max_block_error_bound(x.to(torch.float32)) * 1.01 + 1e-3)


def test_dual_axis_row_matches_col_quant_of_transpose():
    """Row-axis quantization data must equal column-axis quantization of the
    transposed tensor (identical shared-exponent grid semantics).
    """
    torch.manual_seed(2)
    x = (torch.randn(16, 48, device=_device()) * 4).to(torch.bfloat16)

    _, _, row_data, _ = dynamic_mx_quant_with_dual_axis(x, dst_type=torch.float8_e4m3fn)
    expected_data, _ = dynamic_mx_quant(x.t().contiguous(), axis=-1, dst_type=torch.float8_e4m3fn)

    # compare in float32: fp8 tensors are not supported by torch.equal on NPU
    assert torch.equal(row_data.to(torch.float32), expected_data.t().contiguous().to(torch.float32))


def test_grouped_quant_matches_per_group_single_axis():
    torch.manual_seed(3)
    x = (torch.randn(10, 32, device=_device()) * 4).to(torch.bfloat16)
    group_counts = torch.tensor([3, 7], device=_device(), dtype=torch.int32)  # npu requires int32 indices
    # npu wants cumsum (production contract); the CPU fallback cumsums internally.
    group_arg = torch.cumsum(group_counts, dim=0, dtype=torch.int32) if _device() != "cpu" else group_counts

    data, scale = grouped_dynamic_mx_quant(x, group_arg, dst_type=torch.float8_e4m3fn)

    if _device() == "cpu":
        # CPU fallback: grouped == per-group single-axis quant, concatenated
        expected = torch.cat(
            [
                dynamic_mx_quant(x[:3], dst_type=torch.float8_e4m3fn)[0],
                dynamic_mx_quant(x[3:], dst_type=torch.float8_e4m3fn)[0],
            ]
        )
        expected_scale = torch.cat(
            [
                dynamic_mx_quant(x[:3], dst_type=torch.float8_e4m3fn)[1],
                dynamic_mx_quant(x[3:], dst_type=torch.float8_e4m3fn)[1],
            ]
        )
        assert torch.equal(data.to(torch.float32), expected.to(torch.float32))
        assert torch.equal(scale, expected_scale)
    else:
        # NPU uses per-column exponents (per-token scale), not single-axis equivalence.
        assert data.shape == x.shape
        assert scale.dtype == torch.uint8
        assert data.to(torch.float32).abs().max() <= 448.0


def _pow2(rows, cols):
    return (2 ** torch.randint(-6, 6, (rows, cols), device=_device())).to(torch.bfloat16)


def _random(rows, cols):
    return (torch.randn(rows, cols, device=_device()) * 2).to(torch.bfloat16)


@pytest.mark.parametrize(
    "gen,rtol,atol",
    [
        (_pow2, 1e-2, 1e-2),  # powers of two quantize exactly
        (_random, 0.5, 1.5),  # random values stay within a loose bound
    ],
)
def test_mx_quant_matmul_matches_bf16(gen, rtol, atol):
    torch.manual_seed(4)
    m, k, n = 8, 64, 6
    x = gen(m, k)
    w = gen(k, n)

    x_q, x_scale, w_q, w_scale = _matmul_pair(x, w)
    out = mx_quant_matmul(x_q, w_q, w_scale, pertoken_scale=x_scale, output_dtype=torch.bfloat16)

    torch.testing.assert_close(out, x @ w, rtol=rtol, atol=atol)

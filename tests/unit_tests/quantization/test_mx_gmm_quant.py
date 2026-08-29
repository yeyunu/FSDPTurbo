from contextlib import ExitStack
from unittest.mock import Mock, patch

import pytest
import torch

# Register the pure-torch op fallbacks so real quantization runs without NPU.
import fsdp_turbo.ops.cpu.mx_quant  # noqa: F401
from fsdp_turbo.quantization.core.post_quant_weight import PostQuantWeight
from fsdp_turbo.quantization.mx_formats.mx_gmm import (
    mx_quant_group_gemm,
    weight_quant_gmm,
)
from fsdp_turbo.utils.gmm_layout import to_gmm_3d


def _device():
    """Run tensors on the active accelerator so real quantization dispatches
    to the NPU implementation there; plain CPU hosts use CPU tensors.
    """
    accelerator = torch.accelerator.current_accelerator()
    return accelerator.type if accelerator is not None else "cpu"


def _quantizer_mock(captured):
    def fake_dual_axis(tensor, dst_type=None):
        captured["input"] = tensor
        return (
            tensor.clone(),
            torch.ones_like(tensor),
            tensor.clone(),
            torch.ones_like(tensor),
        )

    return fake_dual_axis


def test_weight_quant_gmm_2d_quantizes_in_op_layout_no_transpose():
    """2D [E*K, N]: the dual-axis quantizer must see [E, K, N] as a plain view
    (no transpose) and outputs must be stored back in the original 2D shape.
    """
    e, k, n = 4, 8, 6
    weight = torch.randn(e * k, n)
    captured = {}

    with patch(
        "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant_with_dual_axis",
        side_effect=_quantizer_mock(captured),
    ):
        weight_fwd, _, weight_bwd, _ = weight_quant_gmm(weight, torch.bfloat16, (-1, k, n), quant_format="E4M3")

    assert torch.equal(captured["input"], weight.view(e, k, -1))  # plain view, no transpose
    assert captured["input"].shape == (e, k, n)
    assert weight_fwd.shape == weight.shape and torch.equal(weight_fwd, weight)
    assert weight_bwd.shape == weight.shape


def test_weight_quant_gmm_3d_transposes_to_op_layout():
    """3D [E, N, K]: the quantizer must transpose to [E, K, N] and store back."""
    e, k, n = 4, 8, 6
    weight = torch.randn(e, n, k)
    captured = {}

    with patch(
        "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant_with_dual_axis",
        side_effect=_quantizer_mock(captured),
    ):
        weight_fwd, _, weight_bwd, _ = weight_quant_gmm(weight, torch.bfloat16, (-1, k, n), quant_format="E4M3")

    assert torch.equal(captured["input"], weight.transpose(1, 2))
    assert weight_fwd.shape == weight.shape and torch.equal(weight_fwd, weight)
    assert weight_bwd.shape == weight.shape


def test_storage_layout_round_trip_preserves_quantized_op_layout():
    """Quantized [E, K, N] stored in the original layout must re-view losslessly:
    to_gmm_3d recovers exactly the tensor the dual-axis quantizer saw.
    """
    e, k, n = 4, 8, 6

    for stored_shape in ((e * k, n), (e, n, k), (e, k, n)):
        captured = {}
        weight = torch.randn(*stored_shape)
        with patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant_with_dual_axis",
            side_effect=_quantizer_mock(captured),
        ):
            weight_fwd, _, weight_bwd, _ = weight_quant_gmm(weight, torch.bfloat16, (-1, k, n), quant_format="E4M3")

        assert torch.equal(to_gmm_3d(weight_fwd, e, k=k), captured["input"])
        assert torch.equal(to_gmm_3d(weight_bwd, e, k=k), captured["input"])


def test_weight_quant_gmm_3d_op_layout_quantizes_without_transpose():
    """3D [E, K, N] (op layout, MindSpeed-MM style): no transpose in or out."""
    e, k, n = 4, 8, 6
    weight = torch.randn(e, k, n)
    captured = {}

    with patch(
        "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant_with_dual_axis",
        side_effect=_quantizer_mock(captured),
    ):
        weight_fwd, _, weight_bwd, _ = weight_quant_gmm(weight, torch.bfloat16, (-1, k, n), quant_format="E4M3")

    assert captured["input"] is not None
    assert captured["input"].shape == (e, k, n)
    assert torch.equal(captured["input"], weight)  # no transpose
    assert weight_fwd.shape == weight.shape and torch.equal(weight_fwd, weight)
    assert weight_bwd.shape == weight.shape and torch.equal(weight_bwd, weight)


def _post_quant_weight(fwd_shape, bwd_shape=None):
    weight_fwd = torch.randn(fwd_shape, dtype=torch.bfloat16)
    scale_fwd = torch.ones(fwd_shape[-1], dtype=torch.bfloat16)
    weight_bwd = torch.randn(bwd_shape, dtype=torch.bfloat16) if bwd_shape else None
    scale_bwd = torch.ones(bwd_shape[-1], dtype=torch.bfloat16) if bwd_shape else None
    return PostQuantWeight(weight_fwd, scale_fwd, weight_bwd, scale_bwd, torch.bfloat16)


def test_post_quant_path_views_fwd_pair_in_gmm_layout():
    """is_post_quant: the stored fwd pair must be converted to [E, K, N] and
    the bwd pair passed as None (resolved live in backward).
    """
    e, k, n = 4, 8, 6
    x = torch.randn(8, k, dtype=torch.bfloat16)
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    pq = _post_quant_weight(fwd_shape=(e * k, n))
    config = Mock()
    config.get_key_dtype.return_value = torch.bfloat16

    with (
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant",
            return_value=(x, torch.ones(8, 1)),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
            return_value=torch.zeros(8, n, dtype=torch.bfloat16),
        ) as grouped_matmul,
    ):
        mx_quant_group_gemm(x, pq, group_list, config, weight_owner=Mock(), weight_attr="down_proj")

    weight_fwd = grouped_matmul.call_args.args[4]
    weight_bwd = grouped_matmul.call_args.args[6]
    assert weight_fwd.shape == (e, k, n)
    assert torch.equal(weight_fwd, pq._weight_fwd.view(e, k, -1))
    assert weight_bwd is None


def test_post_quant_path_keeps_3d_op_layout_storage():
    """is_post_quant with 3D [E, K, N] stored: fwd pair reaches kernel as-is."""
    e, k, n = 4, 8, 6
    x = torch.randn(8, k, dtype=torch.bfloat16)
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    pq = _post_quant_weight(fwd_shape=(e, k, n))
    config = Mock()
    config.get_key_dtype.return_value = torch.bfloat16

    with (
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant",
            return_value=(x, torch.ones(8, 1)),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
            return_value=torch.zeros(8, n, dtype=torch.bfloat16),
        ) as grouped_matmul,
    ):
        mx_quant_group_gemm(x, pq, group_list, config, weight_owner=Mock(), weight_attr="down_proj")

    weight_fwd = grouped_matmul.call_args.args[4]
    assert weight_fwd.shape == (e, k, n)
    assert torch.equal(weight_fwd, pq._weight_fwd)  # stored as-is, no transpose


def test_plain_weight_3d_op_layout_quantizes_without_transpose():
    """else path with 3D [E, K, N] stored: fwd stays [E, K, N], bwd transposed."""
    e, k, n = 4, 8, 6
    x = torch.randn(8, k, dtype=torch.bfloat16)
    weight = torch.randn(e, k, n, dtype=torch.bfloat16)
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    config = Mock()
    config.get_key_dtype.return_value = torch.bfloat16
    captured = {}

    with (
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant",
            return_value=(x, torch.ones(8, 1)),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant_with_dual_axis",
            side_effect=_quantizer_mock(captured),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
            return_value=torch.zeros(8, n, dtype=torch.bfloat16),
        ) as grouped_matmul,
    ):
        mx_quant_group_gemm(x, weight, group_list, config)

    assert captured["input"].shape == (e, k, n)  # no transpose for op layout
    weight_fwd = grouped_matmul.call_args.args[4]
    weight_bwd = grouped_matmul.call_args.args[6]
    assert weight_fwd.shape == (e, k, n)
    assert weight_bwd.shape == (e, n, k)
    assert torch.equal(weight_bwd, weight.transpose(1, 2))


def test_plain_weight_grad_enabled_quantizes_bwd_pair_transposed():
    """else path with grad_enabled: fwd pair [E, K, N] and bwd pair transposed
    to [E, N, K] for the dx matmul.
    """
    e, k, n = 4, 8, 6
    x = torch.randn(8, k, dtype=torch.bfloat16)
    weight = torch.randn(e * k, n, dtype=torch.bfloat16)
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    config = Mock()
    config.get_key_dtype.return_value = torch.bfloat16
    captured = {}

    with (
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant",
            return_value=(x, torch.ones(8, 1)),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant_with_dual_axis",
            side_effect=_quantizer_mock(captured),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
            return_value=torch.zeros(8, n, dtype=torch.bfloat16),
        ) as grouped_matmul,
    ):
        mx_quant_group_gemm(x, weight, group_list, config)

    assert captured["input"].shape == (e, k, n)  # no transpose for 2D
    weight_fwd = grouped_matmul.call_args.args[4]
    weight_bwd = grouped_matmul.call_args.args[6]
    assert weight_fwd.shape == (e, k, n)
    assert weight_bwd.shape == (e, n, k)
    assert torch.equal(weight_bwd, weight.view(e, k, -1).transpose(1, 2))


def test_plain_weight_inference_skips_backward_pair():
    """else path without grad_enabled: only fwd quantization, empty bwd pair."""
    e, k, n = 4, 8, 6
    x = torch.randn(8, k, dtype=torch.bfloat16)
    weight = torch.randn(e * k, n, dtype=torch.bfloat16)
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    config = Mock()
    config.get_key_dtype.return_value = torch.bfloat16

    def fake_quant(tensor, axis=-1, dst_type=None):
        if axis == -2:  # weight quantization in the else branch
            return torch.zeros(e, k, n, dtype=torch.bfloat16), torch.ones(e, k, 1)
        return x, torch.ones(8, 1)  # x quantization

    with (
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant",
            side_effect=fake_quant,
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
            return_value=torch.zeros(8, n, dtype=torch.bfloat16),
        ) as grouped_matmul,
    ):
        mx_quant_group_gemm(x, weight, group_list, config, grad_enabled=False)

    weight_bwd = grouped_matmul.call_args.args[6]
    weight_scale_bwd = grouped_matmul.call_args.args[7]
    assert weight_bwd.numel() == 0
    assert weight_scale_bwd.numel() == 0


@pytest.mark.parametrize("entry", ["ep_forward", "forward"])
def test_mxfp8_gmm_propagates_owner_and_attr(entry):
    """Real MXFP8GMM wiring: both GMMs must carry the module as weight_owner
    and the canonical attribute names (real quantization runs, only the
    kernel/swiglu/permute are mocked).
    """
    from fsdp_turbo.quantization.mx_formats.mx_gmm import MXFP8GMM

    hidden, intermediate, num_experts = 64, 128, 2
    config = Mock()
    config.get_key_dtype.return_value = torch.float8_e4m3fn
    module = MXFP8GMM(
        config=config,
        num_experts=num_experts,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
    )
    module.gate_up_proj = torch.nn.Parameter(
        torch.randn(
            num_experts * hidden,
            intermediate * 2,
            device=_device(),
            dtype=torch.bfloat16,
        )
    )
    module.down_proj = torch.nn.Parameter(
        torch.randn(num_experts * intermediate, hidden, device=_device(), dtype=torch.bfloat16)
    )

    x = torch.randn(4, hidden, device=_device(), dtype=torch.bfloat16)
    fc2_input = torch.randn(4, intermediate, device=_device(), dtype=torch.bfloat16)
    permuted = torch.randn(4, hidden, device=_device(), dtype=torch.bfloat16)
    selected_experts = torch.tensor([0, 0, 1, 1], device=_device())

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "fsdp_turbo.quantization.mx_formats.mx_gmm.swiglu",
                return_value=fc2_input,
            )
        )
        grouped_matmul = stack.enter_context(
            patch(
                "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
                return_value=torch.zeros(4, 64, device=_device(), dtype=torch.bfloat16),
            )
        )
        if entry == "forward":
            stack.enter_context(
                patch(
                    "fsdp_turbo.ops.moe.permute",
                    return_value=(permuted, torch.arange(4)),
                )
            )
            stack.enter_context(
                patch(
                    "fsdp_turbo.ops.moe.unpermute",
                    return_value=torch.zeros(4, hidden, device=_device(), dtype=torch.bfloat16),
                )
            )
            module.forward(
                x,
                selected_experts=selected_experts,
                routing_weights=torch.ones(4, device=_device(), dtype=torch.float32),
            )
        else:
            module.ep_forward(x, [2, 2])

    assert grouped_matmul.call_count == 2
    fc1 = grouped_matmul.call_args_list[0]
    fc2 = grouped_matmul.call_args_list[1]
    assert fc1.args[-2] is module
    assert fc1.args[-1] == "gate_up_proj"
    assert fc2.args[-2] is module
    assert fc2.args[-1] == "down_proj"

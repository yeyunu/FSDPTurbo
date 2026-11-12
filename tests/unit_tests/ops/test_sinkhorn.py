"""Verify mHC device dispatch and fused forward/backward parity with eager."""

from unittest.mock import Mock

import pytest
import torch

from fsdp_turbo.ops import registry
from fsdp_turbo.ops.sinkhorn import hc_split_sinkhorn
from fsdp_turbo.utils.env import env_flag


def _inputs(device, sequence_length, dtype):
    generator = torch.Generator().manual_seed(1109)
    mixes = torch.randn(1, sequence_length, 48, generator=generator)[..., ::2].to(dtype)
    scale = torch.tensor([0.3, 0.7, 1.0])
    base = torch.randn(48, generator=generator)[::2]
    return [tensor.to(device).detach().requires_grad_(True) for tensor in (mixes, scale, base)]


def test_cpu_sinkhorn_uses_input_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU fallback independent of an installed or selected accelerator."""
    backend = Mock(side_effect=AssertionError("CPU inputs must not reach the accelerator backend"))
    monkeypatch.setitem(registry._OPERATOR_REGISTRY["hc_split_sinkhorn"], "meta", backend)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("meta"))
    inputs = _inputs(torch.device("cpu"), 3, torch.float32)
    actual = hc_split_sinkhorn(*inputs)
    expected = hc_split_sinkhorn(*inputs, use_eager=True)
    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)
    backend.assert_not_called()


@pytest.fixture(name="npu_device", scope="module")
def fixture_npu_device():
    """Initialize accelerator compatibility only for explicitly enabled NPU tests."""
    from fsdp_turbo.utils.device import set_accelerator_compatible

    set_accelerator_compatible()
    accelerator = torch.accelerator.current_accelerator()
    assert accelerator is not None and accelerator.type == "npu", "RUN_NPU_TESTS=1 requires an NPU"
    assert torch.accelerator.is_available(), "RUN_NPU_TESTS=1 requires an available NPU"
    return accelerator


@pytest.mark.skipif(not env_flag("RUN_NPU_TESTS"), reason="requires explicit RUN_NPU_TESTS=1 on an idle NPU")
@pytest.mark.parametrize("sequence_length", [1, 63, 64, 65])
@pytest.mark.parametrize("sinkhorn_iters", [1, 20])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6, 1e-2])
def test_npu_sinkhorn_matches_eager(sequence_length, sinkhorn_iters, dtype, eps, npu_device) -> None:
    """Check tail masks, mixed input dtypes and all three output VJPs on NPU."""
    from fsdp_turbo.ops.npu import sinkhorn

    assert registry.get_op("hc_split_sinkhorn", "npu") is sinkhorn.hc_split_sinkhorn_npu
    actual_inputs = _inputs(npu_device, sequence_length, dtype)
    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in actual_inputs]
    actual = hc_split_sinkhorn(*actual_inputs, sinkhorn_iters=sinkhorn_iters, eps=eps)
    expected = hc_split_sinkhorn(*reference_inputs, sinkhorn_iters=sinkhorn_iters, eps=eps, use_eager=True)
    assert all(type(tensor.grad_fn).__name__ == "HcSplitSinkhornFunctionBackward" for tensor in actual)
    tolerance = 2e-5 if dtype == torch.float32 else 8e-3
    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(actual_output, expected_output, atol=tolerance, rtol=tolerance)

    generator = torch.Generator().manual_seed(1110)
    upstream = [torch.randn(tensor.shape, generator=generator).to(tensor) for tensor in actual]
    for branch in range(3):
        gradients = [tensor if index == branch else torch.zeros_like(tensor) for index, tensor in enumerate(upstream)]
        actual_gradients = torch.autograd.grad(actual, actual_inputs, gradients, retain_graph=True)
        expected_gradients = torch.autograd.grad(expected, reference_inputs, gradients, retain_graph=True)
        # Autograd can cast a returned gradient, so inspect the kernel result too.
        kernel_gradients = sinkhorn._hc_split_sinkhorn_backward_kernel(
            *gradients, *[tensor.contiguous() for tensor in actual_inputs], sinkhorn_iters=sinkhorn_iters, eps=eps
        )
        assert [tensor.dtype for tensor in kernel_gradients] == [tensor.dtype for tensor in actual_inputs]
        for actual_gradient, expected_gradient, original in zip(actual_gradients, expected_gradients, actual_inputs):
            assert actual_gradient.dtype == original.dtype
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16])
def test_cpu_sinkhorn_dtype_contract(dtype, parameter_dtype) -> None:
    """Keep output and leaf-gradient dtypes stable for mixed-precision callers."""
    inputs = _inputs(torch.device("cpu"), 3, dtype)
    inputs = [inputs[0], *(tensor.detach().to(parameter_dtype).requires_grad_(True) for tensor in inputs[1:])]
    outputs = hc_split_sinkhorn(*inputs, use_eager=True)
    assert all(tensor.dtype == dtype and tensor.device.type == "cpu" for tensor in outputs)
    gradients = torch.autograd.grad(outputs, inputs, [torch.ones_like(tensor) for tensor in outputs])
    for gradient, original in zip(gradients, inputs):
        assert gradient.dtype == original.dtype
        assert torch.isfinite(gradient).all()


def test_cpu_hyper_connection_matches_transformers() -> None:
    """Compare mHC outputs and input/parameter gradients against Transformers."""
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection

    from fsdp_turbo.modules.hyper_connection import HyperConnection

    config = DeepseekV4Config(hidden_size=8, hc_mult=4, hc_sinkhorn_iters=20)
    generator = torch.Generator().manual_seed(1111)
    actual_module = HyperConnection(config)
    reference_module = DeepseekV4HyperConnection(config)
    with torch.no_grad():
        for parameter in actual_module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
    reference_module.load_state_dict(actual_module.state_dict())
    actual_input = torch.randn(2, 3, 4, 8, generator=generator, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    actual = actual_module(actual_input)
    expected = reference_module(reference_input)
    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(actual_output, expected_output, atol=1e-6, rtol=1e-6)
    upstream = [torch.randn(tensor.shape, generator=generator) for tensor in actual]
    actual_gradients = torch.autograd.grad(actual, (actual_input, *actual_module.parameters()), upstream)
    expected_gradients = torch.autograd.grad(expected, (reference_input, *reference_module.parameters()), upstream)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-6)

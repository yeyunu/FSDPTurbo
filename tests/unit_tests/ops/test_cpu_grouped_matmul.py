import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).parents[3] / "fsdp_turbo" / "ops" / "cpu" / "grouped_matmul.py"
_SPEC = importlib.util.spec_from_file_location("test_cpu_grouped_matmul_impl", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cpu_grouped_matmul = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cpu_grouped_matmul)


def _reference_grouped_matmul(inputs, group_sizes, weights):
    output = inputs.new_zeros((inputs.size(0), weights.size(1)))
    start_index = 0
    for group_index, group_size in enumerate(group_sizes):
        end_index = start_index + group_size
        output[start_index:end_index] = inputs[start_index:end_index] @ weights[group_index].transpose(0, 1)
        start_index = end_index
    return output


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("group_sizes", [(2, 3, 1), (0, 6, 0), (0, 0, 0)])
@pytest.mark.parametrize("needs_input_grad,needs_weight_grad", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_grouped_matmul_output_and_gradients(dtype, group_sizes, needs_input_grad, needs_weight_grad, noncontiguous):
    torch.manual_seed(123)
    token_count = sum(group_sizes)
    if noncontiguous:
        inputs = torch.randn(4, token_count, dtype=dtype).transpose(0, 1)
        weights = torch.randn(3, 4, 5, dtype=dtype).transpose(1, 2)
    else:
        inputs = torch.randn(token_count, 4, dtype=dtype)
        weights = torch.randn(3, 5, 4, dtype=dtype)
    inputs.requires_grad_(needs_input_grad)
    weights.requires_grad_(needs_weight_grad)
    reference_inputs = inputs.detach().clone().requires_grad_(needs_input_grad)
    reference_weights = weights.detach().clone().requires_grad_(needs_weight_grad)
    splits = torch.tensor(group_sizes, dtype=torch.int64)

    output = cpu_grouped_matmul.grouped_matmul_cpu(inputs, splits, weights)
    expected = _reference_grouped_matmul(reference_inputs, group_sizes, reference_weights)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    upstream = torch.randn(5, token_count, dtype=dtype).transpose(0, 1)
    output.backward(upstream)
    expected.backward(upstream)
    if needs_input_grad:
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    else:
        assert inputs.grad is None
    if needs_weight_grad:
        torch.testing.assert_close(weights.grad, reference_weights.grad, rtol=0, atol=0)
    else:
        assert weights.grad is None


def test_grouped_matmul_without_gradients():
    inputs = torch.randn(4, 3)
    weights = torch.randn(2, 5, 3)
    splits = torch.tensor([2, 2])
    with torch.no_grad():
        output = cpu_grouped_matmul.grouped_matmul_cpu(inputs, splits, weights)
    assert not output.requires_grad
    torch.testing.assert_close(output, _reference_grouped_matmul(inputs, [2, 2], weights))


def test_grouped_matmul_gradcheck_and_gradgradcheck():
    inputs = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(3, 2, 2, dtype=torch.float64, requires_grad=True)
    splits = torch.tensor([1, 0, 2])

    def operation(activations, expert_weights):
        return cpu_grouped_matmul.grouped_matmul_cpu(activations, splits, expert_weights)

    assert torch.autograd.gradcheck(operation, (inputs, weights))
    assert torch.autograd.gradgradcheck(operation, (inputs, weights))


def test_grouped_matmul_first_backward_has_no_slice_copy_chain():
    inputs = torch.randn(8, 4, requires_grad=True)
    weights = torch.randn(4, 5, 4, requires_grad=True)
    splits = torch.tensor([2, 0, 6, 0])
    output = cpu_grouped_matmul.grouped_matmul_cpu(inputs, splits, weights)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        output.backward(torch.randn_like(output))
    assert not any("CopySlices" in event.key for event in profile.key_averages())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_matmul_nonreentrant_checkpoint(dtype):
    from torch.utils.checkpoint import checkpoint

    inputs = torch.randn(4, 3, dtype=dtype, requires_grad=True)
    weights = torch.randn(3, 5, 3, dtype=dtype, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    splits = torch.tensor([1, 0, 3])
    output = checkpoint(cpu_grouped_matmul.grouped_matmul_cpu, inputs, splits, weights, use_reentrant=False)
    expected = _reference_grouped_matmul(reference_inputs, [1, 0, 3], reference_weights)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, reference_weights.grad, rtol=0, atol=0)


@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_grouped_matmul_preserves_forward_autocast_in_backward(autocast_enabled):
    torch.manual_seed(123)
    inputs = torch.randn(4, 3, requires_grad=True)
    weights = torch.randn(3, 5, 3, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    splits = torch.tensor([1, 0, 3])
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        output = cpu_grouped_matmul.grouped_matmul_cpu(inputs, splits, weights)
        expected = _reference_grouped_matmul(reference_inputs, [1, 0, 3], reference_weights)
    upstream = torch.randn_like(output)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=not autocast_enabled):
        output.backward(upstream)
        expected.backward(upstream)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, reference_weights.grad, rtol=0, atol=0)

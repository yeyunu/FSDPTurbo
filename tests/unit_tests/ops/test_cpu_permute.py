import importlib.util
from pathlib import Path

import pytest
import torch


_MODULE_PATH = Path(__file__).parents[3] / "fsdp_turbo" / "ops" / "cpu" / "permute.py"
_SPEC = importlib.util.spec_from_file_location("test_cpu_permute_impl", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cpu_permute = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cpu_permute)


def _reference_unpermute(permuted_tokens, sorted_indices, probs=None):
    # permute_cpu returns the inverse permutation: original route -> sorted row.
    route_positions = sorted_indices.long()
    if probs is None:
        return permuted_tokens.index_select(0, route_positions)
    if probs.size(0) == 0:
        return permuted_tokens[:0] * probs.sum(dim=1, keepdim=True)
    token_outputs = []
    for token_index, route_weights in enumerate(probs):
        first_route = token_index * route_weights.numel()
        token_routes = route_positions[first_route : first_route + route_weights.numel()]
        route_values = permuted_tokens.index_select(0, token_routes)
        token_outputs.append((route_values * route_weights[:, None]).sum(dim=0))
    return torch.stack(token_outputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [0, 1, 8])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_unpermute_without_probs_matches_output_and_gradient(dtype, num_tokens, index_dtype):
    activations = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4).to(dtype)
    activations.requires_grad_()
    reference_activations = activations.detach().clone().requires_grad_()
    sorted_indices = torch.arange(num_tokens - 1, -1, -1, dtype=index_dtype)

    output = cpu_permute.unpermute_cpu(activations, sorted_indices)
    expected = _reference_unpermute(reference_activations, sorted_indices)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.dtype == dtype
    output.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(activations.grad, reference_activations.grad, rtol=0, atol=0)


def test_unpermute_without_probs_does_not_allocate_a_reduction_output(monkeypatch):
    activations = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    sorted_indices = torch.arange(7, -1, -1)
    original_zeros = torch.zeros
    allocated_tensors = []

    def record_zeros(*args, **kwargs):
        result = original_zeros(*args, **kwargs)
        allocated_tensors.append(result)
        return result

    monkeypatch.setattr(cpu_permute.torch, "zeros", record_zeros)
    output = cpu_permute.unpermute_cpu(activations, sorted_indices)

    assert len(allocated_tensors) == 1
    assert output is allocated_tensors[0]
    torch.testing.assert_close(output, activations.flip(0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_unpermute_with_probs_preserves_output_and_gradients(dtype, topk):
    num_tokens = 3
    activations = torch.arange(num_tokens * topk * 4, dtype=torch.float32).reshape(-1, 4).to(dtype)
    activations.requires_grad_()
    probs = torch.full((num_tokens, topk), 0.25, dtype=dtype, requires_grad=True)
    reference_activations = activations.detach().clone().requires_grad_()
    reference_probs = probs.detach().clone().requires_grad_()
    sorted_indices = torch.arange(num_tokens * topk - 1, -1, -1)

    output = cpu_permute.unpermute_cpu(activations, sorted_indices, probs)
    expected = _reference_unpermute(reference_activations, sorted_indices, reference_probs)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(activations.grad, reference_activations.grad, rtol=0, atol=0)
    torch.testing.assert_close(probs.grad, reference_probs.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_unpermute_random_routes_matches_output_and_gradients(dtype, topk, index_dtype):
    generator = torch.Generator().manual_seed(456)
    activations = torch.randn(5 * topk, 7, generator=generator).to(dtype).requires_grad_()
    probs = torch.rand(5, topk, generator=generator).to(dtype).requires_grad_()
    sorted_indices = torch.randperm(5 * topk, generator=generator).to(index_dtype)
    reference_activations = activations.detach().clone().requires_grad_()
    reference_probs = probs.detach().clone().requires_grad_()

    output = cpu_permute.unpermute_cpu(activations, sorted_indices, probs)
    expected = _reference_unpermute(reference_activations, sorted_indices, reference_probs)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    upstream = torch.randn(output.shape, generator=generator).to(dtype)
    output.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(activations.grad, reference_activations.grad, rtol=0, atol=0)
    torch.testing.assert_close(probs.grad, reference_probs.grad, rtol=0, atol=0)


@pytest.mark.parametrize("topk", [1, 2, 4])
def test_permute_unpermute_round_trip_matches_weighted_tokens_and_gradients(topk):
    """Use real routing indices to verify the inverse mapping contract end to end."""
    generator = torch.Generator().manual_seed(789)
    tokens = torch.randn(5, 7, generator=generator, requires_grad=True)
    expert_indices = torch.randint(0, 4, (5, topk), generator=generator)
    probs = torch.rand(5, topk, generator=generator, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()
    reference_probs = probs.detach().clone().requires_grad_()

    permuted_tokens, inverse_mapping = cpu_permute.permute_cpu(tokens, expert_indices)
    output = cpu_permute.unpermute_cpu(permuted_tokens, inverse_mapping, probs)
    expected = (reference_tokens.unsqueeze(1) * reference_probs.unsqueeze(-1)).sum(dim=1)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    upstream = torch.randn(output.shape, generator=generator)
    output.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(tokens.grad, reference_tokens.grad)
    torch.testing.assert_close(probs.grad, reference_probs.grad)


@pytest.mark.parametrize("num_tokens", [0, 3])
def test_unpermute_with_probs_preserves_input_dtype_when_reduction_promotes(monkeypatch, num_tokens):
    """NPU autocast promotes bf16 sum(dim=1) to fp32; eager unpermute must cast back.

    The fusion-free path runs on the zero-token rank during MoE combine.  Returning
    fp32 there pollutes the residual stream and makes the next layer's all-to-all
    mix dtypes.  CPU autocast does not promote bf16 reductions, so emulate the NPU
    behavior here to keep the regression covered on CPU CI.  The weights use
    the activation dtype to isolate the reduction promotion; fp32 router
    weights take the same final sum path.
    """
    topk = 2
    dtype = torch.bfloat16
    activations = torch.arange(num_tokens * topk * 4, dtype=torch.float32).reshape(-1, 4).to(dtype)
    probs = torch.full((num_tokens, topk), 0.25, dtype=dtype)
    sorted_indices = torch.arange(num_tokens * topk - 1, -1, -1)
    expected = _reference_unpermute(activations, sorted_indices, probs)

    original_sum = torch.Tensor.sum

    def promoted_sum(self, *args, **kwargs):
        result = original_sum(self, *args, **kwargs)
        return result.float() if self.dtype == torch.bfloat16 else result

    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "sum", promoted_sum)
        output = cpu_permute.unpermute_cpu(activations, sorted_indices, probs)

    assert output.dtype == dtype, f"unpermute returned {output.dtype}, expected {dtype}"
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

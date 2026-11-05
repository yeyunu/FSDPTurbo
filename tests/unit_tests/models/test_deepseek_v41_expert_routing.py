"""Compare native eager routing with direct token/expert arithmetic and gradients."""

import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from fsdp_turbo.ops.moe import permute


@pytest.fixture(name="experts")
def expert_module(monkeypatch):
    # Load the real model leaf without unrelated model exports or Host Engram.
    # Neither the expert helpers nor the semantic permutation op is mocked.
    package_name = "_test_v41_expert_routing"
    package = ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[3] / "fsdp_turbo/models/deepseek/deepseek_v41")]
    monkeypatch.setitem(sys.modules, package_name, package)
    yield importlib.import_module(f"{package_name}.experts")
    for name in (f"{package_name}.experts", f"{package_name}.activation"):
        sys.modules.pop(name, None)


@pytest.mark.parametrize("topk", [1, 3])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_inverse_route_map_preserves_probabilities_and_gradients(experts, topk, index_dtype):
    tokens = torch.arange(20, dtype=torch.float64).reshape(5, 4).requires_grad_()
    indices = torch.tensor([[2, 0, 1], [0, 3, 2], [3, 1, 0], [1, 2, 3], [0, 3, 1]], dtype=index_dtype)[:, :topk]
    probabilities = torch.linspace(0.1, 0.9, 5 * topk, dtype=torch.float64).reshape(5, topk).requires_grad_()
    packed, inverse_map = permute(tokens, indices, use_eager=True)
    packed_to_route = torch.argsort(indices.reshape(-1), stable=True)
    assert not torch.equal(inverse_map, packed_to_route), "must exercise a non-self-inverse map"
    packed_probabilities = experts._reorder_router_weights(probabilities.reshape(-1), inverse_map)
    expert_scale = indices.reshape(-1)[packed_to_route].unsqueeze(-1) + 1
    contributions = (packed * expert_scale + expert_scale.square()) * packed_probabilities.unsqueeze(-1)
    actual = experts._unpermute_preweighted(contributions, inverse_map, probabilities.shape)
    expected = sum(
        (tokens * (indices[:, slot, None] + 1) + (indices[:, slot, None] + 1).square()) * probabilities[:, slot, None]
        for slot in range(topk)
    )
    torch.testing.assert_close(actual, expected)
    upstream = torch.linspace(-1, 1, actual.numel(), dtype=torch.float64).reshape_as(actual)
    actual_grad = torch.autograd.grad(actual, (tokens, probabilities), upstream, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (tokens, probabilities), upstream)
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right)


def test_empty_routes_preserve_shapes_and_gradients(experts):
    tokens = torch.empty(0, 4, dtype=torch.float64, requires_grad=True)
    probabilities = torch.empty(0, 3, dtype=torch.float64, requires_grad=True)
    indices = torch.empty(0, 3, dtype=torch.int64)
    packed, inverse_map = permute(tokens, indices, use_eager=True)
    packed_probabilities = experts._reorder_router_weights(probabilities.reshape(-1), inverse_map)
    result = experts._unpermute_preweighted(packed * packed_probabilities[:, None], inverse_map, probabilities.shape)
    assert result.shape == tokens.shape
    gradients = torch.autograd.grad(result.sum(), (tokens, probabilities))
    for gradient, original in zip(gradients, (tokens, probabilities)):
        torch.testing.assert_close(gradient, torch.zeros_like(original))

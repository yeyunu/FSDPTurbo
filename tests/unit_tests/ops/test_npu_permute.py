import importlib.util
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from fsdp_turbo.ops.cpu.permute import unpermute_cpu


_MODULE_PATH = Path(__file__).parents[3] / "fsdp_turbo" / "ops" / "npu" / "permute.py"
_SPEC = importlib.util.spec_from_file_location("test_npu_permute_impl", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
npu_permute = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(npu_permute)


class _FakeTorchNPU:
    @staticmethod
    def npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs, *, restore_shape=None):
        assert permuted_tokens.dtype == torch.bfloat16
        assert probs.dtype == torch.float32
        return permuted_tokens


def test_unpermute_npu_preserves_activation_dtype(monkeypatch):
    monkeypatch.setattr(npu_permute, "torch_npu", _FakeTorchNPU)
    activations = torch.ones((2, 4), dtype=torch.bfloat16)
    sorted_indices = torch.arange(2, dtype=torch.int32)
    router_probs = torch.ones((2, 1), dtype=torch.float32)

    output = npu_permute.unpermute_npu(activations, sorted_indices, router_probs)

    assert output.dtype == activations.dtype


def _legacy_unpermute(permuted_tokens, sorted_indices, probs=None):
    num_tokens, topk = (permuted_tokens.size(0), 1) if probs is None else (probs.numel(), probs.size(1))
    output = torch.zeros(
        (num_tokens, permuted_tokens.shape[-1]),
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    output.index_copy_(0, sorted_indices.long(), permuted_tokens)
    output = output.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        output *= probs.unsqueeze(-1)
    return output.sum(dim=1)


def _check_eager_unpermute(device, dtype, topk, use_checkpoint, use_autocast, probability_dtype=torch.float32):
    torch.manual_seed(19)
    token_count, hidden_size = 7, 16
    route_count = token_count * (topk or 1)
    activations = torch.randn(route_count, hidden_size, device=device, dtype=dtype, requires_grad=True)
    indices = torch.randperm(route_count, device=device)
    probabilities = (
        torch.rand(token_count, topk, device=device, dtype=probability_dtype, requires_grad=True)
        if topk is not None
        else None
    )
    reference_activations = activations.detach().clone().requires_grad_()
    reference_probabilities = probabilities.detach().clone().requires_grad_() if probabilities is not None else None

    def forward(current_activations, current_probabilities):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_autocast):
            return unpermute_cpu(current_activations, indices, current_probabilities)

    actual = (
        checkpoint(forward, activations, probabilities, use_reentrant=False)
        if use_checkpoint
        else forward(activations, probabilities)
    )
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_autocast):
        expected = _legacy_unpermute(reference_activations, indices, reference_probabilities)
    assert actual.shape == (token_count, hidden_size)
    assert actual.device == activations.device
    assert actual.dtype == (activations.dtype if topk in (None, 1) else expected.dtype)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    gradient = torch.randn_like(expected)
    actual.backward(gradient.to(actual.dtype))
    expected.backward(gradient)
    torch.testing.assert_close(activations.grad, reference_activations.grad, rtol=0, atol=0)
    if probabilities is not None:
        torch.testing.assert_close(probabilities.grad, reference_probabilities.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [None, 1, 6])
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_eager_unpermute_cpu_forward_backward(dtype, topk, use_checkpoint):
    _check_eager_unpermute(torch.device("cpu"), dtype, topk, use_checkpoint, False)


@pytest.mark.parametrize("topk", [None, 1, 6])
def test_eager_unpermute_empty(topk):
    activations = torch.empty((0, 16), dtype=torch.bfloat16, requires_grad=True)
    probabilities = torch.empty((0, topk), requires_grad=True) if topk is not None else None
    actual = unpermute_cpu(activations, torch.empty(0, dtype=torch.int64), probabilities)
    assert actual.shape == (0, 16)
    assert actual.dtype == torch.bfloat16
    actual.backward(torch.empty_like(actual))
    assert activations.grad.shape == activations.shape


@pytest.mark.parametrize("with_probabilities", [False, True])
def test_eager_unpermute_singleton_does_not_reduce(monkeypatch, with_probabilities):
    activations = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.tensor([2, 4, 0, 1, 3])
    probabilities = torch.rand(5, 1, requires_grad=True) if with_probabilities else None

    def fail_sum(*args, **kwargs):
        pytest.fail("topk=1 must not allocate a reduction or autocast temporary")

    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "sum", fail_sum)
        actual = unpermute_cpu(activations, indices, probabilities)
    actual.backward(torch.ones_like(actual))
    assert torch.isfinite(activations.grad).all()


@pytest.mark.parametrize("topk", [None, 1, 6])
@pytest.mark.parametrize("use_checkpoint", [False, True])
@pytest.mark.parametrize("use_autocast", [False, True])
@pytest.mark.parametrize("probability_dtype", [torch.float32, torch.bfloat16])
def test_eager_unpermute_npu_autocast(topk, use_checkpoint, use_autocast, probability_dtype):
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None or accelerator.type != "npu" or not torch.accelerator.is_available():
        pytest.skip("Requires Ascend NPU")
    _check_eager_unpermute(
        torch.device("npu"),
        torch.bfloat16,
        topk,
        use_checkpoint,
        use_autocast,
        probability_dtype,
    )

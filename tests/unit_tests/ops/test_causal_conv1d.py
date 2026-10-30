import torch
import torch.nn.functional as F

from fsdp_turbo.ops.causal_conv1d import causal_conv1d, causal_conv1d_update
from fsdp_turbo.ops.registry import get_op


def test_causal_conv1d_dispatches_to_registered_cpu_op(monkeypatch):
    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cpu'))

    x = torch.randn(2, 5, 3)
    weight = torch.randn(4, 3)
    bias = torch.randn(3)
    residual = torch.randn_like(x)

    output, final_state = causal_conv1d(
        x,
        weight,
        bias=bias,
        residual=residual,
        activation='silu',
        output_final_state=True,
    )

    expected = F.conv1d(
        x.transpose(1, 2),
        weight.transpose(0, 1).unsqueeze(1),
        bias=bias,
        padding=weight.shape[0] - 1,
        groups=x.shape[-1],
    )[..., : x.shape[1]].transpose(1, 2)
    expected = F.silu(expected) + residual

    assert get_op('causal_conv1d', 'cpu').__name__ == 'causal_conv1d_cpu'
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(final_state, x[:, -weight.shape[0] :].transpose(1, 2))


def test_causal_conv1d_accepts_head_num_with_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cpu'))
    x = torch.randn(2, 5, 3)
    weight = torch.randn(4, 3)

    output, final_state = causal_conv1d(x, weight, head_num=1, output_final_state=True)
    expected_output, expected_state = causal_conv1d(x, weight, output_final_state=True)

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(final_state, expected_state)


def test_causal_conv1d_update_dispatches_and_updates_cache(monkeypatch):
    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cpu'))

    x = torch.randn(2, 1, 3)
    cache = torch.randn(2, 3, 4)
    original_cache = cache.clone()
    weight = torch.randn(3, 4)

    output, returned_cache = causal_conv1d_update(x, cache, weight)
    expected_output, expected_cache = causal_conv1d(
        x,
        weight.transpose(0, 1),
        initial_state=original_cache,
        output_final_state=True,
    )

    assert returned_cache is cache
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(cache, expected_cache)


def test_causal_conv1d_supports_packed_variable_length_sequences(monkeypatch):
    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cpu'))

    x = torch.randn(1, 5, 2)
    weight = torch.randn(3, 2)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.long)

    output, final_state = causal_conv1d(
        x,
        weight,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    first_output, first_state = causal_conv1d(x[:, :2], weight, output_final_state=True)
    second_output, second_state = causal_conv1d(x[:, 2:], weight, output_final_state=True)

    torch.testing.assert_close(output, torch.cat((first_output, second_output), dim=1))
    torch.testing.assert_close(final_state, torch.cat((first_state, second_state), dim=0))


def test_causal_conv1d_batched_bfloat16_weight_gradient(monkeypatch):
    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cpu'))
    torch.manual_seed(42)
    inputs = torch.randn(3, 16, 8, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    upstream_gradient = torch.randn_like(inputs)

    output, state = causal_conv1d(inputs, weight, activation='silu')
    reference = F.conv1d(
        reference_inputs.transpose(1, 2),
        reference_weight.transpose(0, 1).unsqueeze(1),
        padding=3,
        groups=8,
    )[:, :, : inputs.shape[1]]
    reference = F.silu(reference).transpose(1, 2)
    output.backward(upstream_gradient)
    reference.backward(upstream_gradient)

    assert state is None
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=0, atol=0)


def test_explicit_backend_selects_eager_and_rejects_ascendc_on_cpu(monkeypatch):
    import importlib
    import pytest
    from types import SimpleNamespace

    module = importlib.import_module('fsdp_turbo.ops.causal_conv1d')
    calls = []
    monkeypatch.setattr(module, 'dispatch_op', lambda *args, **kwargs: calls.append(kwargs) or (None, None))
    simulated_npu = SimpleNamespace(device=SimpleNamespace(type='npu'))
    module.causal_conv1d(simulated_npu, None, implementation='eager')
    assert calls[-1]['device_type'] == 'cpu'
    module.causal_conv1d(simulated_npu, None, implementation='ascendc')
    assert calls[-1]['device_type'] == 'npu'
    assert calls[-1]['implementation'] == 'ascendc'
    with pytest.raises(ValueError, match='requires NPU'):
        module.causal_conv1d(torch.empty(1), None, implementation='ascendc')
    with pytest.raises(ValueError, match='Unsupported'):
        module.causal_conv1d(torch.empty(1), None, implementation='unknown')

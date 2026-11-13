import pytest
import torch

from fsdp_turbo.ops import registry
from fsdp_turbo.ops.kda import chunk_kda
from fsdp_turbo.ops.registry import get_op


@pytest.mark.parametrize("disable_recompute", [None, False, True])
def test_chunk_kda_dispatches_to_registered_cpu_op(monkeypatch, disable_recompute):
    calls = []

    def fake_chunk_kda(*args, **kwargs):
        calls.append((args, kwargs))
        return 'output', 'final_state'

    def unexpected_backend(*args, **kwargs):
        pytest.fail('CPU input must select the CPU backend')

    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cuda'))
    monkeypatch.setitem(registry._OPERATOR_REGISTRY['chunk_kda'], 'cpu', fake_chunk_kda)
    monkeypatch.setitem(registry._OPERATOR_REGISTRY['chunk_kda'], 'cuda', unexpected_backend)

    inputs = tuple(torch.tensor(i) for i in range(5))
    result = chunk_kda(*inputs, output_final_state=True, disable_recompute=disable_recompute)

    expected_kwargs = {"output_final_state": True}
    if disable_recompute is not None:
        expected_kwargs["disable_recompute"] = disable_recompute
    assert result == ('output', 'final_state')
    assert calls == [(inputs, expected_kwargs)]


def test_chunk_kda_has_cpu_fallback():
    assert get_op('chunk_kda', 'cpu').__name__ == 'chunk_kda_cpu'


def test_chunk_kda_cpu_returns_expected_shapes():
    batch_size, sequence_length, head_count, key_dim, value_dim = 1, 3, 1, 2, 2
    q = torch.randn(batch_size, sequence_length, head_count, key_dim)
    k = torch.randn_like(q)
    v = torch.randn(batch_size, sequence_length, head_count, value_dim)
    g = -torch.rand_like(q)
    beta = torch.sigmoid(torch.randn(batch_size, sequence_length, head_count))

    output, final_state, intermediate_states = get_op('chunk_kda', 'cpu')(
        q,
        k,
        v,
        g,
        beta,
        output_final_state=True,
        return_intermediate_states=True,
        chunk_size=2,
    )

    assert output.shape == (batch_size, sequence_length, head_count, value_dim)
    assert final_state.shape == (batch_size, head_count, key_dim, value_dim)
    assert intermediate_states.shape == (batch_size, 2, head_count, key_dim, value_dim)

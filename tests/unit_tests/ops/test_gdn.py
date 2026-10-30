# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from fsdp_turbo.ops import registry
from fsdp_turbo.ops.cpu.gdn import recurrent_gdn_cpu
from fsdp_turbo.ops.gdn import chunk_gdn, fused_recurrent_gdn
from fsdp_turbo.ops.registry import get_op


@pytest.fixture(name='npu_gdn_backends')
def _npu_gdn_backends(monkeypatch):
    # Load the dispatcher without importing unrelated native NPU operators.
    monkeypatch.setattr(registry, 'register_op', lambda *args: lambda operator: operator)
    source = Path(__file__).resolve().parents[3] / 'fsdp_turbo/ops/npu/gdn.py'
    loader = SourceFileLoader('_test_npu_gdn', str(source))
    module = ModuleType(loader.name)
    loader.exec_module(module)

    ascendc = ModuleType('fsdp_turbo.ops.npu.chunk_gdn_ascendc')
    ascendc.chunk_gdn_ascendc = Mock(return_value=(torch.ones(1), None))
    fla = ModuleType('fla.ops.gated_delta_rule')
    fla.chunk_gated_delta_rule = Mock(return_value=(torch.zeros(1), None))
    monkeypatch.setitem(sys.modules, 'fsdp_turbo.ops.npu.chunk_gdn_ascendc', ascendc)
    monkeypatch.setitem(sys.modules, 'fla.ops.gated_delta_rule', fla)
    return module, ascendc, fla


@pytest.mark.parametrize('has_fla, has_mindspeed', [(False, False), (False, True), (True, False), (True, True)])
def test_npu_gdn_auto_requires_both_ascendc_dependencies(monkeypatch, npu_gdn_backends, has_fla, has_mindspeed):
    module, ascendc, fla = npu_gdn_backends
    monkeypatch.setattr(module, 'IS_FLA_NPU_AVAILABLE', has_fla)
    monkeypatch.setattr(module, 'IS_MINDSPEED_OPS_AVAILABLE', has_mindspeed)
    inputs = [torch.randn(1, 4, 2, 8) for _ in range(5)]
    output = module.chunk_gdn_npu(*inputs, scale=0.25, output_final_state=True)
    selected = ascendc.chunk_gdn_ascendc if has_fla and has_mindspeed else fla.chunk_gated_delta_rule
    unselected = fla.chunk_gated_delta_rule if has_fla and has_mindspeed else ascendc.chunk_gdn_ascendc
    assert output is selected.return_value
    selected.assert_called_once()
    unselected.assert_not_called()
    assert selected.call_args.args == tuple(inputs)
    assert selected.call_args.kwargs['scale'] == 0.25
    assert selected.call_args.kwargs['output_final_state'] is True


@pytest.mark.parametrize('has_fla, has_mindspeed', [(False, False), (False, True), (True, False)])
def test_explicit_ascendc_does_not_fall_back_with_missing_dependency(
    monkeypatch, npu_gdn_backends, has_fla, has_mindspeed
):
    module, ascendc, fla = npu_gdn_backends
    monkeypatch.setattr(module, 'IS_FLA_NPU_AVAILABLE', has_fla)
    monkeypatch.setattr(module, 'IS_MINDSPEED_OPS_AVAILABLE', has_mindspeed)
    inputs = [torch.randn(1, 4, 2, 8) for _ in range(5)]
    with pytest.raises(RuntimeError, match='fla_npu and mindspeed_ops'):
        module.chunk_gdn_npu(*inputs, implementation='ascendc')
    ascendc.chunk_gdn_ascendc.assert_not_called()
    fla.chunk_gated_delta_rule.assert_not_called()


def test_chunk_gdn_npu_registry_uses_ascendc_when_available():
    from fsdp_turbo.utils.device import IS_FLA_NPU_AVAILABLE, IS_MINDSPEED_OPS_AVAILABLE

    if not (IS_FLA_NPU_AVAILABLE and IS_MINDSPEED_OPS_AVAILABLE):
        pytest.skip('AscendC GDN backend is not available')

    registered = get_op('chunk_gated_delta_rule', 'npu')
    assert registered.__name__ == 'chunk_gdn_npu'


def test_fused_recurrent_gdn_npu_registry_available():
    from fsdp_turbo.utils.device import IS_FLA_NPU_AVAILABLE

    if not IS_FLA_NPU_AVAILABLE:
        pytest.skip('FLA NPU backend is not available')

    registered = get_op('fused_recurrent_gated_delta_rule', 'npu')
    assert registered.__name__ == 'fused_recurrent_gdn_npu'


def test_chunk_gdn_cuda_registry_available():
    try:
        import fla  # noqa: F401
    except ImportError:
        pytest.skip('FLA (CUDA) backend is not available')

    registered = get_op('chunk_gated_delta_rule', 'cuda')
    assert registered.__name__ == 'chunk_gdn_cuda'


def test_fused_recurrent_gdn_cuda_registry_available():
    try:
        import fla  # noqa: F401
    except ImportError:
        pytest.skip('FLA (CUDA) backend is not available')

    registered = get_op('fused_recurrent_gated_delta_rule', 'cuda')
    assert registered.__name__ == 'fused_recurrent_gdn_cuda'


@pytest.mark.parametrize('length', [1, 17, 65])
@pytest.mark.parametrize('normalize', [False, True])
def test_chunk_matches_recurrence_with_state_gradients(length, normalize):
    torch.manual_seed(42)
    query = torch.randn(2, length, 2, 8) * 0.2
    key = torch.randn_like(query) * 0.2
    value = torch.randn(2, length, 2, 5)
    gates = -torch.rand(2, length, 2)
    beta = torch.rand_like(gates)
    initial_state = torch.randn(2, 2, 8, 5) * 0.1
    inputs = [tensor.requires_grad_() for tensor in (query, key, value, gates, beta, initial_state)]
    options = dict(
        initial_state=initial_state,
        output_final_state=True,
        scale=0.3,
        use_qk_l2norm_in_kernel=normalize,
        implementation='eager',
    )
    chunk_output, chunk_state = chunk_gdn(*inputs[:5], **options)
    recurrent_output, recurrent_state = fused_recurrent_gdn(*inputs[:5], **options)
    torch.testing.assert_close(chunk_output, recurrent_output, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(chunk_state, recurrent_state, atol=1e-6, rtol=1e-4)
    chunk_grads = torch.autograd.grad(chunk_output.square().sum() + chunk_state.square().sum(), inputs)
    recurrent_grads = torch.autograd.grad(recurrent_output.square().sum() + recurrent_state.square().sum(), inputs)
    for actual, expected in zip(chunk_grads, recurrent_grads):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)


def test_packed_sequences_reset_state_and_support_head_first():
    torch.manual_seed(42)
    query, key, value = [torch.randn(1, 83, 2, 8) for _ in range(3)]
    gates, beta = -torch.rand(1, 83, 2), torch.rand(1, 83, 2)
    initial_state = torch.randn(2, 2, 8, 8)
    inputs = (query, key, value, gates, beta)
    options = dict(output_final_state=True, use_qk_l2norm_in_kernel=True, implementation='eager')
    output, state = chunk_gdn(*inputs, initial_state=initial_state, cu_seqlens=torch.tensor([0, 17, 83]), **options)
    for index, (start, end) in enumerate([(0, 17), (17, 83)]):
        expected, expected_state = fused_recurrent_gdn(
            *(tensor[:, start:end] for tensor in inputs), initial_state=initial_state[index : index + 1], **options
        )
        torch.testing.assert_close(output[:, start:end], expected, atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(state[index : index + 1], expected_state, atol=1e-6, rtol=1e-4)
    head_first, _ = chunk_gdn(
        *(tensor.transpose(1, 2) for tensor in inputs),
        initial_state=initial_state,
        cu_seqlens=torch.tensor([0, 17, 83]),
        head_first=True,
        **options,
    )
    torch.testing.assert_close(head_first, output.transpose(1, 2))


@pytest.mark.parametrize('head_first', [False, True])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('output_final_state', [False, True])
def test_recurrent_cpu_layout_and_packed_state(head_first, packed, output_final_state):
    torch.manual_seed(7)
    query, key, value = [torch.randn(1, 7, 2, 4) * 0.2 for _ in range(3)]
    gates, beta = -torch.rand(1, 7, 2), torch.rand(1, 7, 2)
    initial_state = torch.randn(2 if packed else 1, 2, 4, 4) * 0.1
    inputs = [tensor.requires_grad_() for tensor in (query, key, value, gates, beta, initial_state)]
    options = dict(scale=0.4, output_final_state=output_final_state, use_qk_l2norm_in_kernel=True)
    layout_inputs = [tensor.transpose(1, 2) if head_first else tensor for tensor in inputs[:5]]
    output, state = recurrent_gdn_cpu(
        *layout_inputs,
        initial_state=initial_state,
        head_first=head_first,
        cu_seqlens=torch.tensor([0, 3, 7]) if packed else None,
        **options,
    )
    sequences = [(0, 3), (3, 7)] if packed else [(0, 7)]
    reference = [
        recurrent_gdn_cpu(
            *(tensor[:, start:end] for tensor in inputs[:5]),
            initial_state=initial_state[index : index + 1],
            **options,
        )
        for index, (start, end) in enumerate(sequences)
    ]
    expected = torch.cat([result[0] for result in reference], dim=1)
    actual = output.transpose(1, 2) if head_first else output
    torch.testing.assert_close(actual, expected)
    loss, expected_loss = actual.square().sum(), expected.square().sum()
    if output_final_state:
        expected_state = torch.cat([result[1] for result in reference], dim=0)
        torch.testing.assert_close(state, expected_state)
        loss = loss + state.square().sum()
        expected_loss = expected_loss + expected_state.square().sum()
    else:
        assert state is None
    actual_grads = torch.autograd.grad(loss, inputs, retain_graph=True)
    expected_grads = torch.autograd.grad(expected_loss, inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_recurrent_cpu_packed_requires_single_batch():
    inputs = [torch.randn(2, 3, 1, 4) for _ in range(3)]
    gates, beta = torch.randn(2, 3, 1), torch.rand(2, 3, 1)
    with pytest.raises(ValueError, match='batch size one'):
        recurrent_gdn_cpu(*inputs, gates, beta, cu_seqlens=torch.tensor([0, 3]))


def test_recurrent_cpu_packed_without_initial_state_matches_chunk():
    torch.manual_seed(11)
    query, key, value = [torch.randn(1, 7, 2, 4) * 0.2 for _ in range(3)]
    gates, beta = -torch.rand(1, 7, 2), torch.rand(1, 7, 2)
    options = dict(cu_seqlens=torch.tensor([0, 3, 7]), output_final_state=True)
    actual, actual_state = recurrent_gdn_cpu(query, key, value, gates, beta, **options)
    expected, expected_state = chunk_gdn(query, key, value, gates, beta, implementation='eager', **options)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(actual_state, expected_state, atol=1e-6, rtol=1e-4)

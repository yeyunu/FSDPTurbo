"""CPU contracts and explicitly enabled NPU tests for wo_a projection."""

import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from fsdp_turbo.fsdp_turbo_config import DistributedConfig, FSDPTurboConfig, ModelConfig
from fsdp_turbo.ops.attention_projection import attention_output_projection
from fsdp_turbo.ops.registry import _OPERATOR_REGISTRY


class _CPUStorageWithNPUMetadata(torch.Tensor):
    """Simulate NPU routing with CPU storage; no accelerator allocation or kernel."""

    @property
    def device(self):
        return SimpleNamespace(type='npu')


@pytest.mark.parametrize('shape', [(2, 3, 2, 8), (2, 0, 2, 8)])
@pytest.mark.parametrize('use_deepgemm', [False, True])
def test_reference_and_cpu_fallback_gradients(shape, use_deepgemm, monkeypatch):
    """
    Author: tichang
    Date: 2026-10-21
    Description: Preserve reference outputs and gradients, including empty inputs.
    Remarks: CPU synthetic tensors; enabled flag must not require DeepGEMM on CPU.
    """

    def unexpected_backend_import(name):
        pytest.fail(f'Reference execution must not import an optional backend: {name}')

    monkeypatch.setattr('fsdp_turbo.ops.attention_projection.import_module', unexpected_backend_import)
    inputs = torch.randn((*shape[:-1], shape[-1] * 2), dtype=torch.float64)[..., ::2].requires_grad_()
    weight = torch.randn(2, 4, 16, dtype=torch.float64)[..., ::2].requires_grad_()
    expected = torch.einsum('bsgd,grd->bsgr', inputs, weight)
    actual = attention_output_projection(inputs, weight, use_deepgemm=use_deepgemm)
    grad_output = torch.randn(2, shape[1], 4, 2, dtype=torch.float64).transpose(-1, -2)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, (inputs, weight), grad_output),
        torch.autograd.grad(expected, (inputs, weight), grad_output),
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_invalid_contracts_and_boolean_config():
    """
    Author: tichang
    Date: 2026-10-21
    Description: Reject invalid dimensions, dtypes and nonboolean configuration.
    Remarks: No accelerator or external package required.
    """
    with pytest.raises(ValueError, match='B,S,G,D'):
        attention_output_projection(torch.randn(2, 3), torch.randn(2, 4, 3))
    with pytest.raises(ValueError, match='same dtype'):
        attention_output_projection(torch.randn(1, 2, 2, 8), torch.randn(2, 4, 8, dtype=torch.float64))
    with pytest.raises(ValueError, match='use_deepgemm_wo_a must be a boolean'):
        FSDPTurboConfig(model=ModelConfig(use_deepgemm_wo_a='true'))
    assert ModelConfig().use_deepgemm_wo_a is False


def test_deepgemm_rejected_by_native_context_parallelism():
    """
    Author: tichang
    Date: 2026-10-22
    Description: Reject a fused flag that the native CP path would otherwise ignore.
    Remarks: Configuration-only test; no distributed processes or devices needed.
    """
    distributed = DistributedConfig(kv_allgather_parallel_size=2)
    with pytest.raises(ValueError, match='requires all fused flags'):
        FSDPTurboConfig(model=ModelConfig(use_deepgemm_wo_a=True), distributed=distributed)
    FSDPTurboConfig(model=ModelConfig(use_deepgemm_wo_a=False), distributed=distributed)


@pytest.fixture(name='projection_backend')
def load_projection_backend(monkeypatch):
    """Load the NPU adapter in isolation and restore the registry after each test."""
    source = Path(__file__).resolve().parents[3] / 'fsdp_turbo/ops/npu/attention_projection.py'
    monkeypatch.setitem(
        _OPERATOR_REGISTRY, 'attention_output_projection', dict(_OPERATOR_REGISTRY['attention_output_projection'])
    )
    spec = importlib.util.spec_from_file_location('wo_a_cpu_mock_backend', source)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


@pytest.mark.parametrize('rank', [0, 1])
def test_missing_deepgemm_warns_once_and_falls_back(projection_backend, monkeypatch, caplog, rank):
    """
    Author: tichang
    Date: 2026-10-29
    Description: Missing DeepGEMM preserves reference outputs/gradients and warns once on rank zero.
    Remarks: Simulated NPU metadata with real CPU storage; no device execution.
    """

    imports = []

    def missing_dependency(name):
        imports.append(name)
        raise ModuleNotFoundError(name, name=name)

    monkeypatch.setattr(projection_backend, 'import_module', missing_dependency)
    monkeypatch.setattr('fsdp_turbo.ops.attention_projection.import_module', lambda name: projection_backend)
    monkeypatch.setenv('RANK', str(rank))
    caplog.set_level(logging.WARNING, logger=projection_backend.__name__)
    inputs = torch.randn(1, 3, 2, 16, dtype=torch.bfloat16)[..., ::2]
    weight = torch.randn(2, 4, 16, dtype=torch.bfloat16)[..., ::2]
    reference_inputs, reference_weight = inputs.clone().requires_grad_(), weight.clone().requires_grad_()
    inputs = inputs.as_subclass(_CPUStorageWithNPUMetadata).detach().requires_grad_()
    weight = weight.as_subclass(_CPUStorageWithNPUMetadata).detach().requires_grad_()
    for _ in range(2):
        actual = attention_output_projection(inputs, weight, use_deepgemm=True)
    expected = torch.einsum('bsgd,grd->bsgr', reference_inputs, reference_weight)
    grad_output = torch.randn(1, 3, 4, 2, dtype=torch.bfloat16).transpose(-1, -2)
    torch.testing.assert_close(actual.as_subclass(torch.Tensor), expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, (inputs, weight), grad_output),
        torch.autograd.grad(expected, (reference_inputs, reference_weight), grad_output),
    ):
        torch.testing.assert_close(actual_grad.as_subclass(torch.Tensor), expected_grad)
    assert actual.as_subclass(torch.Tensor).device.type == 'cpu'
    assert imports == ['deep_gemm']
    warnings = [record for record in caplog.records if 'falling back to torch.einsum' in record.message]
    assert len(warnings) == (1 if rank == 0 else 0)


@pytest.mark.parametrize(
    'error',
    [
        ModuleNotFoundError('missing extension', name='deep_gemm._C'),
        ModuleNotFoundError('missing dependency', name='broken_dependency'),
        ImportError('incompatible binary'),
    ],
)
def test_broken_deepgemm_imports_are_not_hidden(projection_backend, monkeypatch, error):
    """
    Author: tichang
    Date: 2026-10-29
    Description: Preserve errors inside an installed package instead of hiding them with fallback.
    Remarks: CPU-only import simulation; no accelerator required.
    """

    def broken_import(name):
        raise error

    monkeypatch.setattr(projection_backend, 'import_module', broken_import)
    with pytest.raises(type(error)) as raised:
        projection_backend._get_deepgemm()
    assert raised.value is error


@pytest.mark.parametrize('failed_expression', ['bhr,hdr->bhd', 'bhd,hdr->bhr', 'bhd,bhr->hdr'])
def test_backend_execution_errors_are_not_hidden(projection_backend, monkeypatch, failed_expression):
    """
    Author: tichang
    Date: 2026-10-29
    Description: Propagate forward and backward errors without silently rerunning the reference.
    Remarks: Simulated NPU metadata and CPU contractions; no accelerator required.
    """

    def einsum(expression, left, right, output):
        if expression == failed_expression:
            raise RuntimeError('backend execution failed')
        output.copy_(torch.einsum(expression, left, right))

    def unexpected_fallback(*args):
        pytest.fail('Execution failures must not fall back')

    monkeypatch.setattr(projection_backend, '_get_deepgemm', lambda: SimpleNamespace(einsum=einsum))
    monkeypatch.setattr(projection_backend, 'attention_output_projection_cpu', unexpected_fallback)
    monkeypatch.setattr('fsdp_turbo.ops.attention_projection.import_module', lambda name: projection_backend)
    inputs = torch.randn(1, 3, 2, 8, dtype=torch.bfloat16)
    weight = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    inputs = inputs.as_subclass(_CPUStorageWithNPUMetadata).detach().requires_grad_()
    weight = weight.as_subclass(_CPUStorageWithNPUMetadata).detach().requires_grad_()
    with pytest.raises(RuntimeError, match='backend execution failed'):
        output = attention_output_projection(inputs, weight, use_deepgemm=True)
        torch.autograd.grad(output, (inputs, weight), torch.ones_like(output))


def test_mocked_backend_recompute_and_two_updates(projection_backend, monkeypatch):
    """
    Author: tichang
    Date: 2026-10-21
    Description: Check all contractions, strided dy, saved storage and recompute.
    Remarks: Load just the backend source; substitute CPU contractions for kernels.
    """
    backend = projection_backend
    calls = []

    def einsum(expression, left, right, output):
        calls.append(expression)
        output.copy_(torch.einsum(expression, left, right))

    monkeypatch.setattr(backend, '_get_deepgemm', lambda: SimpleNamespace(einsum=einsum))
    inputs = torch.randn(3, 2, 8, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_()
    first_output = None
    first_value = None
    for _ in range(2):
        output = checkpoint(backend._DeepGemmAttentionProjection.apply, inputs, weight, use_reentrant=False)
        reference = torch.einsum('bhr,hdr->bhd', inputs, reference_weight)
        grad_output = torch.randn(3, 4, 2, dtype=torch.float64).transpose(1, 2)
        if first_output is None:
            first_output, first_value = output, output.detach().clone()
        actual_grads = torch.autograd.grad(output, (inputs, weight), grad_output)
        expected_grads = torch.autograd.grad(reference, (inputs, reference_weight), grad_output)
        torch.testing.assert_close(output, reference)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)
        with torch.no_grad():
            weight.add_(actual_grads[1], alpha=-0.01)
            reference_weight.add_(expected_grads[1], alpha=-0.01)
    torch.testing.assert_close(first_output, first_value)
    assert calls.count('bhr,hdr->bhd') == 4
    assert calls.count('bhd,hdr->bhr') == calls.count('bhd,bhr->hdr') == 2


@pytest.mark.skipif(os.environ.get('RUN_NPU_TESTS') != '1', reason='requires explicit NPU execution')
def test_real_npu_three_way_projection(record_property):
    """
    Author: tichang
    Date: 2026-10-21
    Description: Compare real BF16 NPU projection and gradients to torch and FP32.
    Remarks: Run only behind an eight-card idle gate with compatible DeepGEMM.
    """
    from fsdp_turbo.utils.device import set_accelerator_compatible

    set_accelerator_compatible()
    torch.accelerator.set_device_index(0)
    torch.manual_seed(20261021)
    device = torch.accelerator.current_accelerator()
    inputs = (torch.randn(1, 128, 8, 4096, device=device) * 0.25).bfloat16().requires_grad_()
    weight = (torch.randn(8, 1024, 4096, device=device) * 0.02).bfloat16().requires_grad_()
    grad_output = (torch.randn(1, 128, 1024, 8, device=device) * 0.25).bfloat16().transpose(-1, -2)
    outputs = [attention_output_projection(inputs, weight, use_deepgemm=flag) for flag in (False, True)]
    gradients = [torch.autograd.grad(output, (inputs, weight), grad_output) for output in outputs]
    ref_inputs = inputs.detach().float().requires_grad_()
    ref_weight = weight.detach().float().requires_grad_()
    reference = torch.einsum('bsgd,grd->bsgr', ref_inputs, ref_weight)
    references = [reference, *torch.autograd.grad(reference, (ref_inputs, ref_weight), grad_output.float())]
    for backend_name, output, grads in zip(['torch', 'deepgemm'], outputs, gradients):
        for direction, actual, expected in zip(['forward', 'dgrad', 'wgrad'], (output, *grads), references):
            delta = actual.float() - expected.float()
            relative_l2 = (delta.norm() / expected.float().norm().clamp_min(1e-12)).item()
            normalized_max = (delta.abs().max() / expected.float().abs().max().clamp_min(1e-12)).item()
            record_property(backend_name + '_' + direction + '_relative_l2', relative_l2)
            record_property(backend_name + '_' + direction + '_normalized_max', normalized_max)
            assert torch.isfinite(actual).all()
            assert relative_l2 <= 0.01
            assert normalized_max <= 0.03

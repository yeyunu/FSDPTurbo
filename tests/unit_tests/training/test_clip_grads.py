import importlib.util
import weakref
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).parents[3] / "fsdp_turbo" / "training" / "clip_grads.py"
_SPEC = importlib.util.spec_from_file_location("test_clip_grads_impl", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
clip_grads = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(clip_grads)


@pytest.fixture(autouse=True)
def _use_cpu_for_norm_device(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: torch.device("cpu"))


def _make_parameters(dtype):
    parameters = [torch.nn.Parameter(torch.zeros(3, dtype=dtype)) for _ in range(3)]
    parameters[0].grad = torch.tensor([1, -2, 3], dtype=dtype)
    parameters[1].grad = torch.tensor([-4, 5, 0], dtype=dtype)
    return parameters


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.0, float("inf")])
def test_gradient_norm_matches_fp32_reference(dtype, norm_type):
    parameters = _make_parameters(dtype)
    gradients = torch.cat([parameter.grad.float() for parameter in parameters if parameter.grad is not None])
    expected = torch.linalg.vector_norm(gradients, ord=norm_type).item()
    actual = clip_grads.compute_grad_norm(parameters, norm_type=norm_type)
    assert actual == pytest.approx(expected, rel=1e-6)


def test_gradient_norm_without_gradients():
    assert clip_grads.compute_grad_norm([torch.nn.Parameter(torch.ones(2))]) == 0.0


@pytest.mark.parametrize("max_norm", [1.0, 100.0])
def test_gradient_clipping_matches_reference(max_norm):
    parameters = _make_parameters(torch.bfloat16)
    original_gradients = [parameter.grad.clone() for parameter in parameters if parameter.grad is not None]
    total_norm = clip_grads.compute_grad_norm(parameters)
    clip_grads.clip_grad_by_norm(parameters, max_norm, total_norm)
    coefficient = min(max_norm / (total_norm + 1e-6), 1.0)
    actual_gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    for actual, original in zip(actual_gradients, original_gradients):
        expected = original.clone().mul_(coefficient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.0, float("inf")])
def test_gradient_norm_does_not_retain_all_fp32_copies(monkeypatch, norm_type):
    parameters = _make_parameters(torch.bfloat16)
    references = []
    original_float = torch.Tensor.float

    def tracked_float(tensor, *args, **kwargs):
        assert all(reference() is None for reference in references)
        output = original_float(tensor, *args, **kwargs)
        references.append(weakref.ref(output))
        return output

    monkeypatch.setattr(torch.Tensor, "float", tracked_float)
    clip_grads.compute_grad_norm(parameters, norm_type=norm_type)
    assert len(references) == 2
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.0, float("inf")])
def test_gradient_norm_does_not_retain_all_device_copies(monkeypatch, norm_type):
    parameters = _make_parameters(torch.bfloat16)
    gradients = torch.cat([parameter.grad.float() for parameter in parameters if parameter.grad is not None])
    expected = torch.linalg.vector_norm(gradients, ord=norm_type).item()
    references = []
    original_to = torch.Tensor.to

    def tracked_to(tensor, *args, **kwargs):
        output = original_to(tensor, *args, **kwargs)
        if tensor.dtype == torch.bfloat16 and tensor.ndim == 1:
            assert all(reference() is None for reference in references)
            output = output.clone()
            references.append(weakref.ref(output))
        return output

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)
    actual = clip_grads.compute_grad_norm(parameters, norm_type=norm_type)
    assert actual == pytest.approx(expected, rel=1e-6)
    assert len(references) == 2
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("norm_type", [2.0, float("inf")])
def test_gradient_norm_reduces_each_sharded_group_once(monkeypatch, norm_type):
    parameters = _make_parameters(torch.float32)
    third_parameter = torch.nn.Parameter(torch.ones(1))
    third_parameter.grad = torch.tensor([6.0])
    parameters.append(third_parameter)
    dense_group = object()
    expert_group = object()
    calls = []

    def get_groups(gradient, current_group=None):
        return [expert_group] if gradient is third_parameter.grad else [dense_group]

    def all_reduce(tensor, op, group):
        calls.append((op, group))
        if op == torch.distributed.ReduceOp.SUM:
            tensor.mul_(2)

    monkeypatch.setattr(clip_grads, "_get_shard_groups_if_dtensor", get_groups)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    actual = clip_grads.compute_grad_norm(parameters, norm_type=norm_type)
    expected = (2 * (55 + 36)) ** 0.5 if norm_type == 2.0 else 6.0
    assert actual == pytest.approx(expected, rel=1e-6)
    assert [group for _, group in calls] == [dense_group, expert_group]

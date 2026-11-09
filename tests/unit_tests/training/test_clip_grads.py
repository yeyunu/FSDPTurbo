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
    # The batched L2 path stores each per-tensor norm in the gradients'
    # dtype (FP32 accumulation, low-precision scalar rounding), so bf16
    # only matches the FP32 reference within bf16 rounding.
    tolerance = 1e-6 if dtype == torch.float32 else 1e-2
    assert actual == pytest.approx(expected, rel=tolerance)


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
    gradient_shape = parameters[0].grad.shape
    float_calls = 0
    live_references = []
    original_float = torch.Tensor.float

    def tracked_float(tensor, *args, **kwargs):
        nonlocal float_calls
        float_calls += 1
        output = original_float(tensor, *args, **kwargs)
        if tensor.shape == gradient_shape:
            # Full-gradient FP32 upcast: at most one may be alive at a time.
            live_references.append(weakref.ref(output))
            assert all(reference() is None for reference in live_references[:-1])
        return output

    monkeypatch.setattr(torch.Tensor, "float", tracked_float)
    clip_grads.compute_grad_norm(parameters, norm_type=norm_type)
    assert all(reference() is None for reference in live_references)
    # Canary: every norm path upcasts to FP32 at least once, so the
    # tracking is proven live even when no full-gradient copy is made.
    assert float_calls >= 1


@pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.0, float("inf")])
def test_device_resident_gradients_are_not_copied(monkeypatch, norm_type):
    # Gradients already on the reduction device must be used in place.
    parameters = _make_parameters(torch.bfloat16)
    transfer_count = 0
    original_to = torch.Tensor.to

    def tracked_to(tensor, *args, **kwargs):
        nonlocal transfer_count
        if tensor.ndim == 1 and tensor.dtype == torch.bfloat16:
            transfer_count += 1
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)
    clip_grads.compute_grad_norm(parameters, norm_type=norm_type)
    assert transfer_count == 0


def test_reduction_device_normalizes_int_index(monkeypatch):
    # Production shape: current_device_index() returns an int ordinal while
    # Tensor.device is a torch.device; comparing them directly is always
    # unequal, so the index must be normalized to a torch.device first.
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cuda"))
    assert clip_grads._reduction_device() == torch.device("cuda", 0)


# Each entry is (weakref to the simulated transfer's copy, number of
# previous copies still alive at the moment of the transfer).
_simulated_transfers = []


class _SimulatedOffloadedGradient(torch.Tensor):
    """CPU gradient that reports a foreign device so the cpu_offload
    transfer branch runs in CPU-only test environments.

    ``to`` simulates the H2D copy with a tracked clone; dtype casts fall
    through to the real implementation.
    """

    @property
    def device(self):
        return torch.device("meta")

    def to(self, *args, **kwargs):
        if not args or not isinstance(args[0], torch.device):
            return super().to(*args, **kwargs)
        alive = sum(1 for reference, _ in _simulated_transfers if reference() is not None)
        copy = self.clone()
        _simulated_transfers.append((weakref.ref(copy), alive))
        return copy


def _make_offloaded_parameters(dtype, count=4):
    parameters = []
    raw_gradients = []
    for index in range(count):
        parameter = torch.nn.Parameter(torch.zeros(3, dtype=dtype))
        raw_gradient = torch.full((3,), float(index + 1), dtype=dtype)
        parameter.grad = raw_gradient.as_subclass(_SimulatedOffloadedGradient)
        parameters.append(parameter)
        raw_gradients.append(raw_gradient)
    return parameters, raw_gradients


@pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.0, float("inf")])
def test_offloaded_gradient_norm_transfers_one_gradient_at_a_time(norm_type):
    _simulated_transfers.clear()
    parameters, raw_gradients = _make_offloaded_parameters(torch.bfloat16)
    gradients = torch.cat([gradient.float() for gradient in raw_gradients])
    expected = torch.linalg.vector_norm(gradients, ord=norm_type).item()
    actual = clip_grads.compute_grad_norm(parameters, norm_type=norm_type)
    # Exactly one transfer per gradient, never while a previous transfer's
    # copy is still alive, and all copies released once the call returns.
    assert len(_simulated_transfers) == 4
    assert all(alive == 0 for _, alive in _simulated_transfers)
    assert all(reference() is None for reference, _ in _simulated_transfers)
    # Offloaded gradients keep the original per-gradient FP32-upcast
    # precision.
    assert actual == pytest.approx(expected, rel=1e-6)


def test_offloaded_gradient_clipping_transfers_one_gradient_at_a_time():
    _simulated_transfers.clear()
    parameters, raw_gradients = _make_offloaded_parameters(torch.bfloat16)
    original_gradients = [gradient.clone() for gradient in raw_gradients]
    total_norm = clip_grads.compute_grad_norm(parameters)
    _simulated_transfers.clear()
    clip_grads.clip_grad_by_norm(parameters, 1.0, total_norm)
    coefficient = min(1.0 / (total_norm + 1e-6), 1.0)
    # One transfer per gradient; while a new copy is created the previous
    # one may still be referenced by the loop variable, so at most one
    # earlier copy may be alive (the original one-at-a-time behavior).
    assert len(_simulated_transfers) == 4
    assert all(alive <= 1 for _, alive in _simulated_transfers)
    assert all(reference() is None for reference, _ in _simulated_transfers)
    for original, gradient in zip(original_gradients, raw_gradients):
        expected = original.clone().mul_(coefficient)
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


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

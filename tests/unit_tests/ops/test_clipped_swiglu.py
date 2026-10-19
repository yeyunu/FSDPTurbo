"""Reference and opt-in NPU contract tests for DeepSeek clipped SwiGLU."""

import os
from types import SimpleNamespace

import pytest
import torch

from fsdp_turbo.ops.clipped_swiglu import clipped_swiglu


def test_clamp_boundary_gradients():
    """
    Author: tichang
    Date: 2026-10-18
    Description: Clamp boundary gradients.
    Remarks: Synthetic tensors; no checkpoint or dataset required.
    """
    x = torch.tensor([[2.0, 3.0, -3.0, 1.0, 1.0, 2.0, -2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    y = clipped_swiglu(x, 2.0)
    y.sum().backward()
    gate = x.detach()[..., :4].clamp(max=2.0)
    up = x.detach()[..., 4:].clamp(-2.0, 2.0)
    sig = gate.sigmoid()
    expected_gate = up * sig * (1 + gate * (1 - sig))
    expected_gate[0, 1] = 0.0
    expected_up = gate * sig
    expected_up[0, 3] = 0.0
    torch.testing.assert_close(x.grad, torch.cat((expected_gate, expected_up), -1))
    assert x.grad[0, 0] != 0  # clamp includes the boundary
    assert x.grad[0, 5] != 0 and x.grad[0, 6] != 0


def test_reference_gradcheck():
    """
    Author: tichang
    Date: 2026-10-18
    Description: Reference gradcheck.
    Remarks: Synthetic tensors; no checkpoint or dataset required.
    """
    torch.manual_seed(19)
    x = (torch.randn(2, 10, dtype=torch.float64) * 3).requires_grad_()
    assert torch.autograd.gradcheck(lambda value: clipped_swiglu(value, 2.0), (x,))


@pytest.mark.parametrize("shape", [(0, 16), (2, 0, 16), (3, 0)])
def test_empty_input_preserves_autograd(shape):
    """
    Author: tichang
    Date: 2026-10-18
    Description: Empty input preserves autograd.
    Remarks: Synthetic tensors; no checkpoint or dataset required.
    """
    x = torch.empty(shape, requires_grad=True)
    y = clipped_swiglu(x, 10.0)
    y.sum().backward()
    assert y.shape == (*shape[:-1], shape[-1] // 2)
    assert x.grad.shape == x.shape


@pytest.mark.parametrize("limit", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_limit(limit):
    """
    Author: tichang
    Date: 2026-10-18
    Description: Invalid limit.
    Remarks: Synthetic tensors; no checkpoint or dataset required.
    """
    with pytest.raises(ValueError, match="finite and positive"):
        clipped_swiglu(torch.zeros(2, 8), limit)


@pytest.mark.parametrize("shape", [(), (2, 3), (1,) * 8 + (2,)])
def test_invalid_shape(shape):
    """
    Author: tichang
    Date: 2026-10-18
    Description: Invalid shape.
    Remarks: Synthetic tensors; no checkpoint or dataset required.
    """
    with pytest.raises(ValueError, match="rank 1-8"):
        clipped_swiglu(torch.zeros(shape), 10.0)


def test_integer_input_rejected():
    """
    Author: tichang
    Date: 2026-10-18
    Description: Integer input rejected.
    Remarks: Synthetic tensors; no checkpoint or dataset required.
    """
    with pytest.raises(TypeError, match="floating point"):
        clipped_swiglu(torch.zeros(2, 8, dtype=torch.int64), 10.0)


def test_cann_backward_binding_compatibility(monkeypatch):
    """
    Author: tichang
    Date: 2026-10-18
    Description: CANN backward binding compatibility.
    Remarks: Synthetic tensors; no checkpoint or dataset required.
    """
    backend = pytest.importorskip("fsdp_turbo.ops.npu.clipped_swiglu")

    calls = []

    def public_backward(*args, **kwargs):
        raise TypeError("_clipped_swiglu_grad() takes 2 positional arguments but 8 were given")

    def kernel(grad, x, **kwargs):
        calls.append(kwargs)
        assert grad.is_contiguous()
        # An input above the limit must reach backward unclipped.
        assert x[0, 0] == 11.0
        return torch.zeros_like(x)

    ops = SimpleNamespace(
        forward=lambda x, **kw: clipped_swiglu(x, kw["limit"], use_eager=True),
        backward=public_backward,
    )
    monkeypatch.setattr(backend, "_get_clipped_swiglu_ops", lambda: ops)
    monkeypatch.setattr(backend, "import_module", lambda _: SimpleNamespace(_clipped_swiglu_grad=kernel))
    x = torch.tensor([[11.0, 1.0]], requires_grad=True)
    backend._ClippedSwiglu.apply(x, 10.0).sum().backward()
    assert calls == [dict(group_index=None, dim=-1, alpha=1.0, limit=10.0, bias=0.0, interleaved=False, clamp_mode=0)]
    assert ops.backward is kernel


def test_unrelated_backward_errors_are_not_hidden(monkeypatch):
    """
    Author: tichang
    Date: 2026-10-18
    Description: Unrelated backward errors are not hidden.
    Remarks: Synthetic tensors; no checkpoint or dataset required.
    """
    backend = pytest.importorskip("fsdp_turbo.ops.npu.clipped_swiglu")

    def broken(*args, **kwargs):
        raise TypeError("unexpected kernel contract")

    ops = SimpleNamespace(forward=lambda x, **kw: x[..., :1].clone(), backward=broken)
    monkeypatch.setattr(backend, "_get_clipped_swiglu_ops", lambda: ops)
    x = torch.ones(1, 2, requires_grad=True)
    with pytest.raises(TypeError, match="unexpected kernel contract"):
        backend._ClippedSwiglu.apply(x, 10.0).sum().backward()


@pytest.fixture(name="npu_device")
def fixture_npu_device():
    """Use the backend registered by ops, without importing vendor code here."""
    device = torch.accelerator.current_accelerator()
    assert device is not None and device.type == "npu", "RUN_NPU_TESTS=1 requires an NPU backend"
    assert torch.accelerator.is_available(), "RUN_NPU_TESTS=1 requires an available NPU"
    return device


@pytest.mark.skipif(os.environ.get("RUN_NPU_TESTS") != "1", reason="requires an idle NPU and compatible CANN")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(2,), (3, 14), (2, 3, 2048)])
def test_npu_forward_backward(dtype, shape, npu_device):
    """
    Author: tichang
    Date: 2026-10-18
    Description: Npu forward backward.
    Remarks: Set RUN_NPU_TESTS=1 on an idle NPU with compatible CANN.
    """
    torch.manual_seed(23)
    source = torch.randn(shape, dtype=dtype) * 12
    x = source.to(npu_device).requires_grad_()
    reference = source.float().requires_grad_()
    y = clipped_swiglu(x, 10.0)
    expected = clipped_swiglu(reference, 10.0, use_eager=True)
    grad = torch.randn_like(expected).to(dtype)
    y.backward(grad.to(npu_device))
    expected.backward(grad.float())
    tolerances = {torch.float32: (2e-5, 2e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (1e-2, 1e-2)}
    rtol, atol = tolerances[dtype]
    torch.testing.assert_close(y.cpu().float(), expected.detach().to(dtype).float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(x.grad.cpu().float(), reference.grad.to(dtype).float(), rtol=rtol, atol=atol)


@pytest.mark.skipif(os.environ.get("RUN_NPU_TESTS") != "1", reason="requires an idle NPU and compatible CANN")
def test_npu_boundaries_and_noncontiguous_tensors(npu_device):
    # Noncontiguous input and upstream gradient; exact +/- limit and saturated values.
    """
    Author: tichang
    Date: 2026-10-18
    Description: Npu boundaries and noncontiguous tensors.
    Remarks: Set RUN_NPU_TESTS=1 on an idle NPU with compatible CANN.
    """
    source = torch.tensor([[10.0, 11.0, -11.0, 3.0], [10.0, -10.0, 11.0, -11.0]])
    reference = source.t().detach().requires_grad_()
    x = source.to(npu_device).t().detach().requires_grad_()
    expected = clipped_swiglu(reference, 10.0, use_eager=True)
    y = clipped_swiglu(x, 10.0)
    grad = torch.arange(8, dtype=torch.float32, device=npu_device).reshape(4, 2)[:, ::2]
    assert not x.is_contiguous()
    assert not grad.is_contiguous()
    y.backward(grad)
    expected.backward(grad.cpu())
    torch.testing.assert_close(y.cpu(), expected.detach(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(x.grad.cpu(), reference.grad, rtol=2e-5, atol=2e-5)

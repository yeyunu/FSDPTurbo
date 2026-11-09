# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for the MoonEP grouped-GEMM primitives.

The CPU implementation is the numerical oracle; on a machine with an accelerator
the same assertions run against the device primitives. FP32 on CPU for a tight
check, BF16 elsewhere because the fused kernels accept nothing else.
"""

import pytest
import torch
import torch.nn.functional as F

from fsdp_turbo.ops.moonep_grouped_matmul import (
    moonep_grouped_matmul,
    moonep_grouped_matmul_dgrad,
    moonep_grouped_matmul_wgrad,
)

# One entry per accelerator that has MoonEP primitives registered.
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if hasattr(torch, "npu") and torch.npu.is_available():  # pragma: no cover - Ascend only
    DEVICES.append("npu")


def _reference(inputs, group_ends, weights):
    """Dense per-group reference built from plain ``torch`` matmuls."""
    output = torch.zeros(inputs.shape[0], weights.shape[1], dtype=inputs.dtype, device=inputs.device)
    start = 0
    for group, end in enumerate(group_ends.tolist()):
        if end > start:
            output[start:end] = inputs[start:end] @ weights[group].transpose(0, 1)
        start = end
    return output


def _dtype_for(device):
    """The dtype this device's grouped GEMM accepts.

    CPU runs in FP32 so the reference is checked tightly. The accelerator
    primitives go through fused grouped-GEMM kernels that only accept BF16 --
    handing them FP32 raises "Expected mat_a to be BFloat16 matrix got Float" --
    so the same assertions run there in the production dtype instead.
    """
    return torch.float32 if device == "cpu" else torch.bfloat16


def _tolerance_for(device):
    """BF16 carries about three decimal digits, hence the looser accelerator bound."""
    return (1e-4, 1e-4) if device == "cpu" else (2e-2, 2e-2)


def _case(device, dtype=None, rows=48, groups=5, in_features=16, out_features=24, tail=6):
    """Build a MoonEP-shaped case: ragged groups plus an unowned static tail."""
    dtype = _dtype_for(device) if dtype is None else dtype
    torch.manual_seed(1234)
    sizes = [8, 0, 12, 6, 16]
    assert len(sizes) == groups and sum(sizes) + tail == rows
    ends = torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(0).to(torch.int32)
    # The final group absorbs the reserved tail, mirroring compact_group_ends.
    ends[-1] = rows
    inputs = torch.randn(rows, in_features, dtype=dtype, device=device)
    weights = torch.randn(groups, out_features, in_features, dtype=dtype, device=device)
    return inputs, ends, weights


@pytest.mark.parametrize("device", DEVICES)
def test_forward_matches_dense_reference(device):
    inputs, ends, weights = _case(device)
    rtol, atol = _tolerance_for(device)
    output = moonep_grouped_matmul(inputs, ends, weights)
    assert output.shape == (inputs.shape[0], weights.shape[1])
    torch.testing.assert_close(output, _reference(inputs, ends, weights), rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
def test_dgrad_and_wgrad_match_autograd(device):
    inputs, ends, weights = _case(device)
    rtol, atol = _tolerance_for(device)
    inputs = inputs.clone().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)

    reference = _reference(inputs, ends, weights)
    grad_output = torch.randn_like(reference)
    reference.backward(grad_output)

    dgrad = moonep_grouped_matmul_dgrad(grad_output, ends, weights.detach())
    wgrad = moonep_grouped_matmul_wgrad(grad_output, inputs.detach(), ends)

    torch.testing.assert_close(dgrad, inputs.grad, rtol=rtol, atol=atol)
    assert wgrad.shape == weights.shape
    torch.testing.assert_close(wgrad, weights.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_groups_produce_zero_weight_gradient(device):
    """Group 1 is empty, so its weight gradient must be exactly zero."""
    inputs, ends, weights = _case(device)
    grad_output = torch.randn(inputs.shape[0], weights.shape[1], dtype=inputs.dtype, device=device)
    wgrad = moonep_grouped_matmul_wgrad(grad_output, inputs, ends)
    assert torch.count_nonzero(wgrad[1]) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_the_two_weight_orientations_agree(device):
    """A transposed weight store must produce the same three results.

    ``PatchKimiMoeExperts`` holds ``[E, in, out]`` where Qwen holds
    ``[E, out, in]``. MoonEP keeps whichever the model brought, so the only thing
    that may differ is the operand layout handed to the kernels -- the arithmetic
    must not. Comparing the two paths against each other rather than against a
    reference is what makes that an equality instead of a tolerance.

    ``out_features != in_features`` on purpose: with a square matrix the two
    layouts are indistinguishable and the test would pass vacuously.
    """
    inputs, ends, weights = _case(device)
    assert weights.shape[1] != weights.shape[2], "a square weight cannot tell the layouts apart"
    rtol, atol = _tolerance_for(device)
    flipped = weights.transpose(-2, -1).contiguous()

    forward = moonep_grouped_matmul(inputs, ends, weights)
    forward_flipped = moonep_grouped_matmul(inputs, ends, flipped, weights_transposed=True)
    torch.testing.assert_close(forward_flipped, forward, rtol=rtol, atol=atol)

    grad_output = torch.randn_like(forward)
    dgrad = moonep_grouped_matmul_dgrad(grad_output, ends, weights)
    dgrad_flipped = moonep_grouped_matmul_dgrad(grad_output, ends, flipped, weights_transposed=True)
    torch.testing.assert_close(dgrad_flipped, dgrad, rtol=rtol, atol=atol)

    wgrad = moonep_grouped_matmul_wgrad(grad_output, inputs, ends)
    wgrad_flipped = moonep_grouped_matmul_wgrad(grad_output, inputs, ends, weights_transposed=True)
    # Emitted in the caller's own layout, so it is the transpose of the other.
    assert wgrad_flipped.shape == (wgrad.shape[0], wgrad.shape[2], wgrad.shape[1])
    torch.testing.assert_close(wgrad_flipped, wgrad.transpose(-2, -1), rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", DEVICES)
def test_a_production_shape_matches_the_reference(device):
    """A 128-wide contraction, where the fused kernels pick a different tiling.

    The cases above are deliberately small and ragged. This one has the aligned
    dimensions a real expert matrix has, which is the path production takes.
    """
    dtype = _dtype_for(device)
    rtol, atol = _tolerance_for(device)
    torch.manual_seed(1234)
    ends = torch.tensor([3, 3, 9], dtype=torch.int32, device=device)
    inputs = torch.randn(9, 128, dtype=dtype, device=device)
    weights = torch.randn(3, 256, 128, dtype=dtype, device=device)

    torch.testing.assert_close(
        moonep_grouped_matmul(inputs, ends, weights),
        _reference(inputs, ends, weights),
        rtol=rtol,
        atol=atol,
    )


def test_cuda_per_group_weight_gradient_matches_reference():
    """Exercise the CUDA weight-gradient loop, which is device-agnostic torch.

    The default CUDA weight gradient avoids torch._grouped_mm's ragged-contraction
    form, which has been observed to hang. The loop it uses instead is written in
    plain torch ops, so it can be validated on CPU tensors: this covers the group
    slicing, the empty-group zeroing, and the host-offset hint.
    """
    from fsdp_turbo.ops.cuda.moonep_grouped_matmul import moonep_grouped_matmul_wgrad_cuda

    inputs, ends, weights = _case("cpu", torch.float32)
    grad_output = torch.randn(inputs.shape[0], weights.shape[1], dtype=inputs.dtype)
    expected = moonep_grouped_matmul_wgrad(grad_output, inputs, ends)

    torch.testing.assert_close(moonep_grouped_matmul_wgrad_cuda(grad_output, inputs, ends), expected)
    # Passing the offsets that are already on the host must not change the result.
    torch.testing.assert_close(moonep_grouped_matmul_wgrad_cuda(grad_output, inputs, ends, ends.tolist()), expected)


def test_group_count_mismatch_is_rejected():
    inputs, ends, weights = _case("cpu", torch.float32)
    with pytest.raises(RuntimeError, match="one cumulative end offset per group"):
        moonep_grouped_matmul(inputs, ends[:-1], weights)


@pytest.mark.skipif(not hasattr(torch, "_grouped_mm"), reason="needs PyTorch 2.9+")
def test_grouped_mm_operand_contract():
    """Pin the ``torch._grouped_mm`` layouts the CUDA primitives depend on.

    Meta tensors exercise PyTorch's own shape inference, so this guards the
    contract on machines without a GPU. In particular the 2-D x 2-D form must
    emit ``[G, out_features, in_features]``, which is the weight layout - if it
    ever changed to ``[G, in, out]`` the weight gradient would be transposed.
    """
    groups, rows, in_features, out_features = 5, 48, 16, 24
    offsets = torch.tensor([8, 8, 20, 26, rows], dtype=torch.int32, device="meta")
    inputs = torch.empty((rows, in_features), dtype=torch.bfloat16, device="meta")
    weights = torch.empty((groups, out_features, in_features), dtype=torch.bfloat16, device="meta")
    grad_output = torch.empty((rows, out_features), dtype=torch.bfloat16, device="meta")

    forward = torch._grouped_mm(inputs, weights.transpose(-2, -1), offs=offsets)
    dgrad = torch._grouped_mm(grad_output, weights, offs=offsets)
    wgrad = torch._grouped_mm(grad_output.transpose(0, 1), inputs, offs=offsets)

    assert tuple(forward.shape) == (rows, out_features)
    assert tuple(dgrad.shape) == (rows, in_features)
    assert tuple(wgrad.shape) == weights.shape


from fsdp_turbo.ops.cuda.moonep_grouped_matmul import (  # noqa: E402
    grouped_mm,
    resolve_grouped_mm,
)


def test_the_resolver_reflects_what_the_build_exposes():
    """A callable exactly when one of the two spellings exists, else None.

    Ascend needs neither -- it has ``npu_grouped_matmul`` -- so ``None`` is the
    correct answer on a build that predates the operator.
    """
    available = hasattr(F, "grouped_mm") or hasattr(torch, "_grouped_mm")
    resolved = resolve_grouped_mm()
    if available:
        assert callable(resolved)
    else:
        assert resolved is None


def test_public_alias_is_preferred_when_both_exist(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(F, "grouped_mm", sentinel, raising=False)
    monkeypatch.setattr(torch, "_grouped_mm", object(), raising=False)
    assert resolve_grouped_mm() is sentinel


def test_private_spelling_is_used_when_the_alias_is_absent(monkeypatch):
    """The case that broke on the target machine: only the private name exists."""
    sentinel = object()
    monkeypatch.delattr(F, "grouped_mm", raising=False)
    monkeypatch.setattr(torch, "_grouped_mm", sentinel, raising=False)
    assert resolve_grouped_mm() is sentinel


def test_missing_operator_reports_both_spellings_and_a_way_out(monkeypatch):
    monkeypatch.delattr(F, "grouped_mm", raising=False)
    monkeypatch.delattr(torch, "_grouped_mm", raising=False)
    assert resolve_grouped_mm() is None

    with pytest.raises(RuntimeError) as failure:
        grouped_mm(torch.zeros(2, 2), torch.zeros(1, 2, 2), torch.tensor([2], dtype=torch.int32))
    message = str(failure.value)
    # Both spellings and the fallback, for a machine that has neither.
    assert "torch.nn.functional.grouped_mm" in message
    assert "torch._grouped_mm" in message
    assert "eager" in message


def test_offsets_are_passed_as_a_keyword(monkeypatch):
    """The operator takes `offs` by keyword; a positional third argument is an error."""
    seen = {}

    def fake(mat_a, mat_b, offs=None):
        seen["offs"] = offs
        return mat_a

    monkeypatch.setattr(F, "grouped_mm", fake, raising=False)
    offsets = torch.tensor([4], dtype=torch.int32)
    grouped_mm(torch.zeros(4, 2), torch.zeros(1, 2, 2), offsets)
    assert seen["offs"] is offsets

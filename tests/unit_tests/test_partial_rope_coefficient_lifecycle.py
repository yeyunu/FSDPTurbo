# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Regression tests for the fused partial-RoPE coefficient cache (PR #208 review).

Both defects reproduce on the host: the CANN operator only consumes the packed
coefficients, so packing and the autograd save/restore are plain PyTorch.  The
fused call is therefore replaced by an exact PyTorch implementation of the same
interleaved partial rotation, and nothing here needs an NPU.
"""

import pytest
import torch

from fsdp_turbo.ops.cpu.rope import inplace_rope_cpu, rope_torch

npu_rope = pytest.importorskip(
    "fsdp_turbo.ops.npu.rope",
    reason="the NPU operator package is unavailable (needs torch_npu and flash-linear-attention)",
)
partial_rotary = pytest.importorskip("fsdp_turbo.ops.npu.partial_rotary")


class _FusedPartialRotaryReplacement:
    """Exact PyTorch stand-in for ``inplace_partial_rotary_mul`` and its backward.

    It performs the same interleaved rotation as the CANN operator, in place, and
    records the coefficient shapes it is handed so a test can assert the fused
    leg was (or was not) taken.
    """

    def __init__(self):
        self.coefficient_shapes = []

    def inplace_partial_rotary_mul(self, x, r1, r2, *, rotary_mode, partial_slice):
        self.coefficient_shapes.append(tuple(r1.shape))
        start = partial_slice[0]
        x[..., start:] = rope_torch(x[..., start:], r1[..., ::2].float(), r2[..., ::2].float(), interleaved=True)

    def inplace_partial_rotary_mul_backward(self, grad, r1, r2, *, rotary_mode, partial_slice):
        start = partial_slice[0]
        grad[..., start:] = rope_torch(grad[..., start:], r1[..., ::2].float(), -r2[..., ::2].float(), interleaved=True)


@pytest.fixture(name="fused_operator")
def _fused_operator_replacement(monkeypatch):
    """Route the fused leg through the PyTorch replacement and reset the cache."""
    replacement = _FusedPartialRotaryReplacement()
    monkeypatch.setattr(npu_rope, "IS_CANN_OPS_TRANSFORMER_AVAILABLE", True)
    monkeypatch.setattr(partial_rotary, "_cann_ops", lambda: replacement)
    npu_rope.clear_rope_cache()
    yield replacement
    npu_rope.clear_rope_cache()


def _coefficients(sequence, half, dtype):
    """Fixed positional buffers, i.e. ordinary tensors created outside inference."""
    angles = torch.arange(sequence, dtype=torch.float32).unsqueeze(-1) * torch.arange(half, dtype=torch.float32)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _assert_matches_reference(got, reference, dtype):
    """FP32 rotates exactly; BF16 coefficients are rounded before the operator runs."""
    if dtype == torch.float32:
        assert torch.equal(got, reference)
    else:
        torch.testing.assert_close(got, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inference_warmup_keeps_coefficients_saveable_for_training(fused_operator, dtype):
    """Inference-then-training must not cache inference tensors for backward."""
    sequence, half, heads, batch = 8, 16, 4, 2
    cos, sin = _coefficients(sequence, half, dtype)
    shape = (batch, sequence, heads, 2 * half)

    # Warm the cache the way serving does, then resume training on the same buffers.
    with torch.inference_mode():
        npu_rope.partial_rope_npu(torch.zeros(shape, dtype=dtype), cos, sin)

    warmed_r1, warmed_r2 = npu_rope._pack_partial_coefficients(cos, sin, inverse=False, dtype=dtype)
    assert not torch.is_inference(warmed_r1)
    assert not torch.is_inference(warmed_r2)
    repeated_r1, repeated_r2 = npu_rope._pack_partial_coefficients(cos, sin, inverse=False, dtype=dtype)
    assert repeated_r1 is warmed_r1 and repeated_r2 is warmed_r2

    # Training saves r1/r2 for backward, which inference tensors cannot survive.
    leaf = torch.randn(shape, dtype=dtype, requires_grad=True)
    npu_rope.partial_rope_npu(leaf * 1.0, cos, sin).sum().backward()
    assert leaf.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
def test_canonical_broadcast_coefficients_match_two_dimensional_rows(fused_operator, dtype, inverse):
    """``[1, S, 1, R/2]`` must rotate exactly like the ``[S, R/2]`` rows it expands."""
    sequence, half, heads, batch = 8, 16, 4, 2
    cos, sin = _coefficients(sequence, half, dtype)
    broadcast_cos, broadcast_sin = cos.reshape(1, sequence, 1, half), sin.reshape(1, sequence, 1, half)
    base = torch.randn(batch, sequence, heads, 2 * half, dtype=dtype)

    reference = inplace_rope_cpu(base.clone(), cos, sin, interleaved=True, inverse=inverse, inplace=True)
    from_rows = npu_rope.partial_rope_npu(base.clone(), cos, sin, inverse=inverse)
    from_broadcast = npu_rope.partial_rope_npu(base.clone(), broadcast_cos, broadcast_sin, inverse=inverse)

    _assert_matches_reference(from_rows, reference, dtype)
    assert torch.equal(from_broadcast, from_rows)
    # Both layouts reach the fused operator with one explicit row per position.
    assert fused_operator.coefficient_shapes == [(1, sequence, 1, 2 * half)] * 2

    # Both layouts normalize to the same cache key, so no duplicate packing.
    rows = npu_rope._partial_coefficient_pairs(cos, sin, sequence)
    broadcast_rows = npu_rope._partial_coefficient_pairs(broadcast_cos, broadcast_sin, sequence)
    packed_rows = npu_rope._pack_partial_coefficients(*rows, inverse=inverse, dtype=dtype)
    packed_broadcast = npu_rope._pack_partial_coefficients(*broadcast_rows, inverse=inverse, dtype=dtype)
    assert packed_broadcast[0] is packed_rows[0] and packed_broadcast[1] is packed_rows[1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layout_without_a_sequence_axis_falls_back_to_the_reference(fused_operator, monkeypatch, dtype):
    """One rotation shared by every position cannot be indexed; the reference broadcasts it."""
    sequence, half, heads, batch = 8, 16, 4, 2
    cos, sin = _coefficients(sequence, half, dtype)
    shared_cos, shared_sin = cos[:1].reshape(1, 1, 1, half), sin[:1].reshape(1, 1, 1, half)
    base = torch.randn(batch, sequence, heads, 2 * half, dtype=dtype)
    # The device kernels take NPU tensors; this host-only test keeps the eager leg.
    monkeypatch.setattr(npu_rope, "torch_npu", None)
    monkeypatch.setattr(npu_rope, "triton_inplace_rope", None)

    reference = inplace_rope_cpu(base.clone(), shared_cos, shared_sin, interleaved=True, inverse=True, inplace=True)
    got = npu_rope.partial_rope_npu(base.clone(), shared_cos, shared_sin, inverse=True)

    assert not fused_operator.coefficient_shapes
    _assert_matches_reference(got, reference, dtype)


def test_position_count_mismatch_is_rejected(fused_operator):
    """Rows that cover neither the whole sequence nor a single position are an error."""
    cos, sin = _coefficients(4, 16, torch.float32)
    x = torch.zeros(2, 8, 4, 32)

    with pytest.raises(ValueError, match="positions"):
        npu_rope.partial_rope_npu(x, cos, sin)

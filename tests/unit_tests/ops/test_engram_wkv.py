"""CPU parity for the fused-Engram autograd glue (residual, mask, packed wkv).

The fused kernels themselves are validated on NPU against the golden small
operators; here the pure-torch CPU engram op stands in for them, so the test
only exercises the assembly logic in ``EngramWkvFunction`` (the device-
neutral glue registered as the ``engram_wkv`` op) against the eager
formulation used by the reference ``Engram`` module.
"""

import pytest
import torch
import torch.nn.functional as F

from fsdp_turbo.ops.engram import EngramWkvFunction, add_engram_residual, engram_backward, engram_forward
from fsdp_turbo.ops.registry import get_op

EPS = 1e-6
CLAMP_VALUE = 1e-6


def test_engram_wkv_has_cpu_registration():
    assert get_op("engram_wkv", "cpu").__name__ == "engram_wkv_cpu"


def _reference_engram(x, memory, wkv_weight, k_weight, q_weight, token_mask):
    head_count, dim = k_weight.shape
    key, value = F.linear(memory, wkv_weight).split([head_count * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (head_count, dim))
    weight = q_weight.float() * k_weight.float()
    x_float = x.float()
    rstd = torch.rsqrt(x_float.square().mean(-1) + EPS) * torch.rsqrt(key.square().mean(-1) + EPS)
    dot = (x_float * weight * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(CLAMP_VALUE).sqrt()
    gate = torch.sigmoid(torch.where(dot < 0, -magnitude, magnitude))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return x_float + gate.unsqueeze(-1) * value.float().unsqueeze(-2)


def _make_case(seed, with_mask):
    torch.manual_seed(seed)
    batch_size, sequence_length, head_count, dim, embedding_dim = 2, 4, 4, 32, 64
    x = torch.randn(batch_size, sequence_length, head_count, dim)
    memory = torch.randn(batch_size, sequence_length, embedding_dim)
    wkv_weight = torch.randn((head_count + 1) * dim, embedding_dim) * 0.05
    k_weight = torch.ones(head_count, dim)
    q_weight = torch.ones(head_count, dim)
    token_mask = None
    if with_mask:
        token_mask = torch.rand(batch_size, sequence_length) > 0.25
    grad_output = torch.randn(batch_size, sequence_length, head_count, dim) * 0.1
    return x, memory, wkv_weight, k_weight, q_weight, token_mask, grad_output


def _run_case(seed, with_mask):
    x, memory, wkv_weight, k_weight, q_weight, token_mask, grad_output = _make_case(seed, with_mask)

    ref_inputs = [tensor.clone().requires_grad_(True) for tensor in (x, memory, wkv_weight, k_weight, q_weight)]
    ref_out = _reference_engram(*ref_inputs, token_mask)
    ref_grads = torch.autograd.grad((ref_out * grad_output).sum(), ref_inputs)

    fused_inputs = [tensor.clone().requires_grad_(True) for tensor in (x, memory, wkv_weight, k_weight, q_weight)]
    fused_out = EngramWkvFunction.apply(*fused_inputs, EPS, CLAMP_VALUE, token_mask)
    fused_grads = torch.autograd.grad((fused_out * grad_output).sum(), fused_inputs)

    torch.testing.assert_close(fused_out, ref_out, rtol=1e-4, atol=1e-5)
    for name, fused_grad, ref_grad in zip(("x", "memory", "wkv", "k_weight", "q_weight"), fused_grads, ref_grads):
        assert fused_grad.shape == ref_grad.shape, name
        torch.testing.assert_close(fused_grad, ref_grad, rtol=2e-4, atol=1e-5)


def test_fused_engram_without_mask_matches_reference():
    _run_case(seed=0, with_mask=False)


def test_fused_engram_with_token_mask_matches_reference():
    _run_case(seed=1, with_mask=True)


def test_low_precision_residual_add_matches_fp32_expression_bitwise():
    for dtype in (torch.bfloat16, torch.float16):
        # Every 16-bit operand pattern, paired with extremes, subnormals,
        # cancellation, ties, infinities and NaNs.
        left = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)[:, None]
        right = torch.tensor(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                2.0**-133,
                2.0**-24,
                2.0**-14,
                2.0**-8,
                65504.0,
                -65504.0,
                float("inf"),
                float("-inf"),
                float("nan"),
            ],
            dtype=dtype,
        )
        expected = (left.float() + right.float()).to(dtype)
        actual = add_engram_residual(left, right)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(torch.signbit(actual[expected == 0]), torch.signbit(expected[expected == 0]))


def test_residual_preserves_legacy_mixed_and_double_precision_semantics():
    residual = torch.tensor([1.0 + 2.0**-40, -1.0], dtype=torch.float64)
    for branch in (torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float32)):
        expected = (residual.float() + branch.float()).to(residual.dtype)
        torch.testing.assert_close(add_engram_residual(residual, branch), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mask_mode", ["none", "partial", "all"])
def test_optimized_glue_matches_legacy_output_and_all_gradients(dtype, mask_mode):
    hidden, memory, weight, key_gamma, query_gamma, mask, grad_output = _make_case(7, True)
    inputs = [tensor.to(dtype).requires_grad_() for tensor in (hidden, memory, weight, key_gamma, query_gamma)]
    hidden, memory, weight, key_gamma, query_gamma = inputs
    grad_output = grad_output.to(dtype).transpose(2, 3).contiguous().transpose(2, 3)
    if mask_mode == "none":
        mask = None
    elif mask_mode == "all":
        mask = torch.zeros_like(mask)
    head_count, head_dim = key_gamma.shape
    key_weight, value_weight = weight.split([head_count * head_dim, head_dim])
    key_projection = key_weight.reshape(head_count, head_dim, -1).transpose(1, 2).contiguous()
    value_projection = value_weight.t().contiguous()
    with torch.no_grad():
        value_out, scores, keys, value, gates = engram_forward(
            hidden, memory, key_projection, value_projection, key_gamma, query_gamma
        )
        branch = value_out if mask is None else value_out * mask[:, :, None, None]
        expected_output = (hidden.float() + branch.float()).to(dtype)
        branch_grad = grad_output if mask is None else grad_output * mask[:, :, None, None]
        expected_gradients = list(
            engram_backward(
                branch_grad.contiguous(),
                hidden,
                memory,
                key_projection,
                value_projection,
                key_gamma,
                query_gamma,
                scores,
                gates,
                keys,
                value,
            )
        )
        expected_gradients[0] = (expected_gradients[0].float() + grad_output.float()).to(dtype)
        expected_weight_gradient = torch.cat(
            [expected_gradients[2].transpose(1, 2).reshape(head_count * head_dim, -1), expected_gradients[3].t()]
        )
        expected_gradients = [*expected_gradients[:2], expected_weight_gradient, *expected_gradients[4:]]

    output = EngramWkvFunction.apply(*inputs, EPS, CLAMP_VALUE, mask)
    gradients = torch.autograd.grad(output, inputs, grad_output)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if mask_mode == "all":
        assert torch.equal(output, hidden)
        assert torch.equal(gradients[0], grad_output)
        assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients[1:])


def test_fused_engram_uses_current_weight_after_optimizer_update():
    hidden, memory, weight, key_gamma, query_gamma, _, grad_output = _make_case(4, False)
    inputs = [tensor.requires_grad_() for tensor in (hidden, memory, weight, key_gamma, query_gamma)]
    optimizer = torch.optim.SGD([weight], lr=0.05)
    output_before = EngramWkvFunction.apply(*inputs, EPS, CLAMP_VALUE, None)
    weight_before = weight.detach().clone()
    output_before.backward(grad_output)
    optimizer.step()
    assert torch.count_nonzero(weight.detach() - weight_before) > 0
    output_after = EngramWkvFunction.apply(*inputs, EPS, CLAMP_VALUE, None)
    reference = _reference_engram(*inputs, None)
    torch.testing.assert_close(output_after, reference, rtol=1e-4, atol=1e-5)
    assert not torch.equal(output_before, output_after)

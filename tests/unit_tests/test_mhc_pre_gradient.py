# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU regression tests for the fused MHC h_pre gradient (GitCode PR 173).

``fsdp_turbo/ops/npu/mhc.py`` fuses the MHC pre-Sinkhorn-premix operator with a small
eager h_pre projection. The eager projection exists to keep the activation-recompute
saved-tensor accounting symmetric, but it must not own a second copy of the h_pre
gradient: the operator backward already turns ``grad_h_pre`` into x/phi/alpha/bias
gradients, and ``Block.forward`` additionally feeds h_pre back as the next sublayer's
``pre_mix``, so both a plain h_pre-only backward and the normal fused training path are
affected. These tests pin the fused function to the unfused (``use_fused_mhc=False``)
reference path on CPU, replacing the CANN operators with an equivalent torch
implementation.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fsdp_turbo.ops.cpu.sinkhorn import hc_split_sinkhorn_cpu

_REPOSITORY_ROOT = Path(__file__).parents[2]
_MHC_MODULE_PATH = _REPOSITORY_ROOT / "fsdp_turbo" / "ops" / "npu" / "mhc.py"
_SPEC = importlib.util.spec_from_file_location("test_mhc_pre_gradient_impl", _MHC_MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
mhc_npu = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mhc_npu)

_HC_MULT = 4
_NUM_ITERS = 20
_HC_EPS = 1e-6
_NORM_EPS = 1e-6


def _reference_pre(x, phi, alpha, bias):
    """`Block.hc_mixes` on CPU: the unfused `pre` coefficient, i.e. the reference h_pre."""
    flat_x = x.flatten(2).float()
    inv_rms = torch.rsqrt(flat_x.square().mean(dim=-1, keepdim=True) + _NORM_EPS)
    mixes = F.linear(flat_x, phi.float()) * inv_rms
    pre, _, _ = hc_split_sinkhorn_cpu(
        mixes,
        alpha.float(),
        bias.float(),
        hc_mult=_HC_MULT,
        sinkhorn_iters=_NUM_ITERS,
        eps=_HC_EPS,
    )
    return pre


def _reference_collapse(x, premix):
    """`Block.hc_pre` on CPU: collapse the hc copies with the given premix weights."""
    return torch.sum(premix.float().unsqueeze(-1) * x.float(), dim=2).to(x.dtype)


class _CpuMhcOps:
    """Torch stand-in for the CANN premix forward/backward operator pair."""

    def __init__(self):
        self._config = None

    @staticmethod
    def _forward_math(x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps):
        hc = hc_mult
        flat_x = x.flatten(-2).float()
        inv_rms = torch.rsqrt(flat_x.square().mean(dim=-1, keepdim=True) + norm_eps)
        mixes = F.linear(flat_x, phi.float()) * inv_rms
        pre_w, post_w, comb_w = mixes.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = bias.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = alpha.float().unbind(0)
        h_pre = torch.sigmoid(pre_w * pre_scale + pre_b) + hc_eps
        h_post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
        for _ in range(num_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
        h_in = _reference_collapse(x, premix)
        return h_in, h_post, comb.flatten(-2), h_pre

    def pre(self, x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps, is_training):
        self._config = (hc_mult, num_iters, hc_eps, norm_eps)
        h_in, h_post, h_res, h_pre = self._forward_math(
            x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps
        )
        flat_x = x.flatten(-2).float()
        inv_rms = torch.rsqrt(flat_x.square().mean(dim=-1, keepdim=True) + norm_eps)
        return h_in, h_post, h_res, h_pre, flat_x, inv_rms, flat_x.sum(-1), flat_x.square().sum(-1)

    def pre_backward(
        self,
        grad_h_in,
        grad_h_post,
        grad_h_res,
        x,
        phi,
        alpha,
        bias,
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        premix,
        grad_h_pre,
        hc_eps,
    ):
        hc_mult, num_iters, _, norm_eps = self._config
        with torch.enable_grad():
            inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (x, phi, alpha, bias, premix)]
            h_in, h_post, h_res, h_pre_out = self._forward_math(*inputs, hc_mult, num_iters, hc_eps, norm_eps)
            return torch.autograd.grad(
                (h_in, h_post, h_res, h_pre_out),
                tuple(inputs),
                grad_outputs=(grad_h_in, grad_h_post, grad_h_res.flatten(-2), grad_h_pre),
            )


@pytest.fixture(name="cpu_mhc_ops")
def fixture_cpu_mhc_ops(monkeypatch):
    """Swap the CANN operators and the NPU-only input validation for CPU stand-ins."""
    ops = _CpuMhcOps()
    monkeypatch.setattr(mhc_npu, "_get_mhc_ops", lambda: ops)
    monkeypatch.setattr(mhc_npu, "_validate_pre_inputs", lambda *args, **kwargs: None)
    return ops


def _make_inputs(seed, batch=2, seq=3, head_dim=8):
    generator = torch.Generator().manual_seed(seed)
    mix_dim = _HC_MULT * _HC_MULT + 2 * _HC_MULT
    return (
        torch.randn(batch, seq, _HC_MULT, head_dim, generator=generator),
        torch.randn(mix_dim, _HC_MULT * head_dim, generator=generator),
        torch.randn(3, generator=generator),
        torch.randn(mix_dim, generator=generator),
        torch.randn(batch, seq, _HC_MULT, generator=generator),
    )


def _clone_with_grad(tensors):
    return [tensor.detach().clone().requires_grad_(True) for tensor in tensors]


def _fused_pre(inputs):
    return mhc_npu.fused_mhc_pre_sinkhorn_premix(
        *inputs,
        hc_mult=_HC_MULT,
        num_iters=_NUM_ITERS,
        hc_eps=_HC_EPS,
        norm_eps=_NORM_EPS,
    )


def _assert_gradient_matches_reference(name, fused_grad, reference_grad):
    assert fused_grad is not None, f"{name} received no gradient"
    reference_norm = reference_grad.norm()
    relative_error = (fused_grad - reference_grad).norm() / reference_norm
    # A doubled operator + eager path gives relative_error ~= 1.0; a correct single path stays near fp32 noise.
    assert relative_error < 1e-5, (
        f"{name} gradient differs from the unfused reference (relative error {relative_error:.3e}); "
        "the h_pre gradient is being back-propagated more than once"
    )


def test_h_pre_gradient_is_not_counted_twice(cpu_mhc_ops):
    x, phi, alpha, bias, premix = _make_inputs(seed=0)

    fused_inputs = _clone_with_grad((x, phi, alpha, bias, premix))
    *_, h_pre = _fused_pre(fused_inputs)
    h_pre.sum().backward()

    reference_inputs = _clone_with_grad((x, phi, alpha, bias, premix))
    _reference_pre(*reference_inputs[:4]).sum().backward()

    for name, fused_tensor, reference_tensor in zip(
        ("x", "phi", "alpha", "bias"), fused_inputs[:4], reference_inputs[:4]
    ):
        _assert_gradient_matches_reference(name, fused_tensor.grad, reference_tensor.grad)


def test_no_grad_forward_matches_grad_forward(cpu_mhc_ops):
    x, phi, alpha, bias, premix = _make_inputs(seed=1)

    with torch.no_grad():
        *_, h_pre_no_grad = _fused_pre((x, phi, alpha, bias, premix))

    fused_inputs = _clone_with_grad((x, phi, alpha, bias, premix))
    *_, h_pre_grad_mode = _fused_pre(fused_inputs)

    assert torch.equal(h_pre_no_grad, h_pre_grad_mode)
    torch.testing.assert_close(h_pre_no_grad, _reference_pre(x, phi, alpha, bias), rtol=1e-6, atol=1e-7)


def test_h_pre_cross_layer_gradient_is_not_counted_twice(cpu_mhc_ops):
    """`Block.forward` hands this sublayer's h_pre to the next one as its `pre_mix`."""
    fused_first = _clone_with_grad(_make_inputs(seed=2))
    second_phi, second_alpha, second_bias = _clone_with_grad(_make_inputs(seed=3)[1:4])

    h_in_first, _, _, h_pre_first = _fused_pre(fused_first)
    next_stream = h_in_first.unsqueeze(-2).expand(-1, -1, _HC_MULT, -1)
    h_in_second, _, _, _ = _fused_pre([next_stream, second_phi, second_alpha, second_bias, h_pre_first])
    h_in_second.sum().backward()

    reference_first = _clone_with_grad(_make_inputs(seed=2))
    reference_h_in_first = _reference_collapse(reference_first[0], reference_first[4])
    reference_next_stream = reference_h_in_first.unsqueeze(-2).expand(-1, -1, _HC_MULT, -1)
    reference_pre_first = _reference_pre(*reference_first[:4])
    _reference_collapse(reference_next_stream, reference_pre_first).sum().backward()

    for name, fused_tensor, reference_tensor in zip(
        ("x", "phi", "alpha", "bias", "premix"), fused_first, reference_first
    ):
        _assert_gradient_matches_reference(name, fused_tensor.grad, reference_tensor.grad)

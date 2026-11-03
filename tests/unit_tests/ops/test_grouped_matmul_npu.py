# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

import types
from unittest import mock

import pytest
import torch

from fsdp_turbo.ops import moe
from fsdp_turbo.ops.cpu.grouped_matmul import grouped_matmul_cpu
from fsdp_turbo.ops.npu import grouped_matmul as gm


class _FakeNpuGroupedMatmul:
    """CPU stand-in for torch_npu.npu_grouped_matmul.

    Mirrors the semantics of the call shapes used by GroupedMatmul:
    split_item=2/group_type=0 is a grouped GEMM over per-expert row ranges;
    split_item=3/group_type=2 produces the per-expert weight gradient.
    ``group_list`` carries per-group row counts.
    """

    def __init__(self):
        self.calls = []

    def __call__(
        self, x_list, w_list, bias=None, group_list=None, split_item=None, group_type=None, group_list_type=None
    ):
        x, w = x_list[0], w_list[0]
        ends = torch.cumsum(group_list, dim=0).tolist()
        starts = [0] + ends[:-1]
        self.calls.append(
            {
                "x_shape": tuple(x.shape),
                "w_shape": tuple(w.shape),
                "w_object": w,
                "split_item": split_item,
                "group_type": group_type,
                "group_list_type": group_list_type,
            }
        )
        if split_item == 2 and group_type == 0:
            # x [M, K] @ w [G, K, N] -> y [M, N]
            out = torch.zeros(x.shape[0], w.shape[2], dtype=x.dtype)
            for g, (s, e) in enumerate(zip(starts, ends)):
                out[s:e] = x[s:e] @ w[g]
            return [out]
        if split_item == 3 and group_type == 2:
            # x [K, M] (inp.T) with w [M, N] (grad) -> [G, K, N]
            gw = torch.zeros(len(group_list), x.shape[0], w.shape[1], dtype=w.dtype)
            for g, (s, e) in enumerate(zip(starts, ends)):
                gw[g] = x[:, s:e] @ w[s:e, :]
            return [gw]
        raise AssertionError(f"unexpected op mode: split_item={split_item}, group_type={group_type}")


def _run_grouped_matmul(x, weights, counts, grad_out, weights_layout):
    fake = _FakeNpuGroupedMatmul()
    fake_module = types.SimpleNamespace(npu_grouped_matmul=fake)
    x = x.detach().clone().requires_grad_(True)
    weights = weights.detach().clone().requires_grad_(True)
    with mock.patch.object(gm, "torch_npu", fake_module):
        y = gm.GroupedMatmul.apply(x, weights, None, counts, 1, weights_layout)
        dx, dw = torch.autograd.grad(y, (x, weights), grad_out)
    return y, dx, dw, fake, weights


def _run_cpu_reference(x, weights, counts, grad_out, weights_layout):
    x = x.detach().clone().requires_grad_(True)
    weights = weights.detach().clone().requires_grad_(True)
    y = grouped_matmul_cpu(x, counts, weights, weights_layout=weights_layout)
    dx, dw = torch.autograd.grad(y, (x, weights), grad_out)
    return y, dx, dw


def _assert_matches_cpu(x, weights, counts, grad_out, weights_layout):
    y, dx, dw, fake, w_leaf = _run_grouped_matmul(x, weights, counts, grad_out, weights_layout)
    y_ref, dx_ref, dw_ref = _run_cpu_reference(x, weights, counts, grad_out, weights_layout)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(dx, dx_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(dw, dw_ref, rtol=1e-5, atol=1e-6)
    assert dw.shape == weights.shape
    return fake, w_leaf


def test_input_major_matches_cpu_reference():
    torch.manual_seed(0)
    G, K, N = 3, 5, 7
    counts = torch.tensor([4, 7, 2])
    M = int(counts.sum())
    x = torch.randn(M, K)
    w_in = torch.randn(G, K, N)  # input-major, Kimi's storage layout
    grad_out = torch.randn(M, N)

    fake, w_leaf = _assert_matches_cpu(x, w_in, counts, grad_out, "input_major")

    # input-major weights reach the op without any copy inside forward
    assert fake.calls[0]["w_shape"] == (G, K, N)
    assert fake.calls[0]["w_object"] is w_leaf
    assert fake.calls[0]["group_list_type"] == 1


def test_linear_layout_matches_cpu_reference():
    torch.manual_seed(1)
    G, K, N = 2, 6, 4
    counts = torch.tensor([5, 3])
    M = int(counts.sum())
    x = torch.randn(M, K)
    w_lin = torch.randn(G, N, K)  # linear convention [out, in]
    grad_out = torch.randn(M, N)

    fake, _ = _assert_matches_cpu(x, w_lin, counts, grad_out, "linear")

    assert fake.calls[0]["w_shape"] == (G, K, N)  # op sees input-major


def test_square_linear_layout_matches_cpu_reference():
    # Regression: a square linear weight must stay on the linear convention
    # (x @ w.T); the removed shape heuristic treated it as input-major and
    # silently computed x @ w with wrong gradients.
    torch.manual_seed(2)
    G, D = 2, 4  # K == N: shape-ambiguous
    counts = torch.tensor([3, 3])
    M = int(counts.sum())
    x = torch.randn(M, D)
    w_lin = torch.randn(G, D, D)
    grad_out = torch.randn(M, D)

    _assert_matches_cpu(x, w_lin, counts, grad_out, "linear")


def test_four_dim_legacy_layout_is_supported():
    torch.manual_seed(4)
    G, K, N = 2, 3, 5
    counts = torch.tensor([2, 4])
    M = int(counts.sum())
    x = torch.randn(M, K)
    w_4d = torch.randn(G, 1, N, K)
    grad_out = torch.randn(M, N)

    y, dx, dw, fake, _ = _run_grouped_matmul(x, w_4d, counts, grad_out, "linear")

    # Dense reference for the linear convention, flattened to [G, N, K].
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w_4d.reshape(G, N, K).detach().clone().requires_grad_(True)
    y_ref = torch.zeros(M, N)
    start = 0
    for g in range(G):
        end = start + int(counts[g])
        y_ref[start:end] = x_ref[start:end] @ w_ref[g].transpose(0, 1)
        start = end
    dx_ref, dw_ref = torch.autograd.grad(y_ref, (x_ref, w_ref), grad_out)

    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(dx, dx_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(dw.reshape(G, N, K), dw_ref, rtol=1e-5, atol=1e-6)
    assert fake.calls[0]["w_shape"] == (G, K, N)
    assert dw.shape == w_4d.shape


def test_invalid_weights_layout_is_rejected():
    with pytest.raises(ValueError, match="weights_layout"):
        moe.grouped_matmul(
            torch.randn(4, 3),
            [2, 2],
            torch.randn(2, 3, 3),
            use_eager=True,
            weights_layout="KN",
        )

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Run on Ascend to exercise the actual SLIG backward without SparseFlashMla."""

import pytest
import torch

from fsdp_turbo.ops.cpu.sparse_indexer_loss import sparse_indexer_gradient_torch


@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_real_slig_matches_torch(ratio, monkeypatch):
    pytest.importorskip("torch_npu")
    from fsdp_turbo.ops.npu import sparse_indexer_loss as npu

    def reject_fallback(*args):
        pytest.fail("expected the CANN SLIG kernel")

    monkeypatch.setattr(npu, "sparse_indexer_gradient_torch", reject_fallback)
    torch.manual_seed(19)
    length, dim = 582, 128
    q = torch.randn(1, length, 32, dim, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, length // ratio, dim, dtype=torch.bfloat16) * 0.1
    w = torch.randn(1, length, 32) * 0.1
    idx = torch.arange(64).expand(1, length, -1)
    idx = idx.masked_fill(idx >= (torch.arange(length)[:, None] + 1) // ratio, -1).int()
    target = torch.rand(idx.shape).masked_fill(idx < 0, 0)
    results = []
    for device, op in [
        ("cpu", sparse_indexer_gradient_torch),
        ("npu", npu.sparse_indexer_gradient_npu),
    ]:
        inputs = [x.to(device).requires_grad_() for x in (q, k, w)]
        loss = op(*inputs, idx.to(device), target.to(device), ratio)
        results.append([g.float().cpu() for g in torch.autograd.grad(loss, inputs)])
    failures = []
    for name, actual, expected in zip(("dq", "dk", "dw"), results[1], results[0]):
        assert actual.shape == expected.shape, name
        assert torch.isfinite(actual).all(), f"{name}: nonfinite NPU gradient"
        assert expected.norm() > 0, f"{name}: reference gradient is zero"
        relative_error = (actual - expected).norm() / expected.norm()
        if relative_error >= 0.05:
            failures.append(
                f"ratio={ratio} {name}: relative_L2={relative_error.item():.6g}, "
                f"NPU_norm={actual.norm().item():.6g}, reference_norm={expected.norm().item():.6g}"
            )
    assert not failures, "\n".join(failures)

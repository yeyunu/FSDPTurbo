"""Check the reporting scalar separately from attention and SLIG gradients."""

import pytest
import torch

from fsdp_turbo.ops.indexer_kl_metric import indexer_kl_metric


def test_kl_metric_matches_reference() -> None:
    """Cover non-unit teacher mass, an empty row, token sum, and graph isolation."""
    raw = torch.tensor([[[[0.0, 0.0]], [[3.0, 1.0]]]], requires_grad=True)
    student = torch.tensor([[[[0.0, 0.0]], [[0.4, 0.6]]]], requires_grad=True)
    teacher = torch.nn.functional.normalize(raw.detach(), p=1, dim=-1)
    p = teacher / (teacher.sum(-1, keepdim=True) + 1e-9)
    expected = (teacher * (p.clamp_min(1e-9).log() - (student.detach() + 1e-9).log())).sum()
    actual = indexer_kl_metric(raw, student)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


@pytest.fixture(name="native_kl_kernel")
def _native_kl_kernel():
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("An available Ascend NPU is required")
    from fsdp_turbo.utils.device import set_accelerator_compatible

    try:
        set_accelerator_compatible()
    except RuntimeError:
        pytest.skip("A2/A3 or Ascend 950DT NPU required")
    device = torch.accelerator.current_accelerator()
    if device is None or device.type != "npu":
        pytest.skip("A2/A3 or Ascend 950DT NPU required")
    from fsdp_turbo.ops.npu.sparse_indexer_loss import _supports_indexer_kl_kernel

    if not _supports_indexer_kl_kernel(device.index):
        pytest.skip("A2/A3 or Ascend 950DT NPU required")
    package = pytest.importorskip("cann_ops_transformer")
    kernel = getattr(package, "lightning_indexer_kl_loss", None)
    if not callable(kernel):
        pytest.skip("CANN build does not provide lightning_indexer_kl_loss")
    return device, kernel


def test_fused_kl_sum_on_supported_npu(monkeypatch, native_kl_kernel) -> None:
    """Call the real scalar kernel directly so a fallback cannot pass this test."""

    torch.manual_seed(17)
    teacher = torch.nn.functional.normalize(torch.rand(2, 8, 512), p=1, dim=-1)
    student = torch.randn_like(teacher).softmax(-1)
    teacher[0, 0].zero_()
    expected = indexer_kl_metric(teacher, student)
    from fsdp_turbo.ops.npu import sparse_indexer_loss as npu

    monkeypatch.setattr(npu, "indexer_kl_sum_torch", lambda *args: pytest.fail("expected native KLLOSS"))
    device, kernel = native_kl_kernel
    raw = kernel(teacher.to(device), student.to(device), eps=1e-9, weight_type="logits")
    torch.testing.assert_close(raw.reshape(()).cpu(), expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        indexer_kl_metric(teacher.to(device), student.to(device)).cpu(), expected, atol=1e-5, rtol=1e-4
    )

    from fsdp_turbo.ops import registry

    with monkeypatch.context() as patch:
        patch.setitem(
            registry._OPERATOR_REGISTRY["indexer_kl_sum"],
            "npu",
            lambda *args: pytest.fail("disabled KL fusion must bypass the NPU kernel"),
        )
        torch.testing.assert_close(
            indexer_kl_metric(teacher.to(device), student.to(device), use_fused=False).cpu(),
            expected,
            atol=1e-5,
            rtol=1e-4,
        )


def test_cpu_metric_does_not_change_gradients():
    from fsdp_turbo.ops.indexer_kl_metric import collect_indexer_kl_metrics
    from fsdp_turbo.ops.cpu.sparse_indexer_loss import sparse_indexer_gradient_torch

    torch.manual_seed(42)
    q = torch.randn(1, 37, 2, 4, requires_grad=True)
    k = torch.randn(1, 3, 4, requires_grad=True)
    w = torch.randn(1, 37, 2, requires_grad=True)
    idx = torch.arange(3).expand(1, 37, 3)
    target = torch.rand(1, 37, 3)
    target[:, -1] = 0

    def run():
        return sparse_indexer_gradient_torch(q, k, w, idx, target, 1)

    baseline = torch.autograd.grad(run(), (q, k, w))
    with collect_indexer_kl_metrics() as metrics:
        carrier = run()
    for actual, ref in zip(torch.autograd.grad(carrier, (q, k, w)), baseline):
        torch.testing.assert_close(actual, ref)
    scores = torch.einsum("bshd,bkd->bshk", q, k).relu()
    probs = (scores * w.unsqueeze(-1)).sum(2).softmax(-1)
    metrics.scale = 0.1 / 36
    torch.testing.assert_close(metrics.value(carrier), indexer_kl_metric(target, probs) * (0.1 / 36))


@pytest.mark.usefixtures("native_kl_kernel")
def test_950dt_kl_matches_across_gradient_backends(monkeypatch):
    try:
        from cann_ops_transformer import ops
    except ImportError:
        pytest.skip("CANN build does not provide SLIG ops")
    for name in ("sparse_lightning_indexer_kl_loss_grad", "sparse_lightning_indexer_kl_loss_grad_metadata"):
        if not callable(getattr(ops, name, None)):
            pytest.skip(f"CANN build does not provide {name}")
    from fsdp_turbo.ops.indexer_kl_metric import collect_indexer_kl_metrics
    from fsdp_turbo.ops.cpu.sparse_indexer_loss import sparse_indexer_gradient_torch
    from fsdp_turbo.ops.npu import sparse_indexer_loss as npu

    def reject_fallback(*args):
        pytest.fail("expected the requested CANN kernel on NPU")

    monkeypatch.setattr(npu, "indexer_kl_sum_torch", reject_fallback)
    monkeypatch.setattr(npu, "sparse_indexer_gradient_torch", reject_fallback)
    torch.manual_seed(42)
    length = 128
    q = (torch.randn(1, length, 32, 128) * 0.1).bfloat16()
    k = (torch.randn(1, length, 128) * 0.1).bfloat16()
    w = torch.randn(1, length, 32) * 0.1
    ids = torch.arange(64).expand(1, length, -1)
    ids = ids.masked_fill(ids > torch.arange(length)[:, None], -1).int()
    target = torch.rand(ids.shape).masked_fill(ids < 0, 0)
    values = []
    for device, op, use_fused in [
        ("cpu", sparse_indexer_gradient_torch, False),
        ("npu", sparse_indexer_gradient_torch, True),
        ("npu", npu.sparse_indexer_gradient_npu, True),
        ("npu", npu.sparse_indexer_gradient_npu, False),
    ]:
        inputs = [x.to(device).requires_grad_() for x in (q, k, w)]
        with collect_indexer_kl_metrics(use_fused=use_fused) as metrics:
            term = op(*inputs, ids.to(device), target.to(device), 1)
        term.backward()
        assert metrics.sums
        values.append(metrics.value(term).cpu() / length)
    assert values[0].item() > 0
    for actual in values[1:]:
        torch.testing.assert_close(actual, values[0], rtol=0.03, atol=1e-4)

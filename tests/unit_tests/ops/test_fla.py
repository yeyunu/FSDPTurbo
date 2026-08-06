import pytest
import torch


def test_fla_ops_delegate_through_device_registry(monkeypatch):
    fla_gdn = pytest.importorskip("fla.ops.gated_delta_rule")
    import fsdp_turbo.ops.fla  # noqa: F401
    from fsdp_turbo.ops import registry
    from fsdp_turbo.ops.registry import get_op

    calls = []

    def fake_chunk(*args, **kwargs):
        calls.append((args, kwargs))
        return "chunk"

    def fake_recurrent(*args, **kwargs):
        calls.append((args, kwargs))
        return "recurrent"

    monkeypatch.setattr(fla_gdn, "chunk_gated_delta_rule", fake_chunk)
    monkeypatch.setattr(fla_gdn, "fused_recurrent_gated_delta_rule", fake_recurrent)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cpu"))
    monkeypatch.setitem(
        registry._OPERATOR_REGISTRY["chunk_gated_delta_rule"],
        "cpu",
        registry._OPERATOR_REGISTRY["chunk_gated_delta_rule"]["npu"],
    )
    monkeypatch.setitem(
        registry._OPERATOR_REGISTRY["fused_recurrent_gated_delta_rule"],
        "cpu",
        registry._OPERATOR_REGISTRY["fused_recurrent_gated_delta_rule"]["npu"],
    )

    assert get_op("chunk_gated_delta_rule")("q") == "chunk"
    assert calls == [(("q",), {})]
    assert get_op("fused_recurrent_gated_delta_rule")("k", mode="fused") == "recurrent"
    assert calls[-1] == (("k",), {"mode": "fused"})

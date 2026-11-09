# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for the opt-in Kimi routing diagnostics (``router_debug.py``).

CPU only, no process group required: the entry points are default-off and
must stay inert without a ``gate._router_debug`` handle, and the synthetic
count math is pure Python/torch.
"""

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.fsdp_turbo_config import KimiRouterDebugConfig  # noqa: E402

try:  # noqa: SIM105 - the plain package path needs transformers
    from fsdp_turbo.models.kimi.router_debug import (  # noqa: E402
        _exact_counts,
        _synthetic_indices,
        custom_routing,
        record_load,
    )
except ModuleNotFoundError as error:  # e.g. no transformers: load the file directly
    if error.name not in ("transformers", "fsdp_turbo.models.kimi.router_debug"):
        raise
    import importlib.util
    from pathlib import Path

    _path = Path(__file__).parents[4] / "fsdp_turbo" / "models" / "kimi" / "router_debug.py"
    _spec = importlib.util.spec_from_file_location("router_debug_under_test", _path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    _exact_counts = _module._exact_counts
    _synthetic_indices = _module._synthetic_indices
    custom_routing = _module.custom_routing
    record_load = _module.record_load


class _Gate:
    """The KimiMoEGate surface that router_debug touches."""

    def __init__(self, num_experts=8, top_k=2, debug=None, layer=0):
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = 2.5
        self._router_debug = debug
        self._router_debug_layer = layer


def test_exact_counts_conserve_the_token_budget():
    for experts, top_k, tokens, target in [
        (8, 2, 512, 0.0),
        (8, 2, 512, 0.5),
        (16, 4, 1000, 1.0),
        (4, 4, 96, 0.0),
        (8, 2, 3, 0.0),
    ]:
        counts = _exact_counts(tokens, experts, top_k, target)
        assert len(counts) == experts
        assert sum(counts) == tokens * top_k, (experts, top_k, tokens)
        assert max(counts) <= tokens


def test_exact_counts_hit_the_requested_peak():
    experts, top_k, tokens = 8, 2, 512
    for target in (0.0, 0.25, 0.5, 1.0):
        counts = _exact_counts(tokens, experts, top_k, target)
        peak = max(counts)
        mean = sum(counts) / experts
        assert abs(peak / mean - 1 - target) < 1 / mean + 1e-9


def test_synthetic_indices_every_row_has_top_k_distinct_experts():
    gate = _Gate(num_experts=8, top_k=2)
    indices = _synthetic_indices(gate, tokens=64, target=0.0, mode="exact", device="cpu")
    assert indices.shape == (64, 2)
    assert indices.dtype == torch.int64
    # Each row must hold distinct experts; the flattened histogram conserves
    # the total budget.
    assert torch.all(indices[:, 0] != indices[:, 1])
    counts = torch.bincount(indices.flatten(), minlength=8)
    assert int(counts.sum()) == 64 * 2


def test_synthetic_indices_are_deterministic_per_layer_and_rank():
    gate_a = _Gate(num_experts=8, top_k=2, layer=3)
    gate_b = _Gate(num_experts=8, top_k=2, layer=3)
    first = _synthetic_indices(gate_a, tokens=32, target=0.5, mode="lognormal", device="cpu")
    second = _synthetic_indices(gate_b, tokens=32, target=0.5, mode="lognormal", device="cpu")
    torch.testing.assert_close(first, second)


def test_custom_routing_is_inert_without_a_debug_handle():
    gate = _Gate(debug=None)
    hidden = torch.randn(16, 8)
    assert custom_routing(gate, hidden) is None


def test_custom_routing_is_inert_when_not_applied():
    gate = _Gate(debug=KimiRouterDebugConfig(apply_custom_router=False))
    assert custom_routing(gate, torch.randn(16, 8)) is None


def test_custom_routing_is_inert_when_recording_only():
    gate = _Gate(debug=KimiRouterDebugConfig(record_load_balance=True))
    assert custom_routing(gate, torch.randn(16, 8)) is None


def test_custom_routing_returns_the_declared_shape_and_weights():
    debug = KimiRouterDebugConfig(apply_custom_router=True, target_max_vio=0.0, router_mode="exact")
    gate = _Gate(num_experts=8, top_k=2, debug=debug)
    indices, weights = custom_routing(gate, torch.randn(32, 8))
    assert indices.shape == (32, 2)
    assert weights.shape == (32, 2)
    torch.testing.assert_close(weights, torch.full((32, 2), 2.5 / 2))


def test_custom_routing_warns_once_per_gate():
    debug = KimiRouterDebugConfig(apply_custom_router=True, target_max_vio=0.0, router_mode="exact")
    gate = _Gate(num_experts=8, top_k=2, debug=debug)
    custom_routing(gate, torch.randn(8, 8))
    assert getattr(gate, "_router_debug_warned", False) is True
    # The flag short-circuits the warning, so a second call leaves it set.
    custom_routing(gate, torch.randn(8, 8))
    assert getattr(gate, "_router_debug_warned", False) is True


def test_custom_routing_rejects_a_bad_mode_via_config():
    with pytest.raises(ValueError, match="router_mode"):
        KimiRouterDebugConfig(apply_custom_router=True, target_max_vio=0.5, router_mode="uniform")


def test_record_load_is_silent_without_the_knob(caplog):
    gate = _Gate(debug=KimiRouterDebugConfig())
    with caplog.at_level("INFO"):
        record_load(gate, torch.zeros(4, 2, dtype=torch.int64))
    assert caplog.text == ""


def test_record_load_reports_local_counts(caplog):
    debug = KimiRouterDebugConfig(record_load_balance=True)
    gate = _Gate(num_experts=8, top_k=2, debug=debug, layer=5)
    indices = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
    with caplog.at_level("INFO"):
        record_load(gate, indices)
    assert "[Layer 5]" in caplog.text
    assert "tokens_per_expert=[2, 1, 1, 0, 0, 0, 0, 0]" in caplog.text


def test_record_load_reports_global_maxvio_without_a_group(caplog):
    debug = KimiRouterDebugConfig(record_load_balance=True)
    gate = _Gate(num_experts=4, top_k=2, debug=debug, layer=1)
    indices = torch.tensor([[0, 0], [0, 0]], dtype=torch.int64)
    with caplog.at_level("INFO"):
        record_load(gate, indices)
    assert "global maxvio=3.000000" in caplog.text


def test_config_rejects_a_negative_target():
    with pytest.raises(ValueError, match="target_max_vio"):
        KimiRouterDebugConfig(apply_custom_router=True, target_max_vio=-0.5)


def test_config_off_mode_tolerates_any_target():
    # apply_custom_router=False must not validate the other fields.
    KimiRouterDebugConfig(apply_custom_router=False, target_max_vio=-99.0, router_mode="anything")

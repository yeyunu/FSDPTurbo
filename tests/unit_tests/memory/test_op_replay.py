# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for op replay (CPU-only: a stub SwapCache stands in for the
device-backed cache; dispatch modes work on CPU tensors).
"""

import inspect
import logging

import pytest
import torch
from torch import nn

from fsdp_turbo.memory.op_replay.op_replay import (
    OpReplayController,
    OpReplayPatcher,
    OpReplayScope,
    OpReplayScopeConfig,
    OpReplaySession,
    _get_rng_snapshot,
    _parse_op_list,
    _set_rng_snapshot,
    _default_cache_ops,
    apply_op_replay_inplace,
    build_op_replay_context_fn,
)


class _StubCache:
    """Duck-typed SwapCache stand-in: put/pop over an in-memory dict."""

    def __init__(self):
        self.store = {}
        self._next = 0

    def put(self, tensor):
        handle = f"h{self._next}"
        self._next += 1
        self.store[handle] = tensor
        return handle

    def pop(self, handle):
        return self.store.pop(handle, None)


class TestOpListParsing:
    def test_none_passthrough(self):
        assert _parse_op_list(None) is None

    def test_valid_names_resolve(self):
        ops = _parse_op_list(["aten.mm.default", "aten.addmm.default"])
        assert ops == {torch.ops.aten.mm.default, torch.ops.aten.addmm.default}

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Invalid op-replay op name"):
            _parse_op_list(["aten.no_such_op_xyz.default"])

    def test_default_whitelist_nonempty(self):
        ops = _default_cache_ops()
        # The aten SDPA entry resolves on every platform (the npu.* entry is
        # skipped tolerantly off-NPU), so this holds in CPU-only UTs too.
        assert torch.ops.aten._scaled_dot_product_flash_attention.default in ops


class TestScope:
    def test_none_whitelist_falls_back_to_default(self):
        scope = OpReplayScope(cache_ops=None)
        assert torch.ops.aten._scaled_dot_product_flash_attention.default in scope.cache_ops

    def test_explicit_empty_whitelist_stays_empty(self):
        scope = OpReplayScope(cache_ops=set())
        assert scope.cache_ops == set()

    def test_from_config(self):
        cfg = OpReplayScopeConfig(cache_ops=["aten.mm.default"], save_rng=False, name="cfg")
        scope = OpReplayScope.from_config(cfg)
        assert scope.cache_ops == {torch.ops.aten.mm.default}
        assert scope.save_rng is False
        assert scope.name == "cfg"

    def test_from_config_defaults(self):
        scope = OpReplayScope.from_config(OpReplayScopeConfig())
        assert torch.ops.aten._scaled_dot_product_flash_attention.default in scope.cache_ops
        assert scope.save_rng is True


class TestRngSnapshot:
    def test_cpu_roundtrip(self):
        torch.manual_seed(0)
        _get_rng_snapshot()
        snap = _get_rng_snapshot()
        expected = torch.randn(4)
        _set_rng_snapshot(snap)
        assert torch.equal(torch.randn(4), expected)


class TestZoneStack:
    def test_innermost_wins(self):
        s1, s2 = OpReplayScope(name="s1"), OpReplayScope(name="s2")
        controller = OpReplayController([s1, s2], _StubCache())
        session = OpReplaySession(controller)
        assert session.zone_scope is None
        with controller.replay_zone(s1):
            assert session.zone_scope is s1
            with controller.replay_zone(s2):
                assert session.zone_scope is s2
            assert session.zone_scope is s1
        assert session.zone_scope is None


class TestBuildContextFnGuards:
    def test_empty_scopes_disables(self):
        assert build_op_replay_context_fn(nn.Linear(2, 2), [], False, None) is None

    def test_reentrant_checkpoint_skips(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert build_op_replay_context_fn(nn.Linear(2, 2), [OpReplayScopeConfig()], True, None) is None
        assert "use_reentrant" in caplog.text

    def test_missing_cache_raises(self):
        with pytest.raises(ValueError, match="shared SwapCache"):
            build_op_replay_context_fn(nn.Linear(2, 2), [OpReplayScopeConfig()], False, None)


class TestPatcher:
    def test_patch_and_first_scope_wins(self):
        model = nn.Sequential(nn.Linear(2, 2))
        controller = OpReplayController([], _StubCache())
        s1, s2 = OpReplayScope(name="s1"), OpReplayScope(name="s2")
        apply_op_replay_inplace(model, ["0"], controller, s1)
        assert isinstance(model[0].forward, OpReplayPatcher)
        apply_op_replay_inplace(model, ["0"], controller, s2)
        assert model[0].forward.scope is s1

    def test_signature_exposed(self):
        model = nn.Linear(2, 2)
        controller = OpReplayController([], _StubCache())
        apply_op_replay_inplace(model, [""], controller, OpReplayScope())
        assert "input" in str(inspect.signature(model.forward))


class TestCachingAndReplaying:
    def _make(self, save_rng=True):
        cache = _StubCache()
        scope = OpReplayScope(cache_ops={torch.ops.aten.mm.default}, save_rng=save_rng, name="s")
        controller = OpReplayController([scope], cache)
        caching, replaying = controller.make_context_fn()()
        return cache, scope, controller, caching, replaying

    def test_route_recorded_and_compute_skipped(self):
        cache, scope, controller, caching, replaying = self._make()
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        with caching, controller.replay_zone(scope):
            out_fwd = torch.mm(a, b)
        assert len(cache.store) == 1

        a2, b2 = torch.randn(4, 4), torch.randn(4, 4)
        with replaying, controller.replay_zone(scope):
            out_rep = torch.mm(a2, b2)
        assert torch.equal(out_rep, out_fwd)
        assert not torch.equal(out_rep, torch.mm(a2, b2))

    def test_ignored_ops_pass_through(self):
        cache, scope, controller, caching, _ = self._make()
        with caching, controller.replay_zone(scope):
            torch.randn(2, 2).detach()
        assert len(cache.store) == 0

    def test_replay_restores_post_op_rng(self):
        cache, scope, controller, caching, replaying = self._make(save_rng=True)
        torch.manual_seed(42)
        with caching, controller.replay_zone(scope):
            torch.mm(torch.randn(2, 2), torch.randn(2, 2))
        r_fwd = torch.randn(3)

        torch.manual_seed(42)
        with replaying, controller.replay_zone(scope):
            torch.mm(torch.randn(2, 2), torch.randn(2, 2))
        r_rep = torch.randn(3)
        assert torch.equal(r_fwd, r_rep)

    def test_miss_falls_back_to_recompute(self):
        cache, scope, controller, caching, replaying = self._make()
        a, b = torch.randn(2, 2), torch.randn(2, 2)
        with caching, controller.replay_zone(scope):
            torch.mm(a, b)
        cache.store.clear()  # simulate mid-step clear(): handle orphaned

        a2, b2 = torch.randn(2, 2), torch.randn(2, 2)
        with pytest.warns(RuntimeWarning, match="cache miss"):
            with replaying, controller.replay_zone(scope):
                out = torch.mm(a2, b2)
        assert torch.equal(out, torch.mm(a2, b2))  # recomputed ground truth

"""Pure-CPU tests for SwapPlanConfig validation and SwapManager's device-free
surface. SwapManager.get_cache() is deliberately NOT exercised here: cache
construction fast-fails without a usable device.
"""

import pytest

from fsdp_turbo.memory.swap.swap_manager import SwapManager, SwapPlanConfig


class TestSwapPlanConfigDefaults:
    def test_defaults(self):
        plan = SwapPlanConfig()
        assert plan.capacity_mb == 1024.0
        assert plan.keep_rate == 0.0
        assert plan.enable_prefetch is True
        assert plan.cpu_arena_slab_size_mb == 2048.0
        assert plan.cpu_arena_pool_policy == "standard"
        assert plan.pin_memory is True
        assert plan.eviction_policy == "fifo"


class TestSwapPlanConfigValidation:
    @pytest.mark.parametrize("bad_rate", [-0.1, 1.5])
    def test_keep_rate_out_of_range_raises(self, bad_rate):
        with pytest.raises(ValueError, match="keep_rate"):
            SwapPlanConfig(keep_rate=bad_rate)

    @pytest.mark.parametrize("boundary", [0.0, 1.0])
    def test_keep_rate_boundaries_ok(self, boundary):
        assert SwapPlanConfig(keep_rate=boundary).keep_rate == boundary

    def test_eviction_policy_lru_rejected(self):
        with pytest.raises(ValueError, match="eviction_policy"):
            SwapPlanConfig(eviction_policy="lru")

    def test_eviction_policy_belady_ok(self):
        assert SwapPlanConfig(eviction_policy="belady").eviction_policy == "belady"

    def test_eviction_policy_fifo_ok(self):
        assert SwapPlanConfig(eviction_policy="fifo").eviction_policy == "fifo"


class TestSwapManagerWithoutDevice:
    def test_lazy_cache_not_constructed(self):
        mgr = SwapManager(SwapPlanConfig())
        assert mgr._cache is None
        assert not mgr._tenants

    def test_step_end_without_cache_is_noop(self):
        mgr = SwapManager(SwapPlanConfig())
        mgr.step_end()  # must not touch any device
        assert mgr._cache is None

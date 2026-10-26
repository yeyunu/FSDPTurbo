# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""SwapManager: single owner of the shared swap cache and its lifecycle.

Holds the one SwapCache (swap stream + pinned arena + capacity budget) built
from a SwapPlanConfig. Memory features obtain it via get_cache(tenant) and
use it as pure tenants: they never construct caches and never delimit
iterations. The caller drives step_end() once per iteration, typically as a
step-end callback after optimizer.zero_grad(); without it, prefetch never
arms, stats are not archived and the put history grows without bound
(correctness is unaffected).
"""

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from .slab_arena import SlabArena
from .swap_core import SwapCache
from ...utils.log import print_rank


logger = logging.getLogger(__name__)


@dataclass
class SwapPlanConfig:
    """Physical resources of the shared swap cache (HBM<->DDR async swap),
    owned by the SwapManager. Memory features obtain the shared cache from
    the manager as pure tenants.

    Validation lives in __post_init__, so every construction path is checked.

    Execution is always asynchronous (swap stream + event ordering, no host
    block); hosts without a usable device fail fast at cache construction.
    capacity_mb selects the mode: <0 no management (pass-through), =0
    deterministic eviction (the compute stream waits the swap stream), >0
    asynchronous capacity management (default 1024).
    """

    capacity_mb: float = field(
        default=1024.0,
        metadata={
            "help": "HBM capacity budget (MiB) for swapped payloads, shared by all tenants. <0: no capacity management (pass-through, nothing is evicted); =0: every put evicts and the compute stream waits the swap stream (deterministic, no overlap); >0: asynchronous capacity management. Default 1024; tune to the per-step swap volume (a stash aiming to save HBM needs capacity below its evicted volume)."
        },
    )
    keep_rate: float = field(
        default=0.0,
        metadata={
            "help": "HBM retention ratio [0,1]: the soft limit is capacity x keep_rate, the line above which proactive eviction starts (projected reading, after in-flight D2H). 0 = evict right after put (max async D2H overlap); 1 = retain the full capacity budget, evict only when physically over capacity."
        },
    )
    enable_prefetch: bool = field(
        default=True,
        metadata={
            "help": "Whether enable transition-graph chain prefetch for the swap cache. Turn off (with fifo eviction) for structurally variable workloads: the graph is learned once from the first iteration's pop order."
        },
    )
    cpu_arena_slab_size_mb: Optional[float] = field(
        default=2048.0,
        metadata={
            "help": "CPU slab arena slab size (MB). None selects the on-demand arena (per-tensor exact allocation, released immediately)."
        },
    )
    cpu_arena_pool_policy: str = field(
        default="standard",
        metadata={
            "help": "CPU arena pool policy: 'all' (retain all empty slabs), 'standard' (retain standard slabs, release oversized on empty), 'none' (release immediately). Only effective with cpu_arena_slab_size_mb."
        },
    )
    pin_memory: bool = field(
        default=True,
        metadata={
            "help": "Whether arena slabs use pinned (page-locked) memory. False is the escape hatch when host pinned memory is exhausted; non-pinned copies then execute synchronously (torch behavior)."
        },
    )
    eviction_policy: Literal["fifo", "belady"] = field(
        default="fifo",
        metadata={
            "help": "Swap cache eviction policy: 'fifo' or 'belady'. With keep_rate=0 (default) every put evicts everything and the two policies are equivalent; belady matters when keep_rate > 0 (evict the farthest-future use)."
        },
    )

    def __post_init__(self):
        if not (0.0 <= self.keep_rate <= 1.0):
            raise ValueError(f"keep_rate must be in [0, 1], got {self.keep_rate}")
        if self.eviction_policy not in ("fifo", "belady"):
            raise ValueError(f"eviction_policy must be 'fifo' or 'belady', got {self.eviction_policy}")


class SwapManager:
    """Owns the shared SwapCache and its iteration boundary.

    get_cache(tenant) returns the single shared instance (created lazily on
    the first call); the tenant name is a registration label for logging only.
    step_end() delimits one iteration (cache.clear(): transition-graph
    learning + stats archival + resource release) and must be called exactly
    once per iteration — a second call within the same iteration archives
    empty stats over the real ones.
    """

    def __init__(self, plan: SwapPlanConfig):
        self._plan = plan
        self._cache = None
        self._tenants = []

    def get_cache(self, tenant: str = "default") -> SwapCache:
        if self._cache is None:
            # capacity_mb < 0 -> pass-through (None); 0 -> deterministic
            # eviction; >0 -> async capacity management (default 1024).
            capacity_mb = self._plan.capacity_mb
            capacity_bytes = None if capacity_mb < 0 else int(capacity_mb * 1024 * 1024)
            soft_limit_bytes = int(capacity_bytes * self._plan.keep_rate) if capacity_bytes is not None else 0
            # The arena is composed here, not inside SwapCache: pooled when a
            # slab size is configured (empty slabs kept for reuse), otherwise
            # the on-demand arena (exact per-tensor allocation).
            slab_size_mb = self._plan.cpu_arena_slab_size_mb
            pin_memory = self._plan.pin_memory
            if slab_size_mb is not None:
                arena = SlabArena(
                    slab_size_bytes=max(1, int(slab_size_mb * 1024 * 1024)),
                    device='cpu',
                    pin_memory=pin_memory,
                    pool_policy=self._plan.cpu_arena_pool_policy,
                )
            else:
                arena = SlabArena.on_demand(pin_memory=pin_memory)
            self._cache = SwapCache(
                capacity_bytes=capacity_bytes,
                soft_limit_bytes=soft_limit_bytes,
                cpu_arena=arena,
                eviction_policy=self._plan.eviction_policy,
                enable_prefetch=self._plan.enable_prefetch,
            )
        if tenant not in self._tenants:
            self._tenants.append(tenant)
            print_rank(logger.info, f"Swap cache shared with tenant: {tenant}")
        return self._cache

    def step_end(self):
        """Drive the per-step iteration boundary of the shared cache."""
        if self._cache is not None:
            self._cache.clear()

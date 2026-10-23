# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""HBM<->DDR asynchronous swap in/out foundation (SwapCore).

- SwapHandle: per-tensor HBM<->DDR lifecycle state machine with async copy support
- SwapCache:  SwapHandle collection management, FIFO/Belady eviction + capacity
  limits (hard capacity + soft limit) + hit/prefetch statistics

Execution model: a single async path. Every copy is issued on the swap
stream(s) and ordered via events anchored at the put-time stream (see
"Stream anchoring" below; device-side ordering, no host block). There is
no synchronous copy path: on a host
without a usable device the cache refuses to construct (fast fail) — swap is
meaningless without a device. capacity_bytes decides the sync point:
None (or <0) = no capacity management (pass-through, nothing is evicted);
0 = every put evicts and the compute stream waits for the swap stream
(deterministic, no overlap); >0 = normal asynchronous capacity management.

Stream anchoring: put captures the calling stream as `_put_stream` and
records `_put_event` on it — the contract point where the tensor's content
becomes final. D2H is ordered after `_put_event` (eviction-time ambient is
untrustworthy). Release ordering anchors `_put_stream` (the HBM block
returns to its allocation pool, and the allocator binds freed blocks to the
allocation stream's pool), never the release-time ambient; consume ordering
anchors the consume-time calling stream. swap_in lands in the `_put_stream`
pool with the H2D waiting `_put_stream`, and `_wait_h2d` settles on both
streams, so a zero-cost REPLICATED eviction needs no wait of its own.
"""

import itertools
import torch
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List

from .slab_arena import SlabArena, TensorBuffer


@dataclass(frozen=True)
class MemDelta:
    """Memory state delta (bytes). >0 means increase, <0 means decrease"""

    hbm: int = 0
    ddr: int = 0

    def __add__(self, other):
        return MemDelta(self.hbm + other.hbm, self.ddr + other.ddr)


class SwapState(Enum):
    HBM_ONLY = auto()
    DDR_ONLY = auto()
    # REPLICATED: both copies valid (device tensor + slab slot image, reached
    # after a completed load). Two zero-cost exits: release the HBM side ->
    # DDR_ONLY (eviction without a D2H); release the DDR side -> HBM_ONLY.
    # Invariant: the slot image equals the tensor's data, which holds because
    # a replicated tensor has not been handed out yet (pop is the handout).
    # Contract: a put tensor must not be mutated in place while its handle
    # is alive (get()-peek + in-place write would stale the slot image).
    REPLICATED = auto()
    D2H_IN_PROGRESS = auto()
    H2D_IN_PROGRESS = auto()
    EMPTY = auto()


class SwapHandle:
    """HBM<->DDR lifecycle of a single tensor, with async copy support

    cpu_arena: must be passed explicitly (held by SwapCache or an upper layer).
    The DDR intermediate layout is decided by the arena: a flat 1D copy of the
    copy window (layout fidelity for free); load (H2D) restores the original
    layout exactly via 1D DMA + as_strided.

    Swap streams are required (SwapCache always provides them); constructing a
    handle without streams is a programming error. All copies are issued on
    the swap stream(s) and ordered via events — there is no synchronous path.
    """

    _id_counter = itertools.count()  # process-wide monotonic id, for logging/debugging only

    def __init__(self, tensor, h2d_stream=None, d2h_stream=None, *, cpu_arena: SlabArena):
        self.tensor: Optional[torch.Tensor] = tensor
        self.id = next(SwapHandle._id_counter)
        self._cpu_arena = cpu_arena
        self._bytes = cpu_arena.payload_bytes(tensor)
        # H2D/D2H share one stream by default: on a half-duplex host link,
        # concurrent streams compete for bandwidth while a single stream
        # serializes both directions and eliminates cross-stream races.
        # Full-duplex hardware may pass two streams, with cross-stream buffer
        # reuse ordered via h2d/d2h events (see swap_out/swap_in).
        if h2d_stream is None and d2h_stream is None:
            raise RuntimeError(
                f"SwapHandle {self.id}: no swap streams given "
                "(h2d_stream/d2h_stream both None); construct via SwapCache."
            )
        if h2d_stream is None:
            h2d_stream = d2h_stream
        if d2h_stream is None:
            d2h_stream = h2d_stream
        self._h2d_stream = h2d_stream
        self._d2h_stream = d2h_stream
        self._dual_stream = h2d_stream is not d2h_stream
        self._h2d_event = torch.accelerator.Event()
        self._d2h_event = torch.accelerator.Event()
        # Put anchor: the calling stream plus an event recorded on it — the
        # contract point where the tensor's content becomes final (see "Stream
        # anchoring" in the module docstring).
        self._put_stream = torch.accelerator.current_stream()
        self._put_event = torch.accelerator.Event()
        self._put_event.record()
        self._state = SwapState.HBM_ONLY
        self._cpu_buf: Optional[TensorBuffer] = None
        self._d2h_issued = False
        self._h2d_issued = False

    def _ensure_cpu_buf(self):
        """Lazily register the arena buffer. Only reachable from HBM_ONLY
        (freshly put, never offloaded), where the buf is always None —
        re-offload after a load goes through REPLICATED's zero-cost path.
        """
        if self._cpu_buf is None:
            self._cpu_buf = self._cpu_arena.register(self.tensor)

    def _free_cpu_buf(self):
        """Free the arena buffer eagerly instead of relying on Python GC."""
        if self._cpu_buf is None:
            return
        self._cpu_arena.free(self._cpu_buf)
        self._cpu_buf = None

    def bytes(self) -> int:
        return self._bytes

    def state(self) -> SwapState:
        return self._state

    def _ensure_state(self, *allowed: SwapState) -> None:
        if self._state not in allowed:
            raise RuntimeError(f"Handle {self.id}: invalid state {self._state.name}")

    @property
    def hbm_bytes(self) -> int:
        if self._state in (
            SwapState.HBM_ONLY,
            SwapState.REPLICATED,
            SwapState.D2H_IN_PROGRESS,
            SwapState.H2D_IN_PROGRESS,
        ):
            return self._bytes
        return 0

    @property
    def ddr_bytes(self) -> int:
        if self._state in (
            SwapState.DDR_ONLY,
            SwapState.REPLICATED,
            SwapState.D2H_IN_PROGRESS,
            SwapState.H2D_IN_PROGRESS,
        ):
            return self._bytes
        return 0

    def is_available_in_hbm(self) -> bool:
        # In-flight states still hold a live HBM allocation (D2H reads the
        # valid original; H2D's landing is ordered via its event) — treating
        # them as unavailable would break chain-prefetch walks into bursts.
        return self._state in (
            SwapState.HBM_ONLY,
            SwapState.REPLICATED,
            SwapState.H2D_IN_PROGRESS,
            SwapState.D2H_IN_PROGRESS,
        )

    def can_offload(self) -> bool:
        return self._state in (SwapState.HBM_ONLY, SwapState.REPLICATED)

    def _wait_d2h(self) -> MemDelta:
        if self._state != SwapState.D2H_IN_PROGRESS:
            return MemDelta(0, 0)
        # Consume ordering settles on the calling stream; release ordering
        # anchors the put stream (freed blocks return to the allocation
        # stream's pool), never the release-time ambient.
        cur = torch.accelerator.current_stream()
        self._d2h_event.wait()
        if cur != self._put_stream:
            with torch.accelerator.stream(self._put_stream):
                self._d2h_event.wait()
        self.tensor = None
        self._state = SwapState.DDR_ONLY
        return MemDelta(hbm=-self._bytes, ddr=0)

    def _wait_h2d(self) -> MemDelta:
        if self._state != SwapState.H2D_IN_PROGRESS:
            return MemDelta(0, 0)
        # Dual settle on the calling stream and the put stream, so a later
        # zero-cost REPLICATED eviction needs no wait of its own.
        cur = torch.accelerator.current_stream()
        self._h2d_event.wait()
        if cur != self._put_stream:
            with torch.accelerator.stream(self._put_stream):
                self._h2d_event.wait()
        # REPLICATED: both copies stay accounted until consume or HBM eviction.
        self._state = SwapState.REPLICATED
        return MemDelta(hbm=0, ddr=0)

    def wait(self) -> MemDelta:
        """Settle in-flight copies. Must be called in the FINAL consume
        stream context: the settle edges land on the calling stream (and the
        put stream), so settling early on one stream and consuming on another
        leaves the consumer unordered.
        """
        return self._wait_d2h() + self._wait_h2d()

    def get(self) -> torch.Tensor:
        """Return the tensor wherever it currently lives (device or CPU).
        Calls wait() (same final-consume-stream requirement). A DDR_ONLY
        return value is the CPU `restored` view of the swap slot with no
        host-side guarantee that the D2H copy has physically completed —
        host code reading it directly must synchronize the device first
        (tests call _sync_device()). Tenants should prefer pop/consume,
        which land the payload on device with proper ordering.
        """
        self.wait()
        if self._state in (SwapState.HBM_ONLY, SwapState.REPLICATED):
            return self.tensor
        if self._state == SwapState.DDR_ONLY:
            if self._cpu_buf is None:
                raise RuntimeError(f"Handle {self.id}: no cpu buffer in DDR_ONLY")
            return self._cpu_buf.restored
        raise RuntimeError(f"Handle {self.id}: no tensor in state {self._state.name}")

    def swap_out(self) -> MemDelta:
        if self._state in (SwapState.DDR_ONLY, SwapState.D2H_IN_PROGRESS):
            return MemDelta(0, 0)
        if self._state == SwapState.H2D_IN_PROGRESS:
            raise RuntimeError(f"Handle {self.id}: cannot offload while load in progress")
        self._ensure_state(SwapState.HBM_ONLY, SwapState.REPLICATED)

        if self._state == SwapState.REPLICATED:
            # Zero-cost eviction: the slot image is still valid (untouched by
            # the read-only H2D) and the dual settle already ordered the HBM
            # release on both streams. Just drop the tensor.
            self.tensor = None
            self._state = SwapState.DDR_ONLY
            return MemDelta(hbm=-self._bytes, ddr=0)

        self._ensure_cpu_buf()

        # D2H is ordered after the put event, never the eviction-time ambient
        # (an unrelated put's context may fork off before this tensor's last
        # write); under the no-mutation-after-put contract no later write exists.
        with torch.accelerator.stream(self._d2h_stream):
            self._d2h_stream.wait_event(self._put_event)
            if self._dual_stream and self._h2d_issued:
                # Dual streams: the same cpu_buf may be read by an in-flight H2D
                self._d2h_stream.wait_event(self._h2d_event)
            self._cpu_buf.store_from(self.tensor, non_blocking=True)
            self._d2h_event.record()
        self._d2h_issued = True
        self._state = SwapState.D2H_IN_PROGRESS
        return MemDelta(hbm=0, ddr=self._bytes)

    def swap_in(self) -> MemDelta:
        if self._state in (SwapState.HBM_ONLY, SwapState.REPLICATED, SwapState.H2D_IN_PROGRESS):
            return MemDelta(0, 0)
        if self._state == SwapState.D2H_IN_PROGRESS:
            raise RuntimeError(f"Handle {self.id}: cannot load while offload in progress")
        self._ensure_state(SwapState.DDR_ONLY)

        if self._cpu_buf is None:
            raise RuntimeError(f"Handle {self.id}: no cpu buffer to load from")

        # Land in the put stream's pool: swap-path device allocations stay in
        # the main pool, and the H2D's wait_stream below covers the landing
        # block's past-life readers on it (readers on other streams are the
        # tenant's duty to order before it drops the reference). The transfer
        # is always a 1D contiguous DMA — a non-contiguous pinned source would
        # force torch_npu H2D through slow host-side materialization.
        with torch.accelerator.stream(self._put_stream):
            landing = self._cpu_buf.prepare_landing()
        with torch.accelerator.stream(self._h2d_stream):
            self._h2d_stream.wait_stream(self._put_stream)
            if self._dual_stream and self._d2h_issued:
                self._h2d_stream.wait_event(self._d2h_event)
            self.tensor = self._cpu_buf.load_into(landing, non_blocking=True)
            self._h2d_event.record()
        self._h2d_issued = True
        self._state = SwapState.H2D_IN_PROGRESS
        return MemDelta(hbm=self._bytes, ddr=0)

    def clear(self) -> MemDelta:
        delta = self.wait()
        if self._state == SwapState.HBM_ONLY:
            # HBM_ONLY means never offloaded, so no buffer was ever registered
            # (re-offload after a load goes through REPLICATED's zero-cost path).
            self.tensor = None
            self._state = SwapState.EMPTY
            return delta + MemDelta(hbm=-self._bytes, ddr=0)
        if self._state == SwapState.REPLICATED:
            self._free_cpu_buf()
            self.tensor = None
            self._state = SwapState.EMPTY
            return delta + MemDelta(hbm=-self._bytes, ddr=-self._bytes)
        if self._state == SwapState.DDR_ONLY:
            self._free_cpu_buf()
            self.tensor = None
            self._state = SwapState.EMPTY
            return delta + MemDelta(hbm=0, ddr=-self._bytes)
        # Only EMPTY reaches here (repeated clear is idempotent): wait() above
        # has already settled any in-flight state into the branches above.
        self.tensor = None
        self._free_cpu_buf()
        self._state = SwapState.EMPTY
        return delta

    def consume(self) -> tuple:
        if self._state == SwapState.D2H_IN_PROGRESS:
            tensor = self.tensor
            delta = self.clear()
            return tensor, delta
        delta = self._wait_h2d()
        if self._state == SwapState.DDR_ONLY:
            delta = delta + self.swap_in()
            delta = delta + self._wait_h2d()
        if self._state not in (SwapState.HBM_ONLY, SwapState.REPLICATED):
            raise RuntimeError(f"Handle {self.id}: unexpected state {self._state.name} in consume")
        tensor = self.tensor
        delta = delta + self.clear()
        return tensor, delta


class SwapCache:
    """SwapHandle collection management: FIFO/Belady eviction + capacity
    limits (hard capacity + soft limit) + hit/prefetch statistics

    Token model: the SwapHandle returned by put is the sole token for pop
    (and the cache key, so tenants need no numbering space).
    handle.id is a process-wide monotonic number, for logging/debugging only.

    Capacity semantics (capacity_bytes):
    - None: no capacity management (pass-through, nothing is evicted);
      soft_limit_bytes is ignored.
    - 0: every put evicts immediately and the hard limit waits in-flight D2H
      on the compute stream (device-side ordering): eviction is deterministic,
      no overlap with compute.
    - >0: normal asynchronous capacity management: evict proactively once the
      projected HBM (after in-flight D2H completes) exceeds soft_limit_bytes
      (the eviction line on the projected reading), and never let
      the physical HBM accounting exceed capacity_bytes.
    All copies run on the swap stream(s); the compute stream is ordered after
    them via events (no host block). Requires a usable device (fast fail).

    enable_prefetch defaults to True (callers may override).

    cpu_arena: required (keyword-only), composed and owned by the caller;
    SlabArena.on_demand() gives exact per-tensor
    allocation with immediate physical release. Empty pooled slabs
    deliberately keep their physical memory across clear(); call
    release_empty_slabs() explicitly to return it to the torch host cache
    (not directly to the OS).
    """

    def __init__(
        self,
        capacity_bytes=None,
        soft_limit_bytes=0,
        h2d_stream=None,
        d2h_stream=None,
        *,
        cpu_arena: SlabArena,
        eviction_policy="fifo",
        enable_prefetch: bool = True,
    ):
        if torch.accelerator.current_accelerator() is None:
            raise RuntimeError("SwapCache requires a usable device (CUDA/NPU); swap is not available on CPU-only hosts")
        # capacity_bytes: None (or negative, defensive normalization) disables
        # capacity management (pass-through); 0 means "zero capacity" (evict
        # everything, compute stream waits); >0 is the async managed mode.
        self.capacity_bytes = None if capacity_bytes is None or capacity_bytes < 0 else max(0, capacity_bytes)
        # soft_limit_bytes: the eviction line on the projected reading (HBM
        # that will remain once in-flight D2H copies complete). 0 = evict
        # right after put, maximizing async D2H overlap.
        self.soft_limit_bytes = max(0, soft_limit_bytes)
        # Single shared stream by default (half-duplex-safe); two different
        # streams enable bidirectional concurrency on full-duplex hardware.
        if h2d_stream is None and d2h_stream is None:
            h2d_stream = d2h_stream = torch.accelerator.Stream()
        elif h2d_stream is None:
            h2d_stream = d2h_stream
        elif d2h_stream is None:
            d2h_stream = h2d_stream
        self._h2d_stream = h2d_stream
        self._d2h_stream = d2h_stream
        self._cpu_arena = cpu_arena
        self._eviction_policy = eviction_policy
        self.handles: dict = {}  # insertion-ordered set of live handles (the key IS the entry)
        self.hbm_bytes = 0
        self.ddr_bytes = 0
        # In-flight D2H copies as an insertion-ordered set (dict keys only):
        # exact membership for O(k) _in_flight_bytes queries, and the head is
        # the earliest-issued copy (same-stream copies complete in issue
        # order, so waiting the head frees HBM fastest).
        self._d2h_in_flight: dict = {}
        self._iter_stats = {
            'hbm_hit': 0,
            'ddr_load': 0,
            'prefetch_issued': 0,
            'prefetch_skip_unknown': 0,
            'prefetch_skip_capacity': 0,
            'evict_free_cnt': 0,
            'peak_hbm_bytes': 0,
            'peak_ddr_bytes': 0,
            'put_cnt': 0,
            'pop_cnt': 0,
        }
        self.stats = {k: 0 for k in self._iter_stats}
        # Transition graph: learns the pop order in the first iteration and
        # chain-prefetches from the second on, indexed by put-position so
        # matching survives handle changes across iterations.
        self._put_order: List[SwapHandle] = []
        self._handle_to_put_idx: dict = {}
        self._pop_put_indices: List[int] = []
        self._transitions: List[int] = []
        self._pop_step_for_put_idx: dict = {}  # put position -> pop step, for farthest-future-use eviction
        self._pop_count: int = 0
        self._is_first_iter: bool = True
        self.enable_prefetch: bool = enable_prefetch

    def _apply(self, delta: MemDelta) -> None:
        self.hbm_bytes += delta.hbm
        self.ddr_bytes += delta.ddr
        self._iter_stats['peak_hbm_bytes'] = max(self._iter_stats['peak_hbm_bytes'], self.hbm_bytes)
        self._iter_stats['peak_ddr_bytes'] = max(self._iter_stats['peak_ddr_bytes'], self.ddr_bytes)

    def _track_in_flight(self, handle: SwapHandle) -> None:
        """Register handle iff its eviction actually issued an async D2H
        (zero-cost REPLICATED evictions never enter).
        """
        if handle.state() == SwapState.D2H_IN_PROGRESS:
            self._d2h_in_flight[handle] = None

    def _in_flight_bytes(self) -> int:
        return sum(handle.bytes() for handle in self._d2h_in_flight)

    def _projected_hbm_bytes(self) -> int:
        """Bytes that will remain in HBM once in-flight D2H copies complete."""
        return self.hbm_bytes - self._in_flight_bytes()

    def _wait_oldest_d2h(self) -> None:
        """Wait for the earliest-issued in-flight D2H to complete, freeing HBM fastest."""
        if not self._d2h_in_flight:
            return
        handle = next(iter(self._d2h_in_flight))
        self._apply(handle.wait())
        self._d2h_in_flight.pop(handle, None)

    def _evict_oldest(self) -> bool:
        # belady (graph learned): evict the tensor used farthest in the future
        # (unknown future pop position = "never used again"). Prefetched
        # tensors are not protected: zero-cost REPLICATED eviction needs none.
        # Otherwise FIFO (earliest stored).
        if self._eviction_policy == "belady" and not self._is_first_iter and self._pop_step_for_put_idx:
            candidate = None
            farthest = -1.0
            for handle in self.handles:
                if not handle.can_offload():
                    continue
                put_idx = self._handle_to_put_idx.get(handle, -1)
                step = self._pop_step_for_put_idx.get(put_idx, -1)
                key = float('inf') if step < 0 else float(step)
                if key > farthest:
                    farthest = key
                    candidate = handle
            if candidate is not None:
                self._evict(candidate)
                return True

        # Fallback: FIFO (dict insertion order = earliest stored)
        for handle in self.handles:
            if handle.can_offload():
                self._evict(handle)
                return True
        return False

    def _evict(self, handle: SwapHandle) -> None:
        # REPLICATED handles evict copy-free (the slot image is still valid);
        # count them to make prefetch/evict ping-pong visible.
        if handle.state() is SwapState.REPLICATED:
            self._iter_stats['evict_free_cnt'] += 1
        self._apply(handle.swap_out())
        self._track_in_flight(handle)

    def _trim(self) -> None:
        # Two non-interfering phases: evict to the soft limit on the projected
        # reading, then wait in-flight D2H for the hard limit on the physical
        # reading (waiting never worsens the projection).
        if self.capacity_bytes is None:
            return
        self._evict_to_soft_limit()
        self._wait_for_hard_limit()

    def _evict_to_soft_limit(self) -> None:
        """Soft limit, projected reading: evict proactively until the bytes
        that will remain once in-flight D2H copies complete fit the soft
        limit. With soft_limit_bytes=0 this evicts right after put,
        maximizing async D2H overlap.
        """
        while self._projected_hbm_bytes() > self.soft_limit_bytes:
            if self._evict_oldest():
                continue
            # Nothing evictable; wait for in-flight copies to lower the projection.
            if self._d2h_in_flight:
                self._wait_oldest_d2h()
                continue
            break

    def _wait_for_hard_limit(self) -> None:
        """Hard limit, physical reading: in-flight copies are accounted on
        both sides, so physical HBM can exceed capacity while the projection
        is fine; wait them out (evict first if nothing is in flight).
        """
        while self.hbm_bytes > self.capacity_bytes:
            if self._d2h_in_flight:
                self._wait_oldest_d2h()
                continue
            # Nothing in flight; only option is evict, then wait the copy.
            if not self._evict_oldest():
                break
            if self._d2h_in_flight:
                self._wait_oldest_d2h()

    def put(self, tensor: torch.Tensor) -> SwapHandle:
        # Returns the handle (the sole pop token) with
        # hbm_bytes <= capacity_bytes guaranteed; the soft limit decides how
        # early proactive eviction starts.
        handle = SwapHandle(tensor, h2d_stream=self._h2d_stream, d2h_stream=self._d2h_stream, cpu_arena=self._cpu_arena)
        self.handles[handle] = None
        self._put_order.append(handle)
        self._handle_to_put_idx[handle] = len(self._put_order) - 1
        self._apply(MemDelta(hbm=handle.hbm_bytes, ddr=handle.ddr_bytes))
        self._trim()
        self._iter_stats['put_cnt'] += 1
        return handle

    def pop(self, handle: SwapHandle) -> Optional[torch.Tensor]:
        if handle not in self.handles:
            return None
        del self.handles[handle]
        if handle.is_available_in_hbm():
            self._iter_stats['hbm_hit'] += 1
        else:
            self._iter_stats['ddr_load'] += 1
        tensor, delta = handle.consume()
        self._apply(delta)
        self._d2h_in_flight.pop(handle, None)  # consume may have waited out an in-flight D2H

        put_idx = self._handle_to_put_idx.get(handle, -1)
        self._pop_put_indices.append(put_idx)

        if self.enable_prefetch and not self._is_first_iter:
            self._prefetch_chain()
        self._pop_count += 1
        self._iter_stats['pop_cnt'] += 1

        # An emptied handle set does NOT trigger a spontaneous clear: the
        # cache cannot infer iteration boundaries from its own state,
        # so the upper layer delimits iterations via end_iteration()/clear().

        return tensor

    def _prefetch_chain(self) -> None:
        """Chain prefetch: walk the transition graph from the current pop
        position, pulling future-pop tensors back to HBM. The walk skips over
        handles that already hold a usable HBM copy (in HBM, or D2H/H2D in
        flight) and stops only at a chain break: an unknown handle (stale
        graph) or a handle that does not fit (budget exhausted).
        """
        step = self._pop_count
        while step + 1 < len(self._transitions):
            next_put_idx = self._transitions[step + 1]
            if not (0 <= next_put_idx < len(self._put_order)):
                break
            handle = self._put_order[next_put_idx]
            if handle not in self.handles:
                # Stale graph entry (already popped or cleaned up): stop.
                self._iter_stats['prefetch_skip_unknown'] += 1
                break
            if handle.is_available_in_hbm():
                # Nothing to issue for this one; keep walking.
                step += 1
                continue
            # Only DDR_ONLY can reach here (every other state is usable above).
            # capacity>=0 is a hard cap: 0 means zero budget, so every
            # prefetch is refused (demand pop still swaps in via consume).
            if self.capacity_bytes is not None and handle.bytes() > self.capacity_bytes - self.hbm_bytes:
                self._iter_stats['prefetch_skip_capacity'] += 1
                break
            self._apply(handle.swap_in())
            self._iter_stats['prefetch_issued'] += 1
            step += 1

    def end_iteration(self) -> None:
        """Iteration boundary: learn the transition graph + archive stats +
        reset access history. Resource cleanup lives in clear().
        """
        if self._is_first_iter and len(self._pop_put_indices) >= 2:
            # transitions[i] = put position of the i-th pop
            self._transitions = list(self._pop_put_indices)
            self._is_first_iter = False
            self._pop_step_for_put_idx = {}
            for step, put_idx in enumerate(self._transitions):
                if put_idx >= 0 and put_idx not in self._pop_step_for_put_idx:
                    self._pop_step_for_put_idx[put_idx] = step
        self._pop_put_indices.clear()
        self._put_order.clear()
        self._handle_to_put_idx.clear()
        self._pop_count = 0
        # stats archival: _iter_stats -> stats
        for k in self.stats:
            self.stats[k] = self._iter_stats[k]
        for k in self._iter_stats:
            self._iter_stats[k] = 0

    def clear(self) -> None:
        """end_iteration() + release all handles; keeps _transitions for reuse."""
        self.end_iteration()
        for handle in self.handles:
            handle.clear()
        self.handles.clear()
        self._d2h_in_flight.clear()
        self.hbm_bytes = 0
        self.ddr_bytes = 0

    def release_empty_slabs(self) -> None:
        """Physically release the pinned memory of all empty slabs back to the
        torch host cache. GB-scale pinned re-allocation has noticeable host
        cost, so callers should not invoke it per iteration (phase switches only).
        """
        self._cpu_arena.release_empty_slabs()

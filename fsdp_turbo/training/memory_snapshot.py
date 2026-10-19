# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Config-driven device memory snapshots for the trainer.

Collection is controlled by the ``run.memory_snapshot`` YAML section:

* ``enable`` -- master switch. When true, allocator history recording starts
  at ``start_step``; at ``end_step`` one snapshot is written on the selected
  ranks (``dump_ranks``) and recording stops. Without ``end_step`` the
  tracker falls back to writing one snapshot per step from ``start_step``.
  Steps are 0-based micro steps (one per dataloader batch).
* ``mem_info`` -- print one memory line per step (allocated / per-step peak /
  run peak / reserved, in GiB) and reset the peak stats each step. Works
  independently of ``enable``, so it can be used for stats-only runs.
* ``save_path`` -- snapshot directory; defaults to
  ``<checkpoint.output_dir>/memory_snapshot``. Relative paths resolve against
  the process working directory. The dump uses the NPU memory
  interfaces directly:
  ``torch.npu.memory._record_memory_history`` (``stacks=``/``max_entries=``,
  stopped with ``enabled=None``) and ``torch.npu.memory._dump_snapshot(path)``,
  which writes the pickle itself. CUDA backends use their
  ``torch.cuda.memory`` equivalents; a backend that cannot serve the
  requested collection (e.g. a CPU-only run) fails fast with a
  ``RuntimeError`` when training starts.
* ``dump_ranks`` -- global ranks that record and dump (default: all ranks).
* ``stacks``/``max_entries`` -- the recording scope and the allocator history
  entry cap (``null`` means unbounded, i.e. ``sys.maxsize``).

The OOM trap stays environment-variable controlled (it is deliberately not
part of the YAML schema) and is the only path that reacts to failures: the
config-driven collection covers normal runs only, so an exception (including
an OOM) during an ``enable: true`` run neither dumps a snapshot nor relocates
native allocator files by itself.

``OOM_SNAPSHOT_ENABLE`` arms the dump machinery silently: history recording
starts with the run (no per-step logging or dumps), and a single snapshot is
written only when a step raises ``torch.OutOfMemoryError``. It composes with
the YAML collection settings -- it shares the configured window, ranks,
``stacks`` and ``max_entries``.

While history recording is on, the caching allocator itself dumps
``oom_snapshot_<pid>_<timestamp>.pickle`` the moment an OOM is raised -- before
any Python handler runs. ``OOM_SNAPSHOT_PATH`` names the directory newer
torch_npu runtimes write to directly; the tracker relocates and renames the
file with a ``rank<r>_native_`` prefix there (falling back to the snapshot
directory) so the final layout is identical no matter which runtime wrote it.
"""

import logging
import os
import pickle
import shutil
import sys
import time

import torch
import torch.distributed as dist

from fsdp_turbo.fsdp_turbo_config import MemorySnapshotConfig
from fsdp_turbo.utils.log import log_warning_once

logger = logging.getLogger(__name__)

OOM_SNAPSHOT_ENV = "OOM_SNAPSHOT_ENABLE"
# torch_npu natively honors this for its allocator-written OOM pickles; we
# mirror it as the relocation target for older runtimes that ignore it.
NATIVE_OOM_DIR_ENV = "OOM_SNAPSHOT_PATH"

# torch.OutOfMemoryError exists since torch 2.4; older stacks raise RuntimeError.
OOMError = getattr(torch, "OutOfMemoryError", RuntimeError)

# The stats path needs all four; anything less disables the tracker.
_REQUIRED_MEMORY_APIS = ("memory_allocated", "memory_reserved", "max_memory_allocated", "reset_peak_memory_stats")


def oom_snapshot_enabled() -> bool:
    """Resolve ``OOM_SNAPSHOT_ENABLE`` to a boolean (default false)."""
    raw = os.environ.get(OOM_SNAPSHOT_ENV, "").strip().lower()
    if raw in ("", "0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "on", "yes"):
        return True
    raise ValueError(f"{OOM_SNAPSHOT_ENV} must be a boolean (0/1, true/false, on/off); got {raw!r}.")


def _current_rank() -> int:
    """Global rank from the process group, falling back to the RANK env var."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _device_memory_module():
    """Return the torch module exposing device memory APIs, or ``None``.

    The allocator snapshot APIs (``memory._record_memory_history`` /
    ``memory._dump_snapshot``) are backend-private extensions that
    ``torch.accelerator``'s memory surface does not carry, and its
    device-generic statistics assert on ``torch_npu`` (see
    ``fsdp_turbo/memory/swap_optimizer``), so the device module stays the
    primary handle here and ``torch.accelerator`` is the last resort.
    """
    try:
        import torch_npu  # noqa: F401

        return torch.npu
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.cuda
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "memory_allocated"):
        return accelerator
    return None


def _snapshot_api_module():
    """The ``torch.npu.memory`` / ``torch.cuda.memory`` submodule, if any."""
    device_mod = _device_memory_module()
    return getattr(device_mod, "memory", None) if device_mod is not None else None


def _device_synchronize():
    """Best-effort sync so comm/workspaces materialize before reading stats."""
    try:
        torch.accelerator.synchronize()
    except Exception:  # noqa: BLE001  # nosec B110 - accuracy aid only, never fatal
        pass


class MemorySnapshotTracker:
    """Per-step device memory stats with optional allocator snapshot dumps.

    ``enable`` turns on snapshot dumps. With ``end_step`` set, allocator
    history is recorded from ``start_step`` and exactly one snapshot is
    written at ``end_step`` before recording stops;
    without ``end_step`` one snapshot is written per step from ``start_step``.
    ``mem_info`` independently prints a per-step memory line and resets the
    peak stats, ``oom_snapshot`` arms the silent OOM trap described in the
    module docstring.

    ``start_step``/``end_step`` are 0-based micro steps and the window is
    inclusive on both ends. The step counter still advances outside the
    window so dump filenames carry the true step number.
    """

    def __init__(
        self,
        enable: bool,
        output_dir: str,
        mem_info: bool = False,
        oom_snapshot: bool = False,
        start_step: int = 0,
        end_step: int | None = None,
        dump_dir: str | None = None,
        ranks: set[int] | None = None,
        stacks: str = "all",
        max_entries: int | None = None,
    ):
        self.enable = bool(enable)
        self.mem_info = bool(mem_info)
        self.oom_snapshot = bool(oom_snapshot)
        self.output_dir = output_dir
        self.start_step = start_step
        self.end_step = end_step
        self.ranks = ranks
        self.stacks = stacks
        self.max_entries = max_entries
        # With an explicit end_step the collection is one bounded capture: a
        # single snapshot at end_step; without one it degrades to per-step
        # dumps from start_step.
        self.single_dump_at_end = self.enable and end_step is not None
        self.micro_step = 0
        self.peak_allocated_bytes = 0
        self._history_state = "idle"  # idle -> recording -> stopped
        base_dir = dump_dir or os.path.join(output_dir, "memory_snapshot")
        self.dump_dir = base_dir if (self.enable or self.oom_snapshot) and self._rank_selected() else None
        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)

    def _rank_selected(self) -> bool:
        """Whether the current global rank collects (None means all ranks)."""
        return self.ranks is None or _current_rank() in self.ranks

    def _collects_at(self, step: int) -> bool:
        """Whether ``step`` (0-based micro step) is inside the collection window."""
        if step < self.start_step:
            return False
        if self.end_step is not None and step > self.end_step:
            return False
        return True

    def _wants_history(self) -> bool:
        return self.enable or self.oom_snapshot

    def on_train_begin(self):
        if not self._wants_history() and not self.mem_info:
            return
        if not self._rank_selected():
            return
        self._require_memory_apis()
        if self._wants_history() and self.start_step == 0:
            self._start_memory_history()
            self._history_state = "recording"

    def _require_memory_apis(self):
        """Fail fast when the requested collection is unsupported by the backend.

        Snapshot collection is an explicit opt-in, so a backend that cannot
        serve it stops the run with a clear error instead of silently
        degrading to no collection.
        """
        _unset_hint = f"set run.memory_snapshot.enable=false or unset {OOM_SNAPSHOT_ENV} to run without snapshots"
        device_mod = _device_memory_module()
        if device_mod is None:
            raise RuntimeError(
                f"memory snapshot collection is enabled but no accelerator memory API is available on rank "
                f"{_current_rank()}; {_unset_hint}."
            )
        missing = [api for api in _REQUIRED_MEMORY_APIS if not hasattr(device_mod, api)]
        if missing:
            raise RuntimeError(
                f"memory snapshot collection is enabled but the device memory module lacks {missing} on rank "
                f"{_current_rank()}; {_unset_hint}."
            )
        if self._wants_history():
            memory_mod = getattr(device_mod, "memory", None)
            has_dump_api = callable(getattr(memory_mod, "_dump_snapshot", None)) or callable(
                getattr(device_mod, "memory_snapshot", None)
            )
            if not has_dump_api:
                raise RuntimeError(
                    f"memory snapshot collection is enabled but the device memory module supports neither "
                    f"_dump_snapshot nor memory_snapshot() on rank {_current_rank()}; {_unset_hint}."
                )

    def on_step_start(self):
        if not self._rank_selected():
            return
        if self._wants_history():
            # self.micro_step is the 0-based number of the upcoming step.
            if self._history_state == "idle" and self.micro_step >= self.start_step:
                self._start_memory_history()
                self._history_state = "recording"
            elif (
                self._history_state == "recording"
                and not self.single_dump_at_end
                and self.end_step is not None
                and self.micro_step > self.end_step
            ):
                self._stop_memory_history()
                self._history_state = "stopped"
        if self.mem_info and self._collects_at(self.micro_step):
            _device_memory_module().reset_peak_memory_stats()

    def on_step_end(self, global_step: int):
        step_number = self.micro_step
        self.micro_step += 1
        if not self._rank_selected():
            return
        if self.mem_info and self._collects_at(step_number):
            device_mod = _device_memory_module()
            _device_synchronize()
            allocated = device_mod.memory_allocated()
            reserved = device_mod.memory_reserved()
            step_peak = device_mod.max_memory_allocated()
            self.peak_allocated_bytes = max(self.peak_allocated_bytes, step_peak)
            logger.info(
                "[memory] micro=%d global=%d alloc=%.2fGiB peak_step=%.2fGiB peak_run=%.2fGiB reserved=%.2fGiB",
                self.micro_step,
                global_step,
                allocated / 2**30,
                step_peak / 2**30,
                self.peak_allocated_bytes / 2**30,
                reserved / 2**30,
            )
        if not self.enable or not self._collects_at(step_number):
            return
        if self.single_dump_at_end:
            if step_number == self.end_step:
                self._dump(global_step, tag="step")
                self._stop_memory_history()
                self._history_state = "stopped"
            return
        self._dump(global_step, tag="step")

    def on_error(self, global_step: int, exc: BaseException):
        """Best-effort OOM snapshot after a failed training step.

        Only the environment-variable OOM trap acts here: it dumps when the
        step raises ``torch.OutOfMemoryError``. The config-driven collection
        covers normal runs only -- with just ``memory_snapshot.enable`` set,
        an exception (including an OOM) neither dumps nor stops the history,
        and the native allocator files are left where the runtime wrote them.
        The trap is gated by the collection window -- the failing step number
        is ``self.micro_step`` because the step did not complete.
        """
        should_dump = (self.oom_snapshot and isinstance(exc, OOMError)) and self._rank_selected()
        if not should_dump or not self._collects_at(self.micro_step):
            return
        try:
            self._relocate_native_oom_snapshots()
            self._dump(global_step, tag="error")
            self._stop_memory_history()
            self._history_state = "stopped"
        except Exception:  # noqa: BLE001 - diagnostics must not mask the training error
            logger.warning("memory snapshot dump on error failed", exc_info=True)

    def _relocate_native_oom_snapshots(self) -> list[str]:
        """Normalize this process's allocator-written OOM snapshots.

        While memory history recording is on, the caching allocator itself
        dumps ``oom_snapshot_<pid>_<timestamp>.pickle`` the moment an OOM is
        raised -- before any Python handler runs. Newer torch_npu runtimes
        honor ``OOM_SNAPSHOT_PATH`` and write there directly (the variable is a
        documented runtime interface) but still with the hardcoded pid-based
        name; older runtimes write into the current working directory. After
        the error, move CWD files into
        the target directory and rename both kinds with a ``rank<r>_native_``
        prefix so the final layout is identical no matter which runtime
        wrote them. Only this process's pid is touched: concurrent ranks
        share the directories and each handles its own file.
        """
        target_dir = os.environ.get(NATIVE_OOM_DIR_ENV) or self.dump_dir
        if target_dir is None:
            return []
        os.makedirs(target_dir, exist_ok=True)
        pid_prefix = f"oom_snapshot_{os.getpid()}_"
        rank = _current_rank()
        finalized = []

        # Old runtimes: sweep this process's pickles out of the CWD.
        for name in sorted(os.listdir(".")):
            if not name.startswith(pid_prefix) or not name.endswith(".pickle"):
                continue
            destination = os.path.join(target_dir, f"rank{rank}_native_{name}")
            try:
                shutil.move(name, destination)
            except OSError:
                continue
            logger.info("[memory] relocated native OOM snapshot %s -> %s", name, destination)
            finalized.append(destination)

        # New runtimes: the pickle is already in the target dir under its
        # hardcoded pid-based name; rename it in place for the rank prefix.
        for name in sorted(os.listdir(target_dir)):
            if not name.startswith(pid_prefix) or not name.endswith(".pickle"):
                continue
            destination = os.path.join(target_dir, f"rank{rank}_native_{name}")
            try:
                os.rename(os.path.join(target_dir, name), destination)
            except OSError:
                continue
            logger.info("[memory] renamed native OOM snapshot %s -> %s", name, destination)
            finalized.append(destination)
        return finalized

    def _start_memory_history(self):
        snap_mod = _snapshot_api_module()
        record = getattr(snap_mod, "_record_memory_history", None)
        if record is None:
            log_warning_once(
                logger,
                "device backend does not support _record_memory_history; "
                "dumping plain snapshots without allocation stacks",
            )
            return
        # Default: sys.maxsize (unbounded); bound it via
        # memory_snapshot.max_entries or the step window when needed.
        max_entries = sys.maxsize if self.max_entries is None else int(self.max_entries)
        # torch_npu names the scope kwarg `stacks`, CUDA names it `context`.
        for kwargs in (
            {"stacks": self.stacks, "max_entries": max_entries},
            {"context": self.stacks, "max_entries": max_entries},
            {"max_entries": max_entries},
        ):
            try:
                record(**kwargs)
                return
            except TypeError:
                continue

    def _stop_memory_history(self):
        record = getattr(_snapshot_api_module(), "_record_memory_history", None)
        if record is None:
            return
        for kwargs in ({"enabled": None}, {"enabled": False}):
            try:
                record(**kwargs)
                return
            except TypeError:
                continue

    def _dump(self, global_step: int, tag: str):
        rank = _current_rank()
        # The time suffix keeps dumps from different runs apart; the
        # micro/global fields keep multiple dumps within the same minute
        # distinct.
        str_time = time.strftime("%Y-%m-%d-%H-%M")
        path = os.path.join(
            self.dump_dir,
            f"rank{rank}_{tag}_micro{self.micro_step}_global{global_step}_{str_time}.pkl",
        )
        snap_mod = _snapshot_api_module()
        dump = getattr(snap_mod, "_dump_snapshot", None)
        if dump is not None:
            # NPU/CUDA native: writes the pickle file itself.
            dump(path)
            # Some torch_npu builds ignore the requested path and write
            # device_{id}_memusage artifacts (pickle plus case-varying
            # companions such as device_{id}_memUsageINFO*.csv) into the
            # current working directory; sweep them into the dump dir so it
            # stays authoritative.
            self._relocate_native_dump(path)
            if not os.path.exists(path):
                log_warning_once(
                    logger,
                    "_dump_snapshot produced no file at the requested path; "
                    f"snapshot for this dump may be missing (expected {path})",
                )
        else:
            # The startup capability check guarantees that one of the two dump
            # APIs exists, so the fallback is taken unconditionally. pylint
            # const-folds the getattr against the known device modules.
            snapshot = getattr(_device_memory_module(), "memory_snapshot", None)
            with open(path, "wb") as f:
                pickle.dump(snapshot(), f)  # pylint: disable=not-callable
        logger.info("[memory] dumped allocator snapshot to %s", path)

    def _relocate_native_dump(self, path: str) -> str | None:
        """Move torch_npu native dump artifacts from CWD into the dump dir.

        Matching is case-insensitive on the ``device_`` prefix and the
        ``memusage`` marker because the artifacts' spelling varies across
        builds (``device_0_memusage.pickle``, ``device_0_memUsageINFO*.csv``).
        The snapshot pickle is moved to ``path``; companions such as the
        memUsage CSV keep their own names inside the dump directory. This
        process's device id (LOCAL_RANK first) is preferred so concurrent
        ranks sharing a working directory never grab each other's files; the
        any-device fallback only runs when no own-device artifact exists.
        """
        device_id = os.environ.get("LOCAL_RANK") or os.environ.get("ASCEND_DEVICE_ID")

        def _candidates(prefixes):
            names = [
                name
                for name in os.listdir(".")
                if "memusage" in name.lower() and name.lower().startswith(tuple(prefixes))
            ]
            return sorted(names, key=os.path.getmtime, reverse=True)

        own_prefixes = (f"device_{device_id}_",) if device_id is not None else ()
        candidates = _candidates(own_prefixes) or _candidates(("device_",))
        snapshot_placed = os.path.exists(path)
        relocated = None
        for name in candidates:
            if snapshot_placed or relocated is not None or name.lower().endswith(".csv"):
                # Companion stats (e.g. memUsageINFO*.csv) keep their names.
                destination = os.path.join(self.dump_dir, name)
            else:
                destination = path
            try:
                shutil.move(name, destination)
            except OSError:
                continue
            logger.info("[memory] relocated native dump %s -> %s", name, destination)
            if destination == path:
                relocated = name
        return relocated


def build_memory_snapshot_tracker(memory_snapshot: MemorySnapshotConfig, output_dir: str) -> MemorySnapshotTracker:
    """Build the tracker from the ``run.memory_snapshot`` YAML section.

    The OOM trap and its relocation target stay environment-variable
    controlled (``OOM_SNAPSHOT_ENABLE`` / ``OOM_SNAPSHOT_PATH``); it shares
    the configured window, ranks, ``stacks`` and ``max_entries``.
    """
    dump_dir = None
    if memory_snapshot.save_path:
        dump_dir = os.path.expanduser(str(memory_snapshot.save_path).strip())
    ranks = None
    if memory_snapshot.dump_ranks is not None:
        ranks = {int(rank) for rank in memory_snapshot.dump_ranks}
    return MemorySnapshotTracker(
        enable=bool(memory_snapshot.enable),
        output_dir=output_dir,
        mem_info=bool(memory_snapshot.mem_info),
        oom_snapshot=oom_snapshot_enabled(),
        start_step=int(memory_snapshot.start_step),
        end_step=None if memory_snapshot.end_step is None else int(memory_snapshot.end_step),
        dump_dir=dump_dir,
        ranks=ranks,
        stacks=str(memory_snapshot.stacks),
        max_entries=memory_snapshot.max_entries,
    )

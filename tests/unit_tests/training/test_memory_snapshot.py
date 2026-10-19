# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import glob
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
import yaml

from fsdp_turbo.fsdp_turbo_config import MemorySnapshotConfig, load_config_from_yaml
from fsdp_turbo.training import memory_snapshot as ms

_DUMP_MARKER = b"npu-dump"


def _stub_device_module(allocated, reserved, peak, *, with_snapshot=True, with_fallback=False):
    """A device module exposing the tracker's stats and NPU-style dump APIs."""
    calls = {"reset": 0, "record": 0, "dump": 0, "snapshot": 0}
    record_calls = []

    def _record(**kwargs):
        record_calls.append(kwargs)
        calls["record"] += 1

    def _dump_snapshot(path):
        calls["dump"] += 1
        with open(path, "wb") as f:
            f.write(_DUMP_MARKER)

    stub = types.SimpleNamespace(
        memory_allocated=lambda: allocated,
        memory_reserved=lambda: reserved,
        max_memory_allocated=lambda: peak,
        reset_peak_memory_stats=lambda: calls.__setitem__("reset", calls["reset"] + 1),
    )
    if with_snapshot:
        stub.memory = types.SimpleNamespace(
            _record_memory_history=_record,
            _dump_snapshot=_dump_snapshot,
        )
    if with_fallback:
        stub.memory_snapshot = lambda: (calls.__setitem__("snapshot", calls["snapshot"] + 1) or [{"segment": 0}])
    return stub, calls, record_calls


class MemorySnapshotConfigTests(unittest.TestCase):
    """run.memory_snapshot parsing and validation."""

    def _load(self, memory_snapshot):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"run": {"memory_snapshot": memory_snapshot}}, f, sort_keys=False)
            path = f.name
        try:
            return load_config_from_yaml(path)
        finally:
            os.remove(path)

    def test_user_yaml_round_trip(self):
        config = self._load(
            {
                "enable": False,
                "start_step": 1,
                "end_step": 2,
                "save_path": "./memory_snapshot",
                "dump_ranks": [0],
                "stacks": "all",
                "max_entries": None,
                "mem_info": False,
            }
        )
        profile = config.run.memory_snapshot
        self.assertFalse(profile.enable)
        self.assertEqual(1, profile.start_step)
        self.assertEqual(2, profile.end_step)
        self.assertEqual("./memory_snapshot", profile.save_path)
        self.assertEqual([0], profile.dump_ranks)
        self.assertEqual("all", profile.stacks)
        self.assertIsNone(profile.max_entries)
        self.assertFalse(profile.mem_info)

    def test_defaults(self):
        profile = self._load({}).run.memory_snapshot
        self.assertFalse(profile.enable)
        self.assertEqual(0, profile.start_step)
        self.assertIsNone(profile.end_step)
        self.assertIsNone(profile.save_path)
        self.assertIsNone(profile.dump_ranks)
        self.assertEqual("all", profile.stacks)
        self.assertIsNone(profile.max_entries)
        self.assertFalse(profile.mem_info)

    def test_stacks_normalized_to_lowercase(self):
        profile = self._load({"stacks": "ALL"}).run.memory_snapshot
        self.assertEqual("all", profile.stacks)

    def test_invalid_values_raise(self):
        cases = (
            ({"start_step": -1}, "start_step"),
            ({"start_step": 3, "end_step": 2}, "end_step"),
            ({"stacks": "deep"}, "stacks"),
            ({"dump_ranks": [0, -1]}, "dump_ranks"),
            ({"dump_ranks": "0"}, "dump_ranks"),
            ({"max_entries": 0}, "max_entries"),
            ({"save_path": "   "}, "save_path"),
        )
        for memory_snapshot, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                self._load(memory_snapshot)


class BuildTrackerTests(unittest.TestCase):
    """build_memory_snapshot_tracker maps the YAML config onto the tracker."""

    def test_build_tracker_from_config(self):
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as save_dir:
            with mock.patch.dict(os.environ, {ms.OOM_SNAPSHOT_ENV: ""}, clear=False):
                config = MemorySnapshotConfig(
                    enable=True,
                    start_step=1,
                    end_step=2,
                    save_path=os.path.join(save_dir, "snaps"),
                    dump_ranks=[0],
                    stacks="python",
                    max_entries=100,
                    mem_info=True,
                )
                tracker = ms.build_memory_snapshot_tracker(config, out_dir)
            try:
                self.assertTrue(tracker.enable)
                self.assertTrue(tracker.mem_info)
                self.assertFalse(tracker.oom_snapshot)
                self.assertEqual(1, tracker.start_step)
                self.assertEqual(2, tracker.end_step)
                self.assertTrue(tracker.single_dump_at_end)
                self.assertEqual(os.path.join(save_dir, "snaps"), tracker.dump_dir)
                self.assertEqual({0}, tracker.ranks)
                self.assertEqual("python", tracker.stacks)
                self.assertEqual(100, tracker.max_entries)
            finally:
                del tracker

    def test_build_tracker_defaults_and_cwd_relative_save_path(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with mock.patch.dict(os.environ, {ms.OOM_SNAPSHOT_ENV: ""}, clear=False):
                tracker = ms.build_memory_snapshot_tracker(MemorySnapshotConfig(), out_dir)
            self.assertFalse(tracker.enable)
            self.assertFalse(tracker.mem_info)
            self.assertFalse(tracker.oom_snapshot)
            self.assertIsNone(tracker.dump_dir)  # fully disabled: no directory is created
            self.assertFalse(os.path.exists(os.path.join(out_dir, "memory_snapshot")))
            self.assertIsNone(tracker.ranks)
            self.assertIsNone(tracker.max_entries)
            self.assertFalse(tracker.single_dump_at_end)

    def test_oom_trap_env_is_read_at_build_time(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with mock.patch.dict(os.environ, {ms.OOM_SNAPSHOT_ENV: "1"}, clear=False):
                tracker = ms.build_memory_snapshot_tracker(MemorySnapshotConfig(), out_dir)
            self.assertTrue(tracker.oom_snapshot)

    def test_configured_stacks_and_max_entries_reach_record(self):
        with tempfile.TemporaryDirectory() as out_dir:
            tracker = ms.MemorySnapshotTracker(enable=True, output_dir=out_dir, stacks="python", max_entries=100)
            stub, _, record_calls = _stub_device_module(0, 0, 0)
            with (
                mock.patch.object(ms, "_device_memory_module", return_value=stub),
                mock.patch.object(ms, "_device_synchronize"),
            ):
                tracker.on_train_begin()
            self.assertEqual([{"stacks": "python", "max_entries": 100}], record_calls)


class OomSnapshotEnableTests(unittest.TestCase):
    def test_boolean_aliases(self):
        cases = {
            None: False,
            "": False,
            "0": False,
            "false": False,
            "OFF": False,
            "no": False,
            "1": True,
            "true": True,
            "ON": True,
            "yes": True,
        }
        for raw, expected in cases.items():
            env = {} if raw is None else {ms.OOM_SNAPSHOT_ENV: raw}
            with mock.patch.dict(os.environ, env, clear=False):
                self.assertEqual(ms.oom_snapshot_enabled(), expected, msg=repr(raw))

    def test_invalid_value_raises(self):
        with mock.patch.dict(os.environ, {ms.OOM_SNAPSHOT_ENV: "maybe"}):
            with self.assertRaisesRegex(ValueError, "OOM_SNAPSHOT_ENABLE"):
                ms.oom_snapshot_enabled()


class SingleDumpSemanticsTests(unittest.TestCase):
    """With end_step set the collection is one bounded capture."""

    def test_single_dump_at_end_step(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with mock.patch.dict(os.environ, {"RANK": "0"}, clear=False):
                tracker = ms.MemorySnapshotTracker(enable=True, output_dir=out_dir, start_step=1, end_step=2)
                stub, calls, record_calls = _stub_device_module(0, 0, 0)
                with (
                    mock.patch.object(ms, "_device_memory_module", return_value=stub),
                    mock.patch.object(ms, "_device_synchronize"),
                ):
                    tracker.on_train_begin()
                    self.assertEqual("idle", tracker._history_state)  # deferred to the window
                    tracker.on_step_start()
                    tracker.on_step_end(global_step=0)  # step 0: nothing
                    self.assertEqual(0, calls["dump"])
                    self.assertEqual([], record_calls)
                    tracker.on_step_start()  # step 1: history arms
                    self.assertEqual([{"stacks": "all", "max_entries": sys.maxsize}], record_calls)
                    tracker.on_step_end(global_step=1)
                    self.assertEqual(0, calls["dump"])  # not the end step yet
                    tracker.on_step_start()  # step 2: still recording through the end step
                    tracker.on_step_end(global_step=2)  # the single dump
                    self.assertEqual(1, calls["dump"])
                    self.assertEqual({"enabled": None}, record_calls[-1])
                    tracker.on_step_start()
                    tracker.on_step_end(global_step=3)  # step 3: window closed
                    self.assertEqual(1, calls["dump"])
            dumped = glob.glob(os.path.join(out_dir, "memory_snapshot", "rank0_step_micro3_global2_*.pkl"))
            self.assertEqual(1, len(dumped))
            self.assertEqual(1, len(os.listdir(os.path.join(out_dir, "memory_snapshot"))))

    def test_error_without_oom_trap_does_not_dump(self):
        """The config-driven collection covers normal runs only."""
        with tempfile.TemporaryDirectory() as out_dir:
            tracker = ms.MemorySnapshotTracker(enable=True, output_dir=out_dir, start_step=1, end_step=5)
            stub, calls, record_calls = _stub_device_module(0, 0, 0)
            with (
                mock.patch.object(ms, "_device_memory_module", return_value=stub),
                mock.patch.object(ms, "_device_synchronize"),
            ):
                tracker.on_train_begin()
                tracker.on_step_start()
                tracker.on_error(global_step=0, exc=torch.OutOfMemoryError("oom"))  # step 0: outside the window
                self.assertEqual(0, calls["dump"])
                tracker.on_step_end(global_step=0)  # counter advances past step 0
                tracker.on_step_start()  # step 1: recording armed
                tracker.on_error(global_step=1, exc=RuntimeError("boom"))
                self.assertEqual(0, calls["dump"])  # no error snapshot without the trap
                self.assertEqual("recording", tracker._history_state)  # recording is not stopped
                tracker.on_error(global_step=1, exc=torch.OutOfMemoryError("oom"))
                self.assertEqual(0, calls["dump"])  # even an OOM does not dump without the trap
            # the directory is pre-created, but nothing may land in it
            self.assertEqual([], os.listdir(os.path.join(out_dir, "memory_snapshot")))

    def test_per_step_dumps_without_end_step(self):
        with tempfile.TemporaryDirectory() as out_dir:
            tracker = ms.MemorySnapshotTracker(enable=True, output_dir=out_dir, start_step=1)
            stub, calls, record_calls = _stub_device_module(0, 0, 0)
            with (
                mock.patch.object(ms, "_device_memory_module", return_value=stub),
                mock.patch.object(ms, "_device_synchronize"),
            ):
                tracker.on_train_begin()
                tracker.on_step_start()
                tracker.on_step_end(global_step=0)  # step 0: outside the window
                self.assertEqual(0, calls["dump"])
                for _ in range(3):  # steps 1, 2, 3: one dump each
                    tracker.on_step_start()
                    tracker.on_step_end(global_step=1)
                self.assertEqual(3, calls["dump"])
                self.assertEqual("recording", tracker._history_state)  # no end_step, never stops
            self.assertEqual(3, len(os.listdir(os.path.join(out_dir, "memory_snapshot"))))


class MemInfoTests(unittest.TestCase):
    """mem_info is an independent per-step memory line switch."""

    def test_mem_info_without_enable_prints_only(self):
        with tempfile.TemporaryDirectory() as out_dir:
            tracker = ms.MemorySnapshotTracker(enable=False, output_dir=out_dir, mem_info=True)
            stub, calls, record_calls = _stub_device_module(2**30, 2**30, 2**30)
            with (
                mock.patch.object(ms, "_device_memory_module", return_value=stub),
                mock.patch.object(ms, "_device_synchronize"),
            ):
                tracker.on_train_begin()
                self.assertEqual(0, calls["record"])
                tracker.on_step_start()
                with self.assertLogs(ms.logger, level="INFO") as logs:
                    tracker.on_step_end(global_step=0)
            self.assertEqual(1, calls["reset"])
            self.assertEqual(0, calls["dump"])
            self.assertEqual(1, len([line for line in logs.output if "[memory]" in line]))
            self.assertFalse(os.path.exists(os.path.join(out_dir, "memory_snapshot")))

    def test_mem_info_window_is_inclusive(self):
        with tempfile.TemporaryDirectory() as out_dir:
            tracker = ms.MemorySnapshotTracker(
                enable=False, output_dir=out_dir, mem_info=True, start_step=1, end_step=3
            )
            stub, calls, _ = _stub_device_module(2**30, 2**30, 2**30)
            with (
                mock.patch.object(ms, "_device_memory_module", return_value=stub),
                mock.patch.object(ms, "_device_synchronize"),
            ):
                tracker.on_train_begin()
                with self.assertNoLogs(ms.logger, level="INFO"):
                    tracker.on_step_start()
                    tracker.on_step_end(global_step=0)  # step 0: outside
                    self.assertEqual(0, calls["reset"])
                logged = []
                for _ in range(3):  # steps 1, 2 and 3 are inside the inclusive window
                    tracker.on_step_start()
                    with self.assertLogs(ms.logger, level="INFO") as logs:
                        tracker.on_step_end(global_step=1)
                    logged.extend(logs.output)
                with self.assertNoLogs(ms.logger, level="INFO"):
                    tracker.on_step_start()
                    tracker.on_step_end(global_step=4)  # step 4: outside again
        self.assertEqual(5, tracker.micro_step)  # counter always advances
        self.assertEqual(3, calls["reset"])  # peak reset only inside the window
        self.assertEqual(3, len([line for line in logged if "[memory]" in line]))


class MemorySnapshotDirTests(unittest.TestCase):
    def test_dump_dir_override_directs_dumps(self):
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as custom_dir:
            tracker = ms.MemorySnapshotTracker(
                enable=True, output_dir=out_dir, dump_dir=os.path.join(custom_dir, "oom_shots")
            )
            stub, calls, _ = _stub_device_module(0, 0, 0)
            with (
                mock.patch.object(ms, "_device_memory_module", return_value=stub),
                mock.patch.object(ms, "_device_synchronize"),
            ):
                tracker.on_train_begin()
                tracker.on_step_start()
                tracker.on_step_end(global_step=0)
            self.assertEqual(1, calls["dump"])
            dumped = os.listdir(os.path.join(custom_dir, "oom_shots"))
            self.assertEqual(1, len(dumped))
            self.assertTrue(dumped[0].startswith("rank0_step_micro1_global0_"))
            self.assertFalse(os.path.exists(os.path.join(out_dir, "memory_snapshot")))

    def test_native_dump_ignoring_path_is_relocated(self):
        stub, calls, _ = _stub_device_module(0, 0, 0)

        def _dump_to_cwd(path):
            # torch_npu quirk: ignore the requested path, write fixed names
            # (mixed-case pickle plus a memUsageINFO stats CSV) into CWD.
            device_id = os.environ.get("LOCAL_RANK", "0")
            with open(os.path.join(os.getcwd(), f"device_{device_id}_memUsage.pickle"), "wb") as f:
                f.write(_DUMP_MARKER)
            with open(os.path.join(os.getcwd(), f"device_{device_id}_memUsageINFO1.csv"), "w", encoding="utf-8") as f:
                f.write("usage stats")

        stub.memory._dump_snapshot = _dump_to_cwd
        cwd = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(cwd)
        try:
            # Another rank's dump in the shared working directory must not be picked up.
            decoy = os.path.join(cwd, "device_9_memusage.pickle")
            with open(decoy, "wb") as f:
                f.write(b"other-rank")
            with tempfile.TemporaryDirectory() as out_dir:
                with mock.patch.dict(os.environ, {"RANK": "3", "LOCAL_RANK": "3"}, clear=False):
                    tracker = ms.MemorySnapshotTracker(enable=True, output_dir=out_dir)
                    with (
                        mock.patch.object(ms, "_device_memory_module", return_value=stub),
                        mock.patch.object(ms, "_device_synchronize"),
                    ):
                        tracker.on_train_begin()
                        tracker.on_step_start()
                        with self.assertLogs(ms.logger, level="INFO"):
                            tracker.on_step_end(global_step=0)
                dump_dir = os.path.join(out_dir, "memory_snapshot")
                # The mixed-case pickle lands at our path; the CSV keeps its
                # own name beside it; CWD only retains the other rank's file.
                matches = glob.glob(os.path.join(dump_dir, "rank3_step_micro1_global0_*.pkl"))
                self.assertEqual(1, len(matches))
                with open(matches[0], "rb") as f:
                    self.assertEqual(_DUMP_MARKER, f.read())
                self.assertEqual(
                    ["device_3_memUsageINFO1.csv"], [n for n in os.listdir(dump_dir) if n.endswith(".csv")]
                )
            self.assertEqual(["device_9_memusage.pickle"], os.listdir(cwd))
        finally:
            os.chdir(old_cwd)
            for leftover in glob.glob(os.path.join(cwd, "device_*")):
                os.remove(leftover)
            os.rmdir(cwd)

    def test_companion_csv_swept_even_when_path_is_honored(self):
        stub, _, _ = _stub_device_module(0, 0, 0)

        def _dump_honoring_path_with_csv(path):
            with open(path, "wb") as f:
                f.write(_DUMP_MARKER)
            device_id = os.environ.get("LOCAL_RANK", "0")
            with open(os.path.join(os.getcwd(), f"device_{device_id}_memUsageINFO2.csv"), "w", encoding="utf-8") as f:
                f.write("usage stats")

        stub.memory._dump_snapshot = _dump_honoring_path_with_csv
        cwd = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(cwd)
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                with mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"}, clear=False):
                    tracker = ms.MemorySnapshotTracker(enable=True, output_dir=out_dir)
                    with (
                        mock.patch.object(ms, "_device_memory_module", return_value=stub),
                        mock.patch.object(ms, "_device_synchronize"),
                    ):
                        tracker.on_train_begin()
                        tracker.on_step_start()
                        tracker.on_step_end(global_step=0)
                csvs = [n for n in os.listdir(os.path.join(out_dir, "memory_snapshot")) if n.endswith(".csv")]
                self.assertEqual(["device_0_memUsageINFO2.csv"], csvs)
            self.assertEqual([], os.listdir(cwd))
        finally:
            os.chdir(old_cwd)
            for leftover in glob.glob(os.path.join(cwd, "device_*")):
                os.remove(leftover)
            os.rmdir(cwd)

    def test_native_dump_writing_nothing_warns(self):
        stub, _, _ = _stub_device_module(0, 0, 0)
        stub.memory._dump_snapshot = lambda path: None  # writes nothing anywhere
        with tempfile.TemporaryDirectory() as out_dir:
            tracker = ms.MemorySnapshotTracker(enable=True, output_dir=out_dir)
            with (
                mock.patch.object(ms, "_device_memory_module", return_value=stub),
                mock.patch.object(ms, "_device_synchronize"),
            ):
                tracker.on_train_begin()
                with self.assertLogs(ms.logger, level="WARNING") as logs:
                    tracker.on_step_end(global_step=0)
        self.assertTrue(any("produced no file" in line for line in logs.output))


class MemorySnapshotRankGatingTests(unittest.TestCase):
    def test_unselected_rank_is_silent(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with mock.patch.dict(os.environ, {"RANK": "5"}, clear=False):
                tracker = ms.MemorySnapshotTracker(enable=False, output_dir=out_dir, mem_info=True, ranks={0})
                stub, calls, record_calls = _stub_device_module(2**30, 2**30, 2**30)
                with (
                    mock.patch.object(ms, "_device_memory_module", return_value=stub),
                    mock.patch.object(ms, "_device_synchronize"),
                ):
                    # No-backend validation must also be skipped for unselected ranks.
                    with mock.patch.object(ms, "_device_memory_module", return_value=None):
                        with self.assertNoLogs(ms.logger, level="WARNING"):
                            tracker.on_train_begin()
                    tracker.on_step_start()
                    with self.assertNoLogs(ms.logger, level="INFO"):
                        tracker.on_step_end(global_step=0)
                    tracker.on_error(global_step=0, exc=torch.OutOfMemoryError("oom"))
        self.assertEqual(0, calls["reset"])
        self.assertEqual(0, len(record_calls))
        self.assertEqual(1, tracker.micro_step)  # counter still advances
        self.assertFalse(os.path.exists(os.path.join(out_dir, "memory_snapshot")))

    def test_selected_rank_collects_dump_and_history(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with mock.patch.dict(os.environ, {"RANK": "3"}, clear=False):
                tracker = ms.MemorySnapshotTracker(enable=True, output_dir=out_dir, ranks={0, 3})
                stub, calls, record_calls = _stub_device_module(0, 0, 0)
                with (
                    mock.patch.object(ms, "_device_memory_module", return_value=stub),
                    mock.patch.object(ms, "_device_synchronize"),
                ):
                    tracker.on_train_begin()
                    self.assertEqual([{"stacks": "all", "max_entries": sys.maxsize}], record_calls)
                    tracker.on_step_start()
                    with self.assertLogs(ms.logger, level="INFO"):
                        tracker.on_step_end(global_step=0)
                    self.assertEqual(1, calls["dump"])
            self.assertEqual(
                1, len(glob.glob(os.path.join(out_dir, "memory_snapshot", "rank3_step_micro1_global0_*.pkl")))
            )


class MemorySnapshotWindowTrackerTests(unittest.TestCase):
    def test_oom_trap_respects_window(self):
        with tempfile.TemporaryDirectory() as out_dir:
            tracker = ms.MemorySnapshotTracker(
                enable=False, output_dir=out_dir, mem_info=False, oom_snapshot=True, start_step=1
            )
            stub, calls, record_calls = _stub_device_module(0, 0, 0)
            with (
                mock.patch.object(ms, "_device_memory_module", return_value=stub),
                mock.patch.object(ms, "_device_synchronize"),
            ):
                tracker.on_train_begin()
                self.assertEqual(0, len(record_calls))
                tracker.on_step_start()  # step 0: trap not armed yet
                tracker.on_error(global_step=0, exc=torch.OutOfMemoryError("oom"))
                self.assertEqual(0, calls["dump"])
                tracker.on_step_end(global_step=0)
                tracker.on_step_start()  # step 1: armed
                self.assertEqual([{"stacks": "all", "max_entries": sys.maxsize}], record_calls)
                tracker.on_error(global_step=1, exc=torch.OutOfMemoryError("oom"))
                self.assertEqual(1, calls["dump"])

    def test_oom_trap_relocates_native_oom_snapshots_from_cwd(self):
        cwd = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(cwd)
        try:
            own_name = f"oom_snapshot_{os.getpid()}_20261012120000.pickle"
            other_name = "oom_snapshot_999999_20261012120000.pickle"
            with open(own_name, "wb") as f:
                f.write(_DUMP_MARKER)
            with open(other_name, "wb") as f:
                f.write(b"other-rank")

            with tempfile.TemporaryDirectory() as out_dir:
                with mock.patch.dict(os.environ, {"RANK": "7"}, clear=False):
                    tracker = ms.MemorySnapshotTracker(enable=False, output_dir=out_dir, oom_snapshot=True)
                    stub, calls, _ = _stub_device_module(0, 0, 0)
                    with (
                        mock.patch.object(ms, "_device_memory_module", return_value=stub),
                        mock.patch.object(ms, "_device_synchronize"),
                    ):
                        tracker.on_train_begin()
                        tracker.on_step_start()
                        tracker.on_error(global_step=0, exc=torch.OutOfMemoryError("oom"))

                moved = glob.glob(os.path.join(out_dir, "memory_snapshot", "rank7_native_oom_snapshot_*.pickle"))
                self.assertEqual(1, len(moved))
                self.assertEqual(own_name, os.path.basename(moved[0]).removeprefix("rank7_native_"))
                with open(moved[0], "rb") as f:
                    self.assertEqual(_DUMP_MARKER, f.read())
                # another rank's file must stay in the shared working directory
                self.assertEqual([other_name], os.listdir(cwd))
        finally:
            os.chdir(old_cwd)
            for leftover in glob.glob(os.path.join(cwd, "oom_snapshot_*")):
                os.remove(leftover)
            os.rmdir(cwd)

    def test_native_oom_snapshot_dir_env_takes_priority(self):
        cwd = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(cwd)
        try:
            own_name = f"oom_snapshot_{os.getpid()}_20261012130000.pickle"
            with open(own_name, "wb") as f:
                f.write(_DUMP_MARKER)

            with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as native_dir:
                with mock.patch.dict(
                    os.environ,
                    {"RANK": "2", ms.NATIVE_OOM_DIR_ENV: native_dir},
                    clear=False,
                ):
                    tracker = ms.MemorySnapshotTracker(enable=False, output_dir=out_dir, oom_snapshot=True)
                    stub, _, _ = _stub_device_module(0, 0, 0)
                    with (
                        mock.patch.object(ms, "_device_memory_module", return_value=stub),
                        mock.patch.object(ms, "_device_synchronize"),
                    ):
                        tracker.on_train_begin()
                        tracker.on_step_start()
                        tracker.on_error(global_step=0, exc=torch.OutOfMemoryError("oom"))

                # OOM_SNAPSHOT_PATH wins over the tracker dump dir
                moved = glob.glob(os.path.join(native_dir, "rank2_native_oom_snapshot_*.pickle"))
                self.assertEqual(1, len(moved))
                with open(moved[0], "rb") as f:
                    self.assertEqual(_DUMP_MARKER, f.read())
                self.assertEqual([], glob.glob(os.path.join(out_dir, "memory_snapshot", "*native*")))
                self.assertEqual([], os.listdir(cwd))
        finally:
            os.chdir(old_cwd)
            for leftover in glob.glob(os.path.join(cwd, "oom_snapshot_*")):
                os.remove(leftover)
            os.rmdir(cwd)

    def test_native_oom_snapshot_written_to_target_dir_is_renamed_in_place(self):
        cwd = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(cwd)
        try:
            with tempfile.TemporaryDirectory() as native_dir:
                # New-runtime layout: the allocator wrote straight into
                # OOM_SNAPSHOT_PATH with its hardcoded pid-based name.
                own_name = f"oom_snapshot_{os.getpid()}_20261012140000.pickle"
                other_name = "oom_snapshot_999999_20261012140000.pickle"
                prefixed = "rank0_native_oom_snapshot_111111_20261011140000.pickle"
                for name, payload in ((own_name, _DUMP_MARKER), (other_name, b"other"), (prefixed, b"done")):
                    with open(os.path.join(native_dir, name), "wb") as f:
                        f.write(payload)

                with tempfile.TemporaryDirectory() as out_dir:
                    with mock.patch.dict(
                        os.environ,
                        {"RANK": "9", ms.NATIVE_OOM_DIR_ENV: native_dir},
                        clear=False,
                    ):
                        tracker = ms.MemorySnapshotTracker(enable=False, output_dir=out_dir, oom_snapshot=True)
                        stub, _, _ = _stub_device_module(0, 0, 0)
                        with (
                            mock.patch.object(ms, "_device_memory_module", return_value=stub),
                            mock.patch.object(ms, "_device_synchronize"),
                        ):
                            tracker.on_train_begin()
                            tracker.on_step_start()
                            tracker.on_error(global_step=0, exc=torch.OutOfMemoryError("oom"))

                self.assertEqual([], os.listdir(cwd))  # nothing ever written here
                entries = sorted(os.listdir(native_dir))
                self.assertEqual(
                    [other_name, prefixed, f"rank9_native_{own_name}"],
                    entries,
                )
                renamed = os.path.join(native_dir, f"rank9_native_{own_name}")
                with open(renamed, "rb") as f:
                    self.assertEqual(_DUMP_MARKER, f.read())
        finally:
            os.chdir(old_cwd)
            os.rmdir(cwd)


class MemorySnapshotTrackerTests(unittest.TestCase):
    def test_disabled_tracker_is_inert(self):
        tracker = ms.MemorySnapshotTracker(enable=False, output_dir=tempfile.gettempdir())
        tracker.on_train_begin()
        tracker.on_step_start()
        tracker.on_step_end(0)
        tracker.on_error(0, exc=torch.OutOfMemoryError("oom"))
        self.assertEqual(1, tracker.micro_step)  # counter advances; nothing is logged or dumped

    def test_no_device_backend_raises(self):
        tracker = ms.MemorySnapshotTracker(enable=False, output_dir=tempfile.gettempdir(), mem_info=True)
        with mock.patch.object(ms, "_device_memory_module", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "no accelerator memory API"):
                tracker.on_train_begin()
        self.assertTrue(tracker.mem_info)  # the failure path must not silently disable collection

    def test_mem_info_logs_and_tracks_peak(self):
        with tempfile.TemporaryDirectory() as out_dir:
            tracker = ms.MemorySnapshotTracker(enable=False, output_dir=out_dir, mem_info=True)
            stub, calls, _ = _stub_device_module(allocated=2**30, reserved=3 * 2**30, peak=5 * 2**30)
            with (
                mock.patch.object(ms, "_device_memory_module", return_value=stub),
                mock.patch.object(ms, "_device_synchronize"),
            ):
                tracker.on_train_begin()
                tracker.on_step_start()
                with self.assertLogs(ms.logger, level="INFO") as logs:
                    tracker.on_step_end(global_step=3)
        self.assertEqual(1, calls["reset"])
        self.assertEqual(1, tracker.micro_step)
        self.assertEqual(5 * 2**30, tracker.peak_allocated_bytes)
        self.assertIn("peak_step=5.00GiB", "".join(logs.output))
        self.assertIn("global=3", "".join(logs.output))
        self.assertFalse(os.path.exists(os.path.join(out_dir, "memory_snapshot")))

    def test_enable_records_history_and_dumps_natively(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with mock.patch.dict(os.environ, {"RANK": "7"}, clear=False):
                tracker = ms.MemorySnapshotTracker(enable=True, output_dir=out_dir)
                stub, calls, record_calls = _stub_device_module(0, 0, 0)
                with (
                    mock.patch.object(ms, "_device_memory_module", return_value=stub),
                    mock.patch.object(ms, "_device_synchronize"),
                ):
                    tracker.on_train_begin()
                    self.assertEqual([{"stacks": "all", "max_entries": sys.maxsize}], record_calls)
                    tracker.on_step_start()
                    tracker.on_step_end(global_step=1)
                    # An exception without the OOM trap must not dump or stop.
                    tracker.on_error(global_step=1, exc=RuntimeError("boom"))
                    self.assertEqual(1, calls["dump"])
                    self.assertEqual("recording", tracker._history_state)

            dump_dir = os.path.join(out_dir, "memory_snapshot")
            matches = glob.glob(os.path.join(dump_dir, "rank7_step_micro1_global1_*.pkl"))
            self.assertEqual(1, len(matches))
            # Time suffix, e.g. ..._global1_2026-10-10-10-00.pkl
            self.assertRegex(matches[0], r"rank7_step_micro1_global1_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}\.pkl$")
            with open(matches[0], "rb") as f:
                self.assertEqual(_DUMP_MARKER, f.read())
            self.assertEqual(1, len(os.listdir(dump_dir)))

    def test_dump_without_any_snapshot_api_raises(self):
        stub, calls, _ = _stub_device_module(0, 0, 0, with_snapshot=False, with_fallback=False)
        with tempfile.TemporaryDirectory() as out_dir:
            tracker = ms.MemorySnapshotTracker(enable=True, output_dir=out_dir)
            with (
                mock.patch.object(ms, "_device_memory_module", return_value=stub),
                mock.patch.object(ms, "_device_synchronize"),
            ):
                with self.assertRaisesRegex(RuntimeError, "neither _dump_snapshot nor memory_snapshot"):
                    tracker.on_train_begin()
        self.assertEqual(0, calls["snapshot"])

    def test_dump_falls_back_to_module_memory_snapshot(self):
        stub, calls, _ = _stub_device_module(0, 0, 0, with_snapshot=False, with_fallback=True)
        with tempfile.TemporaryDirectory() as out_dir:
            with mock.patch.dict(os.environ, {"RANK": "0"}, clear=False):
                tracker = ms.MemorySnapshotTracker(enable=True, output_dir=out_dir)
                with (
                    mock.patch.object(ms, "_device_memory_module", return_value=stub),
                    mock.patch.object(ms, "_device_synchronize"),
                ):
                    tracker.on_train_begin()
                    tracker.on_step_end(global_step=0)
            self.assertEqual(
                1, len(glob.glob(os.path.join(out_dir, "memory_snapshot", "rank0_step_micro1_global0_*.pkl")))
            )
            self.assertEqual(1, calls["snapshot"])

    def test_oom_trap_is_silent_and_dumps_only_on_oom(self):
        with tempfile.TemporaryDirectory() as out_dir:
            tracker = ms.MemorySnapshotTracker(enable=False, output_dir=out_dir, oom_snapshot=True)
            stub, calls, record_calls = _stub_device_module(0, 0, 0)
            with (
                mock.patch.object(ms, "_device_memory_module", return_value=stub),
                mock.patch.object(ms, "_device_synchronize"),
            ):
                tracker.on_train_begin()
                self.assertEqual([{"stacks": "all", "max_entries": sys.maxsize}], record_calls)
                tracker.on_step_start()
                with self.assertNoLogs(ms.logger, level="INFO"):
                    tracker.on_step_end(global_step=5)
                self.assertEqual(1, tracker.micro_step)  # counter advances for dump filenames
                # A non-OOM error must not trigger a dump in the silent trap.
                tracker.on_error(global_step=5, exc=RuntimeError("nan loss"))
                self.assertEqual(0, calls["dump"])
                # The OOM error dumps once and stops the history recording.
                tracker.on_error(global_step=5, exc=torch.OutOfMemoryError("npu oom"))
                self.assertEqual(1, calls["dump"])
                self.assertEqual({"enabled": None}, record_calls[-1])
            self.assertEqual(
                1, len(glob.glob(os.path.join(out_dir, "memory_snapshot", "rank0_error_micro1_global5_*.pkl")))
            )

    def test_oom_trap_composes_with_mem_info(self):
        with tempfile.TemporaryDirectory() as out_dir:
            tracker = ms.MemorySnapshotTracker(enable=False, output_dir=out_dir, mem_info=True, oom_snapshot=True)
            stub, calls, record_calls = _stub_device_module(2**30, 2**30, 2**30)
            with (
                mock.patch.object(ms, "_device_memory_module", return_value=stub),
                mock.patch.object(ms, "_device_synchronize"),
            ):
                tracker.on_train_begin()
                self.assertEqual(1, calls["record"])
                with self.assertLogs(ms.logger, level="INFO"):
                    tracker.on_step_end(global_step=0)
                self.assertEqual(0, calls["dump"])  # stats lines, but no per-step dump
                tracker.on_error(global_step=0, exc=torch.OutOfMemoryError("oom"))
                self.assertEqual(1, calls["dump"])

    def test_error_dump_failure_does_not_raise(self):
        stub, _, _ = _stub_device_module(0, 0, 0)

        def _boom(path):
            raise RuntimeError("snapshot backend broke")

        stub.memory._dump_snapshot = _boom
        with tempfile.TemporaryDirectory() as out_dir:
            tracker = ms.MemorySnapshotTracker(enable=False, output_dir=out_dir, oom_snapshot=True)
            with mock.patch.object(ms, "_device_memory_module", return_value=stub):
                tracker.on_error(global_step=0, exc=torch.OutOfMemoryError("oom"))  # must not raise


if __name__ == "__main__":
    unittest.main()

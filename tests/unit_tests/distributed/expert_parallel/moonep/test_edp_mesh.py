# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU construction of the smallest mesh that exposes unsynced MoonEP replicas.

``edp = world / (efsdp * ep)``. With world=4, EP=2, efsdp=1 the mesh is
``(edp, efsdp, ep) = (2, 1, 2)``: two EP groups, each holding a full copy of the
experts. fused HSDP all-reduces those copies on the replicate dim; MoonEP's
``sync_replica_gradients`` is the matching SUM, active when an ``edp`` group is
passed to the runtime. This file only checks the mesh comes up on gloo.
It is not a correctness ST and does not claim ``edp>1`` is supported.

Four ranks are enough: the 32-card story (``FSDP=32, EP=16, efsdp=1``) gives the
same ``edp=2`` ratio, and an unsynced copy shows up at two replicas just as it
does at sixteen, so exposing it does not need a second node.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch.distributed.tensor")

WORKER = pathlib.Path(__file__).with_name("_edp_mesh_worker.py")


def repository_root():
    for candidate in pathlib.Path(__file__).resolve().parents:
        if (candidate / "fsdp_turbo").is_dir():
            return candidate
    raise RuntimeError(f"cannot locate the repository root above {__file__}")


REPO_ROOT = repository_root()
WORLD_SIZE = 4
# DeviceMesh (2, 1, 2) over ranks 0..3: EP groups {0,1} and {2,3};
# EDP (replica) groups {0,2} and {1,3}. Replica SUM of the rank ids:
# {0,2} → 2.0, {1,3} → 4.0.
EXPECTED_EP = {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]}
EXPECTED_EDP = {0: [0, 2], 1: [1, 3], 2: [0, 2], 3: [1, 3]}
EXPECTED_REPLICA_SUM = {0: 2.0, 1: 4.0, 2: 2.0, 3: 4.0}


def run_ranks():
    rendezvous = pathlib.Path(tempfile.gettempdir()) / f"moonep_edp_rdzv_{os.getpid()}_{time.monotonic_ns()}"
    rendezvous.unlink(missing_ok=True)

    env = dict(os.environ)
    env.update(
        RENDEZVOUS_FILE=str(rendezvous),
        WORLD_SIZE=str(WORLD_SIZE),
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
    )
    if sys.platform.startswith("linux"):
        env.setdefault("GLOO_SOCKET_IFNAME", "lo")
    processes = []
    for rank in range(WORLD_SIZE):
        processes.append(
            subprocess.Popen(  # pylint: disable=consider-using-with
                [sys.executable, str(WORKER)],
                env={**env, "RANK": str(rank)},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    records = {}
    failures = []
    for rank, process in enumerate(processes):
        try:
            out, err = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
            failures.append(f"rank {rank} timed out")
            continue
        finally:
            rendezvous.unlink(missing_ok=True)
        if process.returncode != 0:
            failures.append(f"rank {rank} exited {process.returncode}:\n{err}")
            continue
        line = next((ln for ln in out.splitlines() if ln.startswith("{")), None)
        if line is None:
            failures.append(f"rank {rank} printed no result:\n{out}\n{err}")
            continue
        records[rank] = json.loads(line)

    if failures:
        pytest.skip("cannot run a 4-rank gloo group here: " + "; ".join(failures))
    return records


def test_world4_ep2_efsdp1_constructs_two_replicas():
    records = run_ranks()
    assert set(records) == {0, 1, 2, 3}
    for rank, record in records.items():
        assert record["edp_size"] == 2, record
        assert record["ep_size"] == 2, record
        assert record["ep_ranks"] == EXPECTED_EP[rank], record
        assert record["edp_ranks"] == EXPECTED_EDP[rank], record
        assert record["replica_sum"] == pytest.approx(EXPECTED_REPLICA_SUM[rank]), record

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for MoonEP's trace markers reading the config, not the environ.

The knobs moved from ``MOONEP_TRACE`` / ``MOONEP_TRACE_SYNC`` environment
variables to ``moonep_config`` fields; the markers must be inert unless the
per-runtime config asks for them, and ``trace_sync_enabled`` must serialize
each marker through the device module's current stream.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.distributed.expert_parallel.moonep.settings import MoonEPSettings  # noqa: E402
from fsdp_turbo.distributed.expert_parallel.moonep.trace import trace  # noqa: E402


class _FakeStream:
    def __init__(self):
        self.synchronized = 0

    def synchronize(self):
        self.synchronized += 1


def _runtime(trace_enabled=False, trace_sync_enabled=False):
    stream = _FakeStream()
    return (
        SimpleNamespace(
            config=MoonEPSettings(trace_enabled=trace_enabled, trace_sync_enabled=trace_sync_enabled),
            device_module=SimpleNamespace(current_stream=lambda: stream),
            rank=3,
            call_index=17,
        ),
        stream,
    )


def test_trace_is_silent_when_the_config_leaves_it_off(capsys):
    runtime, stream = _runtime()

    trace(runtime, "dispatch.enter")

    captured = capsys.readouterr()
    assert captured.err == ""
    assert stream.synchronized == 0


def test_trace_emits_the_rank_call_and_phase(capsys):
    runtime, _ = _runtime(trace_enabled=True)

    trace(runtime, "prefetch.await", "gate_up")

    captured = capsys.readouterr()
    assert "rank=3" in captured.err
    assert "call=17" in captured.err
    assert "prefetch.await gate_up" in captured.err
    assert captured.err.endswith("\n")


def test_trace_sync_serializes_through_the_current_stream(capsys):
    runtime, stream = _runtime(trace_enabled=True, trace_sync_enabled=True)

    trace(runtime, "combine.exit")

    assert stream.synchronized == 1
    assert "combine.exit" in capsys.readouterr().err


def test_trace_without_sync_never_touches_the_stream(capsys):
    runtime, stream = _runtime(trace_enabled=True)

    trace(runtime, "combine.exit")

    assert stream.synchronized == 0

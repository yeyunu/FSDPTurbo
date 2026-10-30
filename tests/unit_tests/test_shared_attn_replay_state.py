# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU regressions for the shared-attention replay isolation (PR #173 review).

A checkpoint replay used to run against the *live* ``shared_attn`` slots. By backward time those
slots hold whatever the last ``kv_source_layers`` group published, so a consumer replayed for an
earlier group read the wrong version, and a replayed source's own write was discarded only to be
read back from the stale slot. Each checkpoint now captures the slots when it is wired (the start
of that module's forward, when the slots still belong to the preceding sources) and replays on a
private copy; the live state is restored when the replay leaves.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fsdp_turbo.memory.recompute.recompute import (
    recompute_wrapper,
    register_replay_context,
    unregister_replay_context,
)

# Importing the runtime pulls in `fsdp_turbo.models.__init__`, which needs the vendored
# transformers build; skip rather than error where that build is absent.
pytest.importorskip("transformers.models.qwen4_exp", reason="needs the vendored transformers build")
from fsdp_turbo.models.deepseek.deepseek_v41.runtime import (  # noqa: E402  pylint: disable=wrong-import-position
    SharedAttentionRuntime,
)


def test_replay_window_isolates_writes_and_restores_live_state():
    runtime = SharedAttentionRuntime()
    runtime.publish("compress_kv", torch.tensor([1.0]))
    captured = runtime.snapshot()
    runtime.publish("compress_kv", torch.tensor([2.0]))

    window = runtime.replay_window(captured)
    with window:
        assert runtime.compress_kv.item() == 1.0, "the replay must see the captured state"
        runtime.publish("compress_kv", torch.tensor([3.0]))
        assert runtime.compress_kv.item() == 3.0, "replay writes stay inside the window"
    assert runtime.compress_kv.item() == 2.0, "the live state must come back on exit"

    # torch re-enters the replay context on every recompute: the same object must be reusable.
    with window:
        assert runtime.compress_kv.item() == 1.0
    assert runtime.compress_kv.item() == 2.0


def test_snapshot_is_taken_per_checkpoint_not_at_replay_time():
    runtime = SharedAttentionRuntime()
    captured = []

    @register_replay_context
    def barrier(module):
        captured.append(runtime.compress_kv.item() if runtime.compress_kv is not None else None)
        return runtime.replay_window(runtime.snapshot())

    class Consumer(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.values = []

        def forward(self, x):
            value = runtime.compress_kv.reshape(-1)[0]
            self.values.append(float(value))
            return x * value

    first, second = Consumer("first"), Consumer("second")
    first_forward = recompute_wrapper(first.forward, module=first)
    second_forward = recompute_wrapper(second.forward, module=second)

    try:
        runtime.publish("compress_kv", torch.tensor([1.0]))
        hidden = first_forward(torch.ones(1, requires_grad=True))
        runtime.publish("compress_kv", torch.tensor([2.0]))
        out = second_forward(hidden)
        out.sum().backward()
    finally:
        unregister_replay_context(barrier)

    assert captured == [1.0, 2.0], "the barrier must capture at forward time, once per checkpoint"
    assert first.values == [1.0, 1.0], "the first group's replay must not read the last group's slot"
    assert second.values == [2.0, 2.0]

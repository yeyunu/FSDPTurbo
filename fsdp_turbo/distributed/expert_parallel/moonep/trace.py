# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Per-rank progress markers for localising a hang inside MoonEP.

A hang in MoonEP's collectives yields no traceback, no device error and no
timeout, so the only way to localise it is to know which rank stopped at which
phase of which expert layer.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fsdp_turbo.distributed.expert_parallel.moonep.runtime import MoonEPRuntime


def trace(runtime: "MoonEPRuntime", phase: str, detail: str = "") -> None:
    """Emit a per-rank progress marker when ``moonep_config.trace_enabled`` is set.

    Every entry point only enqueues work, so a plain trace reports where the host
    got to, not where the device stalled. ``trace_sync_enabled`` turns each marker
    into a device checkpoint; debug only, it serializes the pipeline.
    """
    config = runtime.config
    if not config.trace_enabled:
        return
    if config.trace_sync_enabled:
        runtime.device_module.current_stream().synchronize()
    suffix = f" {detail}" if detail else ""
    sys.stderr.write(
        f"[moonep rank={runtime.rank} call={runtime.call_index} t={time.monotonic():.3f}] {phase}{suffix}\n"
    )
    sys.stderr.flush()

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MoonEP: perfectly balanced expert parallelism over symmetric memory.

Selected by ``ep_plan.dispatcher = "moonep"``. ``dispatcher`` patches the expert
forward, ``runtime`` owns the symmetric weights and per-step state, ``autograd``
places the collectives in the graph, ``plan`` holds the static-shape layout,
``backend`` / ``backend_npu`` hide all device-specific work, and
``settings`` / ``streams`` / ``trace`` are support. Only the three names below are
imported from outside this package.
"""

from fsdp_turbo.distributed.expert_parallel.moonep.dispatcher import (
    parallelize_moonep_model,
    parallelize_moonep_module,
)
from fsdp_turbo.distributed.expert_parallel.moonep.runtime import MoonEPRuntime

__all__ = ["MoonEPRuntime", "parallelize_moonep_model", "parallelize_moonep_module"]

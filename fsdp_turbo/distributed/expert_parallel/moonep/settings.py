# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Normalized view of the user's MoonEP configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fsdp_turbo.fsdp_turbo_config import MoonEPConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoonEPSettings:
    """Normalized, immutable view of :class:`MoonEPConfig`.

    Read through this rather than off the plan's config object so a config that
    predates a knob degrades to the default here instead of raising
    ``AttributeError`` inside an expert forward.
    """

    num_sms: int = 32
    token_padding: int = 128
    comm_stream_priority: int = -1
    async_finish: bool = False
    tokens_per_rank: int | None = None
    top_k: int | None = None
    recompute_dispatch: bool = False
    validate_plan_invariants: bool = False
    shmem_ip_port: str | None = None
    # Ascend only: CANN SHMEM's stream-ordered global barrier before the
    # weight-gradient reduction, instead of a host synchronize plus dist.barrier.
    shmem_stream_barrier: bool = True
    trace_enabled: bool = False
    trace_sync_enabled: bool = False
    vmm_probe_enabled: bool = False
    npu_async_comm: bool = False

    # Knobs that do nothing on one accelerator. Setting one is not an error, but
    # "I turned it on and saw no change" is expensive to work out from outside.
    _INERT_BY_DEVICE = {
        # Both are CANN SHMEM concepts.
        "cuda": ("shmem_stream_barrier", "shmem_ip_port"),
        # The Ascend backend reports supports_async_comm=False by default, so
        # the request is masked off before it reaches MoonEP: no side stream
        # carries work and its priority has no target.
        "npu": ("async_finish", "comm_stream_priority"),
    }

    #: Entries of the table above that stop being inert the moment the backend
    #: does have an asynchronous communication stream -- on Ascend that is
    #: ``npu_async_comm=True``, which gives both knobs a target.
    _ASYNC_ONLY = frozenset({"async_finish", "comm_stream_priority"})

    def inert_knobs(self, device_type: str, supports_async_comm: bool = False) -> tuple:
        """Names of set-but-ignored knobs on ``device_type``, in field order."""
        candidates = self._INERT_BY_DEVICE.get(device_type, ())
        if supports_async_comm:
            candidates = tuple(name for name in candidates if name not in self._ASYNC_ONLY)
        defaults = type(self)()
        return tuple(name for name in candidates if getattr(self, name) != getattr(defaults, name))

    def warn_about_inert_knobs(self, device_type: str, supports_async_comm: bool = False) -> None:
        inert = self.inert_knobs(device_type, supports_async_comm)
        if inert:
            logger.warning(
                "moonep_config %s %s no effect on %s and %s ignored.",
                ", ".join(inert),
                "has" if len(inert) == 1 else "have",
                device_type,
                "is" if len(inert) == 1 else "are",
            )

    def report_effective_async(self, device_type: str, supported: bool) -> None:
        """State what asynchronous completion actually resolved to.

        A config that reads ``async_finish=True`` while nothing overlaps is how
        somebody concludes the communication is overlapping when it is not.
        """
        if self.async_finish and not supported:
            logger.info(
                "moonep_config.async_finish=True has no effect on %s: the backend "
                "has no usable asynchronous stream, so the collectives run on the "
                "compute stream.",
                device_type,
            )

    @classmethod
    def from_config(cls, config: "MoonEPConfig | None") -> "MoonEPSettings":
        if config is None:
            return cls()
        missing = []
        values = {}
        for field in fields(cls):
            if hasattr(config, field.name):
                values[field.name] = getattr(config, field.name)
            else:
                missing.append(field.name)
        if missing:
            logger.warning(
                "MoonEPConfig does not define %s; using the adapter defaults. Update "
                "fsdp_turbo_config.MoonEPConfig to make these options configurable.",
                ", ".join(missing),
            )
        return cls(**values)

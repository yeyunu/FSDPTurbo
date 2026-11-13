# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Centralized environment access used by distributed startup and feature flags."""

import os

_TRUE = frozenset({"1", "true", "yes", "on"})


def get_local_rank() -> int:
    """Return torchrun's local rank, or -1 outside a distributed launcher."""
    return int(os.environ.get("LOCAL_RANK", "-1"))


def sparse_flash_mla_validation_enabled() -> bool:
    """Enable expensive sparse-attention mask/index checks only when explicitly set to 1."""
    return os.environ.get("FSDP_TURBO_VALIDATE_SPARSE_FLASH_MLA", "0") == "1"


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag; ``1``/``true``/``yes``/``on`` (any case) are true.

    Unset variables keep ``default``. Any other value is false, which lets a
    caller disable a switch without deleting it from a launch script.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


__all__ = ["env_flag", "get_local_rank", "sparse_flash_mla_validation_enabled"]

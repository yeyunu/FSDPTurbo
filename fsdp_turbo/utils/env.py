# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Centralized environment access used by distributed startup."""

import os


def get_local_rank() -> int:
    """Return torchrun's local rank, or -1 outside a distributed launcher."""
    return int(os.environ.get("LOCAL_RANK", "-1"))

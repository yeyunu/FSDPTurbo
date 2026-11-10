# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Compatibility imports for KCP operators; implementations live in ops."""

from fsdp_turbo.ops import kcp as _kcp_ops
from fsdp_turbo.ops.kcp import (
    build_kcp_context,
    kcp_causal_conv1d,
    kcp_chunk_kda,
)

_ensure_triton_driver = _kcp_ops._ensure_triton_driver

__all__ = ["build_kcp_context", "kcp_causal_conv1d", "kcp_chunk_kda"]

# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Vendored PyPTO-Pro Engram kernels (DAV_3510).

Source: pypto-gym MR !819 (head ``3094a0a``),
``ops/pypto_pro/experimental/ops_transformer/engram/`` (CANN Open Software
License 2.0). That head natively supports H=5120 (single-tile HMode 4) and
De=6144 (segmented embeddings with double buffering); the backward kernel
pairs De/H output tiles per LEFT load (Nest1-key N=4) and caches the wide
RIGHT operand in L1. Two integration deltas: the backward wrapper accepts
``pretransposed_weights=True`` (packed wkv views [M,H,De] / [H,De] skip the
host weight transposes), and its host synchronize was removed because the
post-kernel casts/reductions run on the same stream.
Same-cache kernel parity is covered by the CPU unit tests under
``tests/unit_tests/ops/test_engram.py``. The package requires the in-tree
``pypto_pro`` DSL shipped with CANN 9.3.0 (B140 and later).
"""

from fsdp_turbo.ops.npu.engram_pypto.engram_forward_impl import engram_forward_wrapper
from fsdp_turbo.ops.npu.engram_pypto.engram_backward_impl import engram_backward_wrapper

__all__ = ["engram_forward_wrapper", "engram_backward_wrapper"]

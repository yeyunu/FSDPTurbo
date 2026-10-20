# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from .vocab_parallel import vocab_parallelize_modules, get_vp_modules
from .vocab_fully_shard_parallel import vocab_fully_shard_modules, is_vfsdp_enabled
from .vp_dispatcher import vocab_parallel_embedding_forward

__all__ = [
    "vocab_parallelize_modules",
    "get_vp_modules",
    "vocab_fully_shard_modules",
    "is_vfsdp_enabled",
    "vocab_parallel_embedding_forward",
]

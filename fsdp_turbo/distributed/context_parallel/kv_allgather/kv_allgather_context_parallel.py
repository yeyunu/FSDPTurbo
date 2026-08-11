# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Configuration-driven KV-AllGather function patching."""

from fsdp_turbo.distributed.context_parallel.kv_allgather.kv_allgather_attention import (
    get_kv_allgather_attention_fn,
    get_kv_allgather_compressor_attention_fn,
)
from fsdp_turbo.distributed.context_parallel.patch import apply_function_patch_specs

_KV_ALLGATHER_FUNCTION_PATCH_TYPES = {
    "full_attention": get_kv_allgather_attention_fn,
    "compressor_full_attention": get_kv_allgather_compressor_attention_fn,
}


def kv_allgather_context_parallelize_modules(model, cp_group, cp_plan):
    """Install KV-AllGather attention wrappers and return ``model`` unchanged."""
    apply_function_patch_specs(
        cp_plan.kv_allgather_function_patches,
        cp_group,
        _KV_ALLGATHER_FUNCTION_PATCH_TYPES,
        label="KVAllGather",
    )
    return model

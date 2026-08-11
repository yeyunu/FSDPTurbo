# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Configuration-driven Ulysses function patching."""

from fsdp_turbo.distributed.context_parallel.patch import apply_function_patch_specs
from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_attention import (
    get_ulysses_attention_fn,
    get_ulysses_compressor_attention_fn,
    get_ulysses_transposed_compressor_attention_fn,
)

_ULYSSES_FUNCTION_PATCH_TYPES = {
    "full_attention": get_ulysses_attention_fn,
    "compressor_full_attention": get_ulysses_compressor_attention_fn,
    "compressor_full_attention_transposed": get_ulysses_transposed_compressor_attention_fn,
}


def ulysses_context_parallelize_modules(model, ulysses_group, cp_plan):
    """Install Ulysses attention wrappers and return ``model`` unchanged."""
    apply_function_patch_specs(
        cp_plan.ulysses_function_patches,
        ulysses_group,
        _ULYSSES_FUNCTION_PATCH_TYPES,
        label="Ulysses",
    )
    return model

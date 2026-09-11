# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Configuration-driven KCP function patching."""

from fsdp_turbo.distributed.context_parallel.patch import apply_function_patch_specs
from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_attention import (
    get_ulysses_attention_fn,
    get_ulysses_compressor_attention_fn,
    get_ulysses_transposed_compressor_attention_fn,
)

_KCP_FUNCTION_PATCH_TYPES = {
    "full_attention": get_ulysses_attention_fn,
    "compressor_full_attention": get_ulysses_compressor_attention_fn,
    "compressor_full_attention_transposed": get_ulysses_transposed_compressor_attention_fn,
}


def kcp_context_parallelize_modules(model, kcp_group, cp_plan):
    """Install Ulysses MLA and state-parallel KDA for complete KCP mode."""
    apply_function_patch_specs(
        cp_plan.kcp_function_patches,
        kcp_group,
        _KCP_FUNCTION_PATCH_TYPES,
        label="KCP",
        model=model,
    )
    return model

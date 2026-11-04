# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Configuration-driven Ulysses function patching."""

from fsdp_turbo.distributed.context_parallel.patch import (
    apply_function_patch_specs,
    install_model_context_parallel,
)
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
    """Install Ulysses attention wrappers and recurrent-attention forwards.

    GDN (``type: gated_delta_net``) and KDA (``type: kda``) specs live alongside attention specs in
    ``ulysses_function_patches`` but are class-method patches routed inside
    :func:`apply_function_patch_specs` (hence ``model`` is forwarded).
    """
    apply_function_patch_specs(
        cp_plan.ulysses_function_patches,
        ulysses_group,
        _ULYSSES_FUNCTION_PATCH_TYPES,
        label="Ulysses",
        model=model,
    )
    if any(
        spec["type"] in {"compressor_full_attention", "compressor_full_attention_transposed"}
        for spec in cp_plan.ulysses_function_patches
    ):
        from fsdp_turbo.models.deepseek.deepseek_v4.context_parallel import (
            expose_deepseek_v4_context_parallel_capability,
        )

        expose_deepseek_v4_context_parallel_capability(model)
        install_model_context_parallel(model, ulysses_group, mode="ulysses")
    return model

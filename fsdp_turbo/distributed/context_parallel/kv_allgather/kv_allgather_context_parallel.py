# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Configure model-native KV-AllGather, a packed-TND runtime, or legacy wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fsdp_turbo.distributed.context_parallel.patch import install_model_context_parallel
from fsdp_turbo.distributed.context_parallel.kv_allgather.kv_allgather_attention import (
    get_kv_allgather_attention_fn,
    get_kv_allgather_compressor_attention_fn,
)
from fsdp_turbo.distributed.context_parallel.kv_allgather.tnd import install_tnd_context_parallel
from fsdp_turbo.distributed.context_parallel.patch import apply_function_patch_specs

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Protocol, TypeVar

    from torch import nn
    from torch.distributed import ProcessGroup

    from fsdp_turbo.distributed.context_parallel.patch import FunctionPatchSpec

    _Model = TypeVar("_Model", bound=nn.Module)

    class _KVAllGatherPlan(Protocol):
        """The normalized plan fields consumed by KV-AllGather."""

        kv_allgather_function_patches: Sequence[FunctionPatchSpec]


_KV_ALLGATHER_FUNCTION_PATCH_TYPES = {
    "full_attention": get_kv_allgather_attention_fn,
    "compressor_full_attention": get_kv_allgather_compressor_attention_fn,
}


def kv_allgather_context_parallelize_modules(
    model: _Model, cp_group: ProcessGroup, cp_plan: _KVAllGatherPlan
) -> _Model:
    """Configure native CP through the model API, or install legacy wrappers.

    A model with ``set_context_parallel_group`` uses native CP; a model without
    it needs function patches. Packed TND passes a runtime instead and is
    exclusive with both, since either would split the batch twice.
    """
    if getattr(cp_plan, "tnd_enabled", False):
        return install_tnd_context_parallel(model, cp_group, cp_plan)
    set_group = getattr(model, "set_context_parallel_group", None)
    if callable(set_group):
        if cp_plan.kv_allgather_function_patches:
            raise ValueError("Native KV-AllGather CP cannot be combined with function patches.")
        set_group(cp_group)
        return model
    # Qwen4-Exp attention and GDN carry KV-AllGather in their class-forward
    # patches.  The Ulysses patch installer will bind the size-1 Ulysses group
    # when needed; there is no model-level native group setter to call here.
    qwen_embedded_types = {"qsa", "gated_delta_net"}
    ulysses_types = {spec["type"] for spec in getattr(cp_plan, "ulysses_function_patches", [])}
    if not cp_plan.kv_allgather_function_patches and ulysses_types and ulysses_types <= qwen_embedded_types:
        return model
    if not cp_plan.kv_allgather_function_patches:
        raise TypeError(
            "KV-AllGather CP requires model.set_context_parallel_group(group) or kv_allgather_function_patches."
        )
    apply_function_patch_specs(
        cp_plan.kv_allgather_function_patches,
        cp_group,
        _KV_ALLGATHER_FUNCTION_PATCH_TYPES,
        label="KVAllGather",
        model=model,
    )
    if any(spec["type"] == "compressor_full_attention" for spec in cp_plan.kv_allgather_function_patches):
        from fsdp_turbo.models.deepseek.deepseek_v4.context_parallel import (
            expose_deepseek_v4_context_parallel_capability,
        )

        expose_deepseek_v4_context_parallel_capability(model)
        install_model_context_parallel(model, cp_group, mode="kv_allgather")
    return model

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Configure model-native KV-AllGather or legacy attention function wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fsdp_turbo.distributed.context_parallel.kv_allgather.kv_allgather_attention import (
    get_kv_allgather_attention_fn,
    get_kv_allgather_compressor_attention_fn,
)
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

    Native models implement ``set_context_parallel_group(group)`` and pass the
    group through their own forward call chain. Their API selects native CP
    without a configuration switch; combining it with wrappers raises ValueError.
    Models without that API must supply function patches or a TypeError is raised.
    """
    set_group = getattr(model, "set_context_parallel_group", None)
    if callable(set_group):
        if cp_plan.kv_allgather_function_patches:
            raise ValueError("Native KV-AllGather CP cannot be combined with function patches.")
        set_group(cp_group)
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
    return model

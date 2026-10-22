# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Configuration-driven KV-AllGather function patching."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fsdp_turbo.distributed.context_parallel.kv_allgather.kv_allgather_attention import (
    get_kv_allgather_attention_fn,
    get_kv_allgather_compressor_attention_fn,
    get_kv_allgather_compressor_sparse_attention_fn,
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
    "compressor_sparse_attention": get_kv_allgather_compressor_sparse_attention_fn,
}


def kv_allgather_context_parallelize_modules(
    model: _Model, cp_group: ProcessGroup, cp_plan: _KVAllGatherPlan
) -> _Model:
    """Install KV-AllGather attention wrappers and return ``model`` unchanged."""
    apply_function_patch_specs(
        cp_plan.kv_allgather_function_patches,
        cp_group,
        _KV_ALLGATHER_FUNCTION_PATCH_TYPES,
        label="KVAllGather",
        model=model,
    )
    return model

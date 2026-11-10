# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Configuration-driven legacy one-dimensional KCP installation."""

from functools import partial
from typing import TYPE_CHECKING

from fsdp_turbo.distributed.context_parallel.kcp.kda_state import get_fla_state_kda_forward
from fsdp_turbo.distributed.context_parallel.patch import apply_function_patch_specs
from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_attention import (
    get_ulysses_attention_fn,
    get_ulysses_compressor_attention_fn,
    get_ulysses_transposed_compressor_attention_fn,
)

if TYPE_CHECKING:
    from torch import nn
    from torch.distributed import ProcessGroup

    from fsdp_turbo.fsdp_turbo_config import CPPlanConfig


_KCP_FUNCTION_PATCH_TYPES = {
    # MLA on the KCP group runs the same Ulysses all-to-all wrapper the
    # Ulysses mode installs, just on the KCP group.
    "full_attention": get_ulysses_attention_fn,
    "compressor_full_attention": get_ulysses_compressor_attention_fn,
    "compressor_full_attention_transposed": get_ulysses_transposed_compressor_attention_fn,
}


def kcp_context_parallelize_modules(
    model: "nn.Module",
    kcp_group: "ProcessGroup",
    cp_plan: "CPPlanConfig",
    *,
    kcp_state_group: "ProcessGroup | None" = None,
    ulysses_group: "ProcessGroup | None" = None,
) -> "nn.Module":
    """Install legacy KDA state CP and full-domain MLA on contiguous shards.

    Singleton Ulysses-group and state-group keywords preserve call compatibility;
    two-dimensional execution is exclusively provided by the new P2P adapter.
    """
    state_group = kcp_group if kcp_state_group is None else kcp_state_group
    patch_types = {
        **_KCP_FUNCTION_PATCH_TYPES,
        "kda": partial(
            get_fla_state_kda_forward,
            ulysses_group=ulysses_group,
            kda_backend=cp_plan.kda_backend,
        ),
    }
    for spec in cp_plan.kcp_function_patches:
        is_kda = spec["type"] == "kda"
        apply_function_patch_specs(
            [spec],
            state_group if is_kda else kcp_group,
            patch_types,
            label="KCP",
            model=model,
        )
    return model

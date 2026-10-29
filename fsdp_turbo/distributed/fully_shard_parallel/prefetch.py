# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Helpers for aligning FSDP2's built-in prefetch with FSDPTurbo config."""

from typing import Any, Iterable


def disable_default_backward_prefetch(modules: Iterable[Any]) -> None:
    """Disable FSDP2's implicit one-unit backward prefetch.

    FSDP2 enables ``default_prefetch`` whenever
    ``FSDPState._states_to_backward_prefetch`` is empty.  That makes
    ``num_to_backward_prefetch=0`` a no-op if we only skip registering an
    explicit prefetch list.  Force ``default_prefetch=False`` on every wrapped
    parameter group so that zero really means zero.

    The patch is applied per parameter group and is idempotent, so both dense
    FSDP and expert FSDP (EFSDP) units can be covered by traversing the model.
    """
    for module in modules:
        get_state = getattr(module, "_get_fsdp_state", None)
        if not callable(get_state):
            continue
        state = get_state()
        # A zero depth must also drop any explicitly registered targets.
        state._states_to_backward_prefetch = []
        param_group = getattr(state, "_fsdp_param_group", None)
        if param_group is None:
            continue
        if getattr(param_group, "_fsdp_turbo_default_backward_prefetch_disabled", False):
            continue

        original_pre_backward = param_group.pre_backward

        def pre_backward(default_prefetch, *args, _original=original_pre_backward, **kwargs):
            return _original(False, *args, **kwargs)

        # Instance attribute assignment is intentional: it shadows the bound
        # method without touching the shared FSDPParamGroup class.
        param_group.pre_backward = pre_backward
        param_group._fsdp_turbo_default_backward_prefetch_disabled = True


__all__ = ["disable_default_backward_prefetch"]

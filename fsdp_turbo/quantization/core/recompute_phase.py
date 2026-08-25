# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MXFP8 weight-phase hand-off for non-reentrant activation recomputation."""

from contextlib import contextmanager
from enum import Enum
from threading import Condition, Lock
from typing import Any

_RECOMPUTE_MODULE_MARKER = "_fsdp_turbo_activation_recompute"
_RECOMPUTE_AWARE_GROUP = "_fsdp_turbo_mxfp8_recompute_aware"
_RECOMPUTE_DEPTH = "_fsdp_turbo_mxfp8_recompute_depth"
_WEIGHT_PHASE = "_fsdp_turbo_mxfp8_weight_phase"
_WEIGHT_PHASE_ERROR = "_fsdp_turbo_mxfp8_weight_phase_error"
_WEIGHT_PHASE_TRANSITION = "_fsdp_turbo_mxfp8_weight_phase_transition"
_WEIGHT_PHASE_CONDITION = "_fsdp_turbo_mxfp8_weight_phase_condition"
_CONDITION_CREATION_LOCK = Lock()


class MXFP8WeightPhase(Enum):
    FORWARD = "forward"
    RECOMPUTE_FORWARD = "recompute_forward"
    BACKWARD = "backward"


def mark_module_for_recompute(module: Any) -> None:
    """Mark a module whose original forward is checkpoint-replayed."""
    setattr(module, _RECOMPUTE_MODULE_MARKER, True)


def register_recompute_aware_group(fsdp_state: Any, config: Any) -> bool:
    """Register an FSDP group that owns a checkpoint-replayed module."""
    if getattr(config, "_recompute_aware", False) is not True:
        return False

    param_group = getattr(fsdp_state, "_fsdp_param_group", None)
    if param_group is None:
        return False
    if getattr(param_group, _RECOMPUTE_AWARE_GROUP, False) is True:
        return True

    state_modules = getattr(fsdp_state, "_modules", None)
    modules = list(state_modules or getattr(param_group, "modules", ()))
    # Fine-grained FSDP may install its hooks on an ancestor outside the
    # managed-module tuple. Check that canonical hook target as well.
    hook_module_ref = getattr(fsdp_state, "hook_module", None) or getattr(param_group, "hook_module", None)
    hook_module = hook_module_ref() if callable(hook_module_ref) else hook_module_ref
    if hook_module is not None and all(hook_module is not module for module in modules):
        modules.append(hook_module)
    for module in modules:
        if any(getattr(child, _RECOMPUTE_MODULE_MARKER, False) is True for child in module.modules()):
            setattr(param_group, _RECOMPUTE_AWARE_GROUP, True)
            return True
    return False


def select_recompute_aware_phase(fsdp_state: Any, config: Any, training_state: Any) -> MXFP8WeightPhase | None:
    """Return the explicit MXFP8 phase for a recompute-aware FSDP group."""
    if not register_recompute_aware_group(fsdp_state, config):
        return None

    param_group = fsdp_state._fsdp_param_group
    if getattr(param_group, "_reshard_after_forward", True) is False:
        return None
    training_state_name = getattr(training_state, "name", str(training_state))
    if training_state_name == "FORWARD":
        phase = MXFP8WeightPhase.FORWARD
        setattr(param_group, _WEIGHT_PHASE, phase)
        return phase
    if training_state_name != "PRE_BACKWARD":
        return None

    phase = getattr(param_group, _WEIGHT_PHASE, MXFP8WeightPhase.RECOMPUTE_FORWARD)
    if phase == MXFP8WeightPhase.FORWARD:
        phase = MXFP8WeightPhase.RECOMPUTE_FORWARD
        setattr(param_group, _WEIGHT_PHASE, phase)
    return phase


def get_module_fsdp_state(module: Any) -> Any:
    """Resolve the closest FSDP state that owns ``module``."""
    if hasattr(module, "_get_fsdp_state"):
        return module._get_fsdp_state()
    if getattr(module, "_cached_parent_fsdp_state", None) is not None:
        return module._cached_parent_fsdp_state

    from torch.distributed._composable_state import _module_state_mapping

    min_nodes_in_parent = float("inf")
    closest_parent_fsdp_mod = None
    for fsdp_mod in _module_state_mapping.keys():
        all_submodules = list(fsdp_mod.modules())
        if module in all_submodules and min_nodes_in_parent > len(all_submodules):
            closest_parent_fsdp_mod = fsdp_mod
            min_nodes_in_parent = len(all_submodules)
    if closest_parent_fsdp_mod is None:
        raise RuntimeError("Module is not FSDP-wrapped and does not have any FSDP-wrapped parent modules.")
    fsdp_state = closest_parent_fsdp_mod._get_fsdp_state()
    module._cached_parent_fsdp_state = fsdp_state
    return fsdp_state


def _get_fsdp_state(module: Any) -> Any:
    try:
        return get_module_fsdp_state(module)
    except RuntimeError:
        return None


def _switch_weight_phase(param_group: Any, phase: MXFP8WeightPhase) -> None:
    if getattr(param_group, _RECOMPUTE_AWARE_GROUP, False) is not True:
        return
    if getattr(param_group, "_reshard_after_forward", True) is False:
        return

    with _CONDITION_CREATION_LOCK:
        condition = getattr(param_group, _WEIGHT_PHASE_CONDITION, None)
        if condition is None:
            condition = Condition()
            setattr(param_group, _WEIGHT_PHASE_CONDITION, condition)

    with condition:
        while getattr(param_group, _WEIGHT_PHASE_TRANSITION, None) is not None:
            condition.wait()
        phase_error = getattr(param_group, _WEIGHT_PHASE_ERROR, None)
        if phase_error is not None:
            raise RuntimeError(f"MXFP8 weight phase is poisoned by an earlier hand-off failure: {phase_error}")

        current_phase = getattr(param_group, _WEIGHT_PHASE, None)
        if current_phase == phase or (
            current_phase == MXFP8WeightPhase.FORWARD and phase == MXFP8WeightPhase.RECOMPUTE_FORWARD
        ):
            setattr(param_group, _WEIGHT_PHASE, phase)
            return

        # Publish the target before unshard: the pre-all-gather hook must know
        # which pair to materialize. Concurrent autograd nodes wait until the
        # collective commits or poisons this transition.
        setattr(param_group, _WEIGHT_PHASE, phase)
        setattr(param_group, _WEIGHT_PHASE_TRANSITION, phase)

    try:
        compute_done = None
        if param_group.is_unsharded:
            compute_done = param_group.device_handle.Event()
            compute_done.record()
            param_group.reshard()

        if compute_done is not None:
            param_group._wait_all_gather_streams_on_event(compute_done)
        param_group.unshard(param_group.unshard_async_op)
        param_group.wait_for_unshard()

        if not param_group.is_unsharded:
            raise RuntimeError(f"FSDP failed to materialize MXFP8 weights for phase {phase.value!r}")
    except BaseException as error:
        # Do not let a later call treat a partially completed collective as an
        # idempotent success merely because the target phase was already set.
        with condition:
            setattr(param_group, _WEIGHT_PHASE_ERROR, f"{type(error).__name__}: {error}")
            setattr(param_group, _WEIGHT_PHASE_TRANSITION, None)
            condition.notify_all()
        raise
    else:
        with condition:
            setattr(param_group, _WEIGHT_PHASE_TRANSITION, None)
            condition.notify_all()


def ensure_mxfp8_backward_phase(module: Any) -> None:
    """Materialize backward-layout weights before an MXFP8 autograd node."""
    fsdp_state = _get_fsdp_state(module)
    param_group = getattr(fsdp_state, "_fsdp_param_group", None)
    if param_group is not None:
        depth = getattr(param_group, _RECOMPUTE_DEPTH, 0)
        if depth > 1:
            raise RuntimeError("Nested checkpoint replay in one FSDP group cannot hand off MXFP8 weights safely")
        _switch_weight_phase(param_group, MXFP8WeightPhase.BACKWARD)


def resolve_live_post_quant_weight(saved_weight: Any, owner_ref: Any, weight_attr: str):
    """Resolve the FSDP-owned weight after the replay-to-backward hand-off."""
    live_weight = saved_weight
    saved_local = saved_weight.to_local() if hasattr(saved_weight, "to_local") else saved_weight
    if owner_ref is not None:
        if not isinstance(weight_attr, str) or not weight_attr:
            raise RuntimeError("MXFP8 live weight resolution requires a non-empty owner attribute name")
        owner = owner_ref()
        if owner is None:
            raise RuntimeError("MXFP8 weight owner was released before backward")
        ensure_mxfp8_backward_phase(owner)
        live_weight = getattr(owner, weight_attr, None)

    if hasattr(live_weight, "to_local"):
        live_weight = live_weight.to_local()

    from fsdp_turbo.quantization.core.post_quant_weight import PostQuantWeight

    # Keep direct/custom operator calls compatible when they do not provide a
    # module owner. With an owner, only its canonical attribute is trusted so
    # a mis-bound owner cannot be hidden by a stale autograd source chain.
    if owner_ref is None:
        while isinstance(live_weight, PostQuantWeight) and live_weight._weight_bwd is None:
            source = getattr(live_weight, "_source", None)
            if source is None:
                break
            live_weight = source

    if not isinstance(live_weight, PostQuantWeight):
        raise TypeError(
            f"MXFP8 owner attribute {weight_attr!r} does not contain a live PostQuantWeight during backward: "
            f"got {type(live_weight).__name__}"
        )
    if isinstance(saved_local, PostQuantWeight) and tuple(saved_local.shape) != tuple(live_weight.shape):
        raise RuntimeError(
            f"MXFP8 live owner weight shape changed across recompute: "
            f"saved={tuple(saved_local.shape)}, live={tuple(live_weight.shape)}"
        )
    return live_weight


@contextmanager
def mxfp8_recompute_context(module: Any):
    """Use forward-layout weights for checkpoint replay, then hand off."""
    fsdp_state = _get_fsdp_state(module)
    param_group = getattr(fsdp_state, "_fsdp_param_group", None)
    if param_group is None or getattr(param_group, _RECOMPUTE_AWARE_GROUP, False) is not True:
        yield
        return

    depth = getattr(param_group, _RECOMPUTE_DEPTH, 0)
    if depth == 0:
        _switch_weight_phase(param_group, MXFP8WeightPhase.RECOMPUTE_FORWARD)
    setattr(param_group, _RECOMPUTE_DEPTH, depth + 1)
    try:
        yield
    finally:
        depth = getattr(param_group, _RECOMPUTE_DEPTH, 1) - 1
        setattr(param_group, _RECOMPUTE_DEPTH, depth)
        if depth == 0:
            _switch_weight_phase(param_group, MXFP8WeightPhase.BACKWARD)

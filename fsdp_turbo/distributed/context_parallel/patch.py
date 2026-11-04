# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Install context-parallel wrappers through the generic patch framework."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fsdp_turbo.distributed.context_parallel.loss import get_cp_causal_lm_loss_fn
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.patch import patch_namespace_members, resolve_callable, resolve_class_member

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import TypedDict

    from torch import nn
    from torch.distributed import ProcessGroup

    class FunctionPatchSpec(TypedDict):
        """Normalized wrapper kind and dotted target paths from the CP plan."""

        type: str
        target_functions: list[str]


logger = logging.getLogger(__name__)

CP_PATCHES_ATTR = "_fsdp_turbo_cp_patches"
CAUSAL_LM_LOSS = "causal_lm_loss"
GATED_DELTA_NET = "gated_delta_net"
KDA_TYPE = "kda"
QSA_TYPE = "qsa"


def apply_function_patch_specs(
    specs: Sequence[FunctionPatchSpec],
    group: ProcessGroup | None,
    patch_types: Mapping[str, Callable[[Callable[..., object], ProcessGroup | None], Callable[..., object]]],
    label: str,
    model: nn.Module | None = None,
) -> None:
    """Build and install direct replacements for configured CP patch specs.

    ``type: qsa``, ``type: gated_delta_net`` and ``type: kda`` specs are class-method patches deferred to
    architecture-aware helpers (they require ``model``); all other types are
    installed as module-level function wrappers via ``patch_types``.
    """
    for spec in specs:
        patch_type = spec["type"]
        if patch_type == QSA_TYPE:
            apply_qsa_patches(model, spec, group, label)
            continue
        if patch_type == GATED_DELTA_NET:
            apply_gdn_patches(model, spec, label)
            continue
        if patch_type == KDA_TYPE:
            apply_kda_patches(model, spec, group, label)
            continue
        if patch_type not in patch_types:
            raise ValueError(f"Unknown {label} function patch type {patch_type!r}.")
        wrapper = patch_types[patch_type]
        patch_key = f"{label}:{patch_type}"

        for target in spec["target_functions"]:
            original = resolve_callable(target)
            if patch_key in getattr(original, CP_PATCHES_ATTR, ()):
                continue
            replacement = wrapper(original, group)
            applied = set(getattr(original, CP_PATCHES_ATTR, ()))
            setattr(replacement, CP_PATCHES_ATTR, tuple(applied | {patch_key}))

            def on_patch(original_fn, wrapped_fn, *, _label=label, _type=patch_type, _wrapper=wrapper):
                _replace_transformers_registry_references(original_fn, wrapped_fn)
                wrapper_name = getattr(_wrapper, "__name__", type(_wrapper).__name__)
                print_rank(logger.info, f"[Context Parallel/{_label}] Patched {_type} with {wrapper_name}.")

            patch_namespace_members(target, replacement, after_patch=on_patch)


def apply_cp_loss_patches(specs, group):
    """Install causal-LM loss targets once on the complete CP group."""
    apply_function_patch_specs(
        specs,
        group,
        {CAUSAL_LM_LOSS: get_cp_causal_lm_loss_fn},
        label="CP",
    )


def apply_gdn_patches(model, spec, label):
    """Install the GDN common-class forward on configured GatedDeltaNet layers.

    Unlike the module-level function wrapping used by the attention/loss
    patches, GDN targets are ``module.Class.member`` paths patched on matching
    model instances via the generic class-member patch path.
    """
    if model is None:
        raise RuntimeError(f"[Context Parallel/{label}] GDN patches require a model.")

    from fsdp_turbo.modules.gated_delta_net import GatedDeltaNet
    from fsdp_turbo.utils.patch import patch_model_members

    for target in spec["target_functions"]:
        matched = patch_model_members(model, target, GatedDeltaNet.forward)
        if not matched:
            raise RuntimeError(f"[Context Parallel/{label}] GDN patch target did not match any module: {target!r}.")
        print_rank(logger.info, f"[Context Parallel/{label}] Patched gdn: {target}")


def apply_kda_patches(model: nn.Module, spec: FunctionPatchSpec, group: ProcessGroup, label: str) -> None:
    """Install the group-bound forward selected by the patch field."""
    if model is None:
        raise RuntimeError(f"[Context Parallel/{label}] KDA patches require a model.")

    from fsdp_turbo.utils.patch import patch_model_members

    if label == "Ulysses":
        from fsdp_turbo.modules.kda import get_ulysses_kda_forward

        forward_factory = get_ulysses_kda_forward
    elif label == "KCP":
        from fsdp_turbo.distributed.context_parallel.kcp.kda_state import get_fla_state_kda_forward

        forward_factory = get_fla_state_kda_forward
    else:
        raise ValueError(f"KDA patches are not supported in the {label!r} patch field.")

    for target in spec["target_functions"]:
        target_class, member_name = resolve_class_member(target)
        original = getattr(target_class, member_name)
        replacement = forward_factory(original, group)
        matched = patch_model_members(model, target, replacement)
        if not matched:
            raise RuntimeError(f"[Context Parallel/{label}] KDA patch target did not match any module: {target!r}.")
        print_rank(logger.info, f"[Context Parallel/{label}] Patched kda: {target}")


def apply_qsa_patches(model: nn.Module, spec: FunctionPatchSpec, group: ProcessGroup, label: str) -> None:
    """Install the group-bound Qwen QSA attention forward on model instances."""
    if model is None:
        raise RuntimeError(f"[Context Parallel/{label}] QSA patches require a model.")
    if label != "Ulysses":
        raise ValueError(f"QSA patches are not supported in the {label!r} patch field.")

    from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_attention import (
        get_qwen4_exp_ulysses_forward,
    )
    from fsdp_turbo.utils.patch import patch_model_members

    for target in spec["target_functions"]:
        target_class, member_name = resolve_class_member(target)
        original = getattr(target_class, member_name)
        replacement = get_qwen4_exp_ulysses_forward(original, group)
        matched = patch_model_members(model, target, replacement)
        if not matched:
            raise RuntimeError(f"[Context Parallel/{label}] QSA patch target did not match any module: {target!r}.")
        print_rank(logger.info, f"[Context Parallel/{label}] Patched qsa: {target}")


def _replace_transformers_registry_references(original_fn, wrapped_fn):
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        ALL_ATTENTION_FUNCTIONS = None
    if ALL_ATTENTION_FUNCTIONS is not None:
        for mapping_name in ("_global_mapping", "_local_mapping"):
            mapping = getattr(ALL_ATTENTION_FUNCTIONS, mapping_name, None)
            if mapping:
                _replace_mapping_references(mapping, original_fn, wrapped_fn)

    try:
        from transformers.loss import loss_utils
    except ImportError:
        loss_utils = None
    if loss_utils is not None:
        _replace_mapping_references(loss_utils.LOSS_MAPPING, original_fn, wrapped_fn)

    try:
        from transformers import modeling_utils
    except ImportError:
        modeling_utils = None
    if modeling_utils is not None:
        _replace_mapping_references(modeling_utils.LOSS_MAPPING, original_fn, wrapped_fn)


def _replace_mapping_references(mapping, original_fn, wrapped_fn):
    for key, value in list(mapping.items()):
        if value is original_fn:
            mapping[key] = wrapped_fn

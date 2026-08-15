# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Install context-parallel wrappers through the generic patch framework."""

import logging

from fsdp_turbo.distributed.context_parallel.loss import get_cp_causal_lm_loss_fn
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.patch import patch_namespace_members, resolve_callable

logger = logging.getLogger(__name__)

CP_PATCHES_ATTR = "_fsdp_turbo_cp_patches"
CAUSAL_LM_LOSS = "causal_lm_loss"
GATED_DELTA_NET = "gated_delta_net"


def apply_function_patch_specs(specs, group, patch_types, label, model=None):
    """Build and install direct replacements for configured CP patch specs.

    ``type: gated_delta_net`` specs are class-method patches deferred to
    :func:`apply_gdn_patches` (they require ``model``); all other types are
    installed as module-level function wrappers via ``patch_types``.
    """
    for spec in specs:
        patch_type = spec["type"]
        if patch_type == GATED_DELTA_NET:
            apply_gdn_patches(model, spec, label)
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

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Install context-parallel wrappers through the generic patch framework."""

from __future__ import annotations

import logging
import types
from functools import wraps
from typing import TYPE_CHECKING, Literal

import torch
import torch.distributed as dist

from fsdp_turbo.distributed.context_parallel.kv_allgather.utils import kv_allgather
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
CP_GROUP_ATTR = "_fsdp_turbo_cp_group"
CAUSAL_LM_LOSS = "causal_lm_loss"
GATED_DELTA_NET = "gated_delta_net"
KDA_TYPE = "kda"
QSA_TYPE = "qsa"
_CONTEXT_PARALLEL_STATE_KEY = "_fsdp_turbo_context_parallel_state"
_CONTEXT_PARALLEL_ATTR = "_fsdp_turbo_context_parallel"
ContextParallelMode = Literal["ulysses", "kv_allgather"]


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


def _apply_instance_wrapper(model, target, wrapper, group, patch_key):
    """Wrap regular instance methods without changing other models of the same class."""
    if model is None:
        raise RuntimeError(f"CP method patch {target!r} requires a model.")
    target_class, member_name = resolve_class_member(target)
    modules = [module for module in model.modules() if isinstance(module, target_class)]
    if not modules:
        raise RuntimeError(f"CP patch target did not match any module: {target!r}.")
    for module in modules:
        original = getattr(module, member_name)
        if not isinstance(original, types.MethodType) or original.__self__ is not module:
            raise TypeError(f"CP method patch {target!r} requires a regular instance method.")
        applied = set(getattr(original, CP_PATCHES_ATTR, ()))
        if patch_key in applied and getattr(original, CP_GROUP_ATTR, None) is not group:
            raise ValueError(f"CP method patch {target!r} is already installed with a different group.")
        if patch_key in applied:
            continue
        replacement = wrapper(original.__func__, group)
        setattr(replacement, CP_PATCHES_ATTR, tuple(applied | {patch_key}))
        setattr(replacement, CP_GROUP_ATTR, group)
        setattr(module, member_name, types.MethodType(replacement, module))
    print_rank(logger.info, f"[Context Parallel/{patch_key}] Patched instance method: {target}")


def install_model_context_parallel(
    model: nn.Module,
    group: ProcessGroup,
    mode: ContextParallelMode,
) -> None:
    """Install CP input preparation for models exposing a CP attention capability.

    The model supplies ``context_parallel_attention_forward`` and
    ``context_parallel_attention_class``; this generic installer owns only
    process-group state and wrapper lifecycle.
    """
    model_modules = [
        module
        for module in model.modules()
        if callable(getattr(module, "context_parallel_attention_forward", None))
        and isinstance(getattr(module, "context_parallel_attention_class", None), type)
    ]
    if not model_modules:
        raise ValueError("The model does not expose the required context-parallel attention capability.")

    attention_modules = []
    attention_forward_by_module = {}
    for model_module in model_modules:
        attention_class = model_module.context_parallel_attention_class
        matching_modules = [module for module in model_module.modules() if isinstance(module, attention_class)]
        if not matching_modules:
            raise ValueError(
                f"The context-parallel model module {type(model_module).__name__} "
                f"contains no {attention_class.__name__} attention modules."
            )
        for attention_module in matching_modules:
            if attention_module not in attention_forward_by_module:
                attention_modules.append(attention_module)
                attention_forward_by_module[attention_module] = model_module.context_parallel_attention_forward

    if not attention_modules:
        raise ValueError("The model does not expose context-parallel attention modules.")

    for module in [*attention_modules, *model_modules]:
        installed = getattr(module, _CONTEXT_PARALLEL_ATTR, None)
        if installed is None:
            continue
        installed_mode, installed_group = installed
        if installed_mode != mode or installed_group is not group:
            raise ValueError("Context-parallel attention is already installed with a different mode or group.")

    for module in model_modules:
        if getattr(module, _CONTEXT_PARALLEL_ATTR, None) is not None:
            continue
        original_forward = module.forward

        def context_parallel_model_forward(
            model_module,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            _original_forward=original_forward,
            **kwargs,
        ):
            if dist.get_world_size(group) == 1:
                return _original_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    **kwargs,
                )
            if past_key_values is not None:
                raise ValueError("Context parallelism requires past_key_values=None during training.")
            if attention_mask is not None and (
                not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 2
            ):
                raise ValueError("Context parallelism supports only a local 2D attention mask or None.")
            if _CONTEXT_PARALLEL_STATE_KEY in kwargs:
                raise ValueError(f"{_CONTEXT_PARALLEL_STATE_KEY} is reserved for internal CP state.")
            sequence_input = input_ids if input_ids is not None else inputs_embeds
            if sequence_input is None:
                raise ValueError("You must specify input_ids or inputs_embeds.")
            local_length = sequence_input.shape[1]
            if position_ids is None:
                rank_start = dist.get_rank(group) * local_length
                position_ids = torch.arange(
                    rank_start, rank_start + local_length, device=sequence_input.device
                ).unsqueeze(0)
            if not isinstance(position_ids, torch.Tensor) or position_ids.dim() != 2:
                raise ValueError("Context parallelism requires 2D position_ids.")
            if position_ids.shape[-1] != local_length:
                raise ValueError("Context-parallel position_ids must match the local sequence length.")
            if attention_mask is not None and attention_mask.shape[-1] != local_length:
                raise ValueError("Context-parallel attention_mask must match the local sequence length.")
            kwargs[_CONTEXT_PARALLEL_STATE_KEY] = {
                "attention_mask": kv_allgather(attention_mask, group, gather_dim=1)
                if attention_mask is not None
                else None,
                "position_ids": kv_allgather(position_ids, group, gather_dim=1),
            }
            return _original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )

        module.forward = types.MethodType(wraps(original_forward)(context_parallel_model_forward), module)
        setattr(module, _CONTEXT_PARALLEL_ATTR, (mode, group))

    for module in attention_modules:
        if getattr(module, _CONTEXT_PARALLEL_ATTR, None) is not None:
            continue
        original_forward = module.forward
        attention_forward = attention_forward_by_module[module]

        def context_parallel_attention_method(
            attention_module,
            hidden_states,
            position_embeddings,
            position_ids,
            attention_mask,
            past_key_values=None,
            _original_forward=original_forward,
            _attention_forward=attention_forward,
            **kwargs,
        ):
            if dist.get_world_size(group) == 1:
                return _original_forward(
                    hidden_states,
                    position_embeddings,
                    position_ids,
                    attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )
            if past_key_values is not None:
                raise ValueError("Context parallelism requires past_key_values=None during training.")
            cp_state = kwargs.pop(_CONTEXT_PARALLEL_STATE_KEY, None)
            if cp_state is None:
                raise ValueError("Context-parallel model input preparation was not installed.")
            global_length = cp_state["position_ids"].shape[-1]
            local_length = hidden_states.shape[1]
            query_indices = torch.arange(global_length, device=hidden_states.device)
            if mode == "kv_allgather":
                query_indices = query_indices.narrow(0, dist.get_rank(group) * local_length, local_length)
            return _attention_forward(
                attention_module,
                hidden_states,
                position_embeddings,
                query_indices,
                cp_state,
                group,
                kv_allgather,
                kwargs,
            )

        module.forward = types.MethodType(wraps(original_forward)(context_parallel_attention_method), module)
        setattr(module, _CONTEXT_PARALLEL_ATTR, (mode, group))


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

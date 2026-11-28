# Copyright (c) 2024 Meta Inc.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
from typing import Set, List, Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, OffloadPolicy
from torch.distributed._composable_state import _get_module_state
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig, FSDPPlanConfig
from fsdp_turbo.distributed.fine_grained_fully_shard import get_fsdp_strategy
from fsdp_turbo.distributed.expert_parallel.utils import wait_last_expert_a2a_before_prefetch
from fsdp_turbo.distributed.fully_shard_parallel.prefetch import disable_default_backward_prefetch
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match
from fsdp_turbo.utils.dtype import get_dtype


logger = logging.getLogger(__name__)


def fully_shard_parallel_modules(
    model: torch.nn.Module, fsdp_mesh: DeviceMesh, fsdp_plan: FSDPPlanConfig, ep_plan: EPPlanConfig = None
) -> torch.nn.Module:
    """Wrap dense FSDP units while preserving ignored state and expert prefetch coordination."""
    ignored_modules, ignored_params = get_ignored_modules(model, fsdp_plan)
    fsdp_modules = get_fsdp_modules(model, fsdp_plan, ignored_modules)
    hook_modules = get_fsdp_hook_modules(model, fsdp_plan)
    mp_policy = get_mixprecision_policy(fsdp_plan)
    offload_policy = get_offload_policy(fsdp_plan)
    config = {
        'mesh': fsdp_mesh,
        'ignored_params': ignored_params,
        'mp_policy': mp_policy,
        'offload_policy': offload_policy,
    }

    strategy = get_fsdp_strategy(fsdp_plan.fsdp_implementation)
    fully_shard_fn = strategy.get_unified_fully_shard_fn()

    parents = {child: parent for parent in model.modules() for child in parent.children()
               if getattr(child, "_kimi_preserve_gate_precision", False)}
    split_attention = {}
    for module, plan in fsdp_modules.items():
        module_config = config.copy()
        module_config.update(plan)
        if module in parents:
            split_attention[module] = module_config
            continue
        if getattr(module, "_fsdp_turbo_chunk_loss_head", False):
            # This head returns an FP32 loss sum rather than BF16 logits.
            # Keep the scalar in FP32 until normalization outside the head.
            policy = module_config['mp_policy']
            module_config['mp_policy'] = MixedPrecisionPolicy(
                param_dtype=policy.param_dtype,
                reduce_dtype=policy.reduce_dtype,
                output_dtype=None,
                cast_forward_inputs=policy.cast_forward_inputs,
            )
        hook_module = find_hook_module(module, hook_modules)
        fully_shard_fn(module, hook_module=hook_module, **module_config)
    # Wrap independent MLP/expert units first. Each remaining parameter belongs
    # to one group, with the same checkpoint names and runtime mesh as before.
    if parents.keys() != split_attention.keys():
        raise RuntimeError("Each precision-preserving KDA attention needs its own FSDP plan entry")
    for attention, options in split_attention.items():
        decoder = parents[attention]
        if decoder in fsdp_modules:
            raise RuntimeError("KDA decoder and its attention cannot both be pre-wrapped")
        auxiliary = {attention.A_log, attention.dt_bias}
        if any(p.dtype != torch.float32 for p in auxiliary):
            raise RuntimeError("KDA checkpoint auxiliaries must use FP32 storage")
        policy = options["mp_policy"]
        dense_options = dict(options)
        dense_options["ignored_params"] = options["ignored_params"] | auxiliary
        # This extra decoder-boundary hook manages weights only. Leave its input
        # and output activation dtypes as dictated by the model's own forward.
        dense_options["mp_policy"] = MixedPrecisionPolicy(param_dtype=policy.param_dtype,
            reduce_dtype=policy.reduce_dtype, output_dtype=None, cast_forward_inputs=False)
        fully_shard_fn(decoder, **dense_options)
        aux_options = dict(options)
        aux_options["ignored_params"] = options["ignored_params"] | (set(attention.parameters()) - auxiliary)
        aux_options["mp_policy"] = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=policy.reduce_dtype,
            output_dtype=policy.output_dtype, cast_forward_inputs=False)
        fully_shard_fn(attention, **aux_options)
    fully_shard_fn(model, **config)
    if ignored_modules and fsdp_mesh.device_type != "cpu":
        device = torch.device(f"{fsdp_mesh.device_type}:{torch.accelerator.current_device_index()}")
        _move_unmanaged_ignored_modules(model, ignored_modules, device)
    layer_expert_units: dict[torch.nn.Module, list[torch.nn.Module]] = {}
    layer_moe_blocks: dict[torch.nn.Module, torch.nn.Module] = {}
    build_expert_map = fsdp_plan.num_to_forward_prefetch > 0 or fsdp_plan.backward_prefetch_on_moe_input
    if build_expert_map and ep_plan is not None and ep_plan.apply_efsdp_modules:
        layer_expert_units, layer_moe_blocks = get_layer_expert_units(model, ep_plan, fsdp_modules)

    set_modules_to_prefetch(model, fsdp_modules, fsdp_plan, layer_expert_units, layer_moe_blocks)
    if fsdp_plan.num_to_backward_prefetch == 0:
        # FSDP2's default backward prefetch is enabled when no explicit list
        # is set, so a configured depth of zero needs an explicit override.
        disable_default_backward_prefetch(model.modules())

    # Order each layer's forward-prefetch AG after the previous layer's
    # trailing a2a. Anchored on prefetch-holding layers, registered from the
    # first experts layer's next layer on: earlier entries have no in-flight
    # a2a to wait on.
    if fsdp_plan.prefetch_wait_a2a and fsdp_plan.num_to_forward_prefetch > 0 and layer_expert_units:
        order = get_prefetch_order(model, fsdp_modules, fsdp_plan)
        first_expert_idx = next(i for i, layer in enumerate(order) if layer in layer_expert_units)
        for layer in order[first_expert_idx + 1 :]:
            layer.register_forward_pre_hook(_wait_expert_a2a_hook, prepend=True)
    return model


def _move_unmanaged_ignored_modules(model, ignored_modules, device):
    """Move ignored state without overriding nested FSDP or host-storage ownership."""
    ignored_submodules = {
        descendant
        for name, module in model.named_modules()
        if name in ignored_modules
        for descendant in module.modules()
    }
    protected_submodules = {
        descendant
        for module in ignored_submodules
        if isinstance(module, FSDPModule) or callable(getattr(module, "pre_fsdp_hook", None))
        for descendant in module.modules()
    }
    protected_tensors = {
        tensor
        for module in protected_submodules
        for tensor in (*module.parameters(recurse=False), *module.buffers(recurse=False))
    }
    moved_tensors = set()
    for module in model.modules():
        if module not in ignored_submodules or module in protected_submodules:
            continue
        for tensor in (*module.parameters(recurse=False), *module.buffers(recurse=False)):
            if tensor.is_meta or tensor in protected_tensors or tensor in moved_tensors:
                continue
            tensor.data = tensor.data.to(device)
            moved_tensors.add(tensor)


def _wait_expert_a2a_hook(module, args):
    wait_last_expert_a2a_before_prefetch(module)


def get_layer_expert_units(
    model: torch.nn.Module,
    ep_plan: EPPlanConfig,
    fsdp_modules: dict[torch.nn.Module, Any],
) -> tuple[
    dict[torch.nn.Module, list[torch.nn.Module]],
    dict[torch.nn.Module, torch.nn.Module],
]:
    """Map each layer to its efsdp units and the owning MoE block (the parent
    of ``experts``, whose forward input is the dispatch a2a input). The layer
    key is the nearest enclosing dense FSDP unit. A ``ModuleList`` is wrapped
    per expert, other modules as a whole, matching
    ``expert_fully_shard_parallel``.
    """
    # Local import breaks the pylint-reported cycle with expert_fully_shard_parallel.
    from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import get_efsdp_modules  # pylint: disable=cyclic-import

    layer_expert_units: dict[torch.nn.Module, list[torch.nn.Module]] = {}
    layer_moe_blocks: dict[torch.nn.Module, torch.nn.Module] = {}
    parent_of = {child: parent for parent in model.modules() for child in parent.children()}
    for experts in get_efsdp_modules(model, ep_plan):
        # fsdp_modules is pre-order, so reversed finds the nearest enclosing
        # dense FSDP unit (the layer, not an ancestor batching all layers).
        layer = find_hook_module(experts, list(reversed(fsdp_modules)))
        if layer is None:
            continue
        # A whole-wrapped container is one unit; a ModuleList expands to its
        # per-expert wrapped units.
        units = [experts] if not isinstance(experts, torch.nn.ModuleList) else list(experts)
        layer_units = layer_expert_units.setdefault(layer, [])
        for unit in units:
            if all(unit is not existing for existing in layer_units):
                layer_units.append(unit)
        moe_block = parent_of.get(experts)
        if moe_block is not None:
            layer_moe_blocks.setdefault(layer, moe_block)
    # get_efsdp_modules matches by pattern only. With efsdp configured but not
    # applied (explicit selection + expert_parallel_size <= 1, which config
    # validation only warns about), matched units carry no FSDP state and the
    # prefetch setup below would crash. Keep wrapped units only.
    for layer in list(layer_expert_units):
        wrapped = [u for u in layer_expert_units[layer] if _get_module_state(u) is not None]
        if wrapped:
            layer_expert_units[layer] = wrapped
        else:
            del layer_expert_units[layer]
            layer_moe_blocks.pop(layer, None)
    return layer_expert_units, layer_moe_blocks


def _register_backward_prefetch_on_moe_input(
    moe_module: torch.nn.Module,
    targets: List[torch.nn.Module],
    wait_a2a: bool = False,
) -> None:
    """Attach MoE backward param prefetch to the MoE block's forward input
    grad hook: that grad is ready right after the reverse dispatch a2a, so
    the prefetch AG overlaps the layer's pure-compute backward. With
    wait_a2a, the AG streams first wait for the reverse-a2a event.
    """
    # Fail fast: a state exposing neither param-group attribute means a torch
    # version incompatible with this feature; raising beats a silent no-op.
    for target in targets:
        state = _get_module_state(target)
        if state is None or (
            getattr(state, "_fsdp_param_groups", None) is None and getattr(state, "_fsdp_param_group", None) is None
        ):
            raise RuntimeError(
                "backward_prefetch_on_moe_input requires a torch version whose FSDP state "
                "exposes _fsdp_param_group(s); none found on a prefetch target."
            )

    def _prefetch_on_grad(grad: torch.Tensor) -> torch.Tensor:
        # Record the event covering the reverse a2a for the AG streams.
        # No deadlock: the reverse a2a depends on no target AG.
        a2a_event = torch.accelerator.current_stream().record_event() if wait_a2a else None
        waited_streams: set = set()
        for target in targets:
            state = _get_module_state(target)
            if state is None:
                continue
            if a2a_event is not None:
                # Units may share one all_gather_stream; dedup by identity.
                ag_stream = getattr(getattr(state, "_comm_ctx", None), "all_gather_stream", None)
                if ag_stream is not None and id(ag_stream) not in waited_streams:
                    waited_streams.add(id(ag_stream))
                    ag_stream.wait_event(a2a_event)
            # Cover both single (_fsdp_param_group) and list (_fsdp_param_groups)
            # FSDP states across torch versions; unshard is idempotent.
            param_groups = getattr(state, "_fsdp_param_groups", None)
            if param_groups is None:
                param_group = getattr(state, "_fsdp_param_group", None)
                param_groups = () if param_group is None else (param_group,)
            for target_param_group in reversed(param_groups):
                with target_param_group.use_training_state(TrainingState.PRE_BACKWARD):
                    target_param_group.unshard()
        return grad

    def _record_moe_input(module, args, kwargs):
        # Fresh tensor hooks per forward; freed with the tensor.
        if not torch.is_grad_enabled():
            return None
        for t in (*args, *kwargs.values()):
            if isinstance(t, torch.Tensor) and t.requires_grad:
                t.register_hook(_prefetch_on_grad)
        return None

    moe_module.register_forward_pre_hook(_record_moe_input, with_kwargs=True)


def get_prefetch_order(
    model: torch.nn.Module,
    fsdp_modules: dict[torch.nn.Module, Any],
    fsdp_plan: FSDPPlanConfig,
) -> list[torch.nn.Module]:
    """Layers holding prefetch lists, in model pre-order (or the explicit
    ``prefetch_modules`` order when configured).
    """
    wrapped_modules_in_order = list(fsdp_modules)
    if fsdp_plan.prefetch_modules is not None:
        named_modules = dict(model.named_modules())
        wrapped_modules_in_order = list(
            dict.fromkeys(
                module
                for pattern in fsdp_plan.prefetch_modules
                for name, module in named_modules.items()
                if module in fsdp_modules and module_name_match(pattern, name)
            )
        )
    if any(getattr(unit, "_kimi_preserve_gate_precision", False) for unit in wrapped_modules_in_order):
        parents = {child: parent for parent in model.modules() for child in parent.children()
                   if getattr(child, "_kimi_preserve_gate_precision", False)}
        wrapped_modules_in_order = [parents.get(unit, unit) for unit in wrapped_modules_in_order]
    return wrapped_modules_in_order


def set_modules_to_prefetch(
    model: torch.nn.Module,
    fsdp_modules: dict[torch.nn.Module, Any],
    fsdp_plan: FSDPPlanConfig,
    layer_expert_units: Optional[dict[torch.nn.Module, list[torch.nn.Module]]] = None,
    layer_moe_blocks: Optional[dict[torch.nn.Module, torch.nn.Module]] = None,
):
    """Configure forward and backward prefetch lists.

    A layer prefetches its own efsdp units plus those of the next
    ``num_to_forward_prefetch`` layers, issuing expert all-gathers at the
    layer's forward entry.
    """
    wrapped_modules_in_order = get_prefetch_order(model, fsdp_modules, fsdp_plan)

    if fsdp_plan.num_to_forward_prefetch > 0:
        for i, layer in enumerate(wrapped_modules_in_order):
            modules_to_prefetch = list(layer_expert_units.get(layer, []))
            next_layers = wrapped_modules_in_order[i + 1 : i + 1 + fsdp_plan.num_to_forward_prefetch]
            for next_layer in next_layers:
                modules_to_prefetch.append(next_layer)
                modules_to_prefetch.extend(layer_expert_units.get(next_layer, []))
            layer.set_modules_to_forward_prefetch(modules_to_prefetch)

    if fsdp_plan.num_to_backward_prefetch > 0:
        num_prefetch = fsdp_plan.num_to_backward_prefetch
        # MoE layers prefetch via the MoE input grad hook (see
        # _register_backward_prefetch_on_moe_input) and get [self] placeholders
        # to block the native default prefetch; expert units are handled the
        # same way in expert_fully_shard_parallel.
        backward_on_moe_input = fsdp_plan.backward_prefetch_on_moe_input and bool(layer_expert_units)
        layer_moe_blocks = layer_moe_blocks or {}
        rev_wrapped_modules_in_order = list(reversed(wrapped_modules_in_order))
        for i, layer in enumerate(rev_wrapped_modules_in_order):
            if backward_on_moe_input and layer in layer_expert_units:
                layer.set_modules_to_backward_prefetch([layer])
                continue
            layers_to_prefetch = rev_wrapped_modules_in_order[i + 1 : i + 1 + num_prefetch]
            if layers_to_prefetch:
                layer.set_modules_to_backward_prefetch(layers_to_prefetch)
        if backward_on_moe_input:
            # Expert units are not in wrapped_modules_in_order; targets =
            # flattened [experts, dense] of the k layers below.
            for i, layer in enumerate(wrapped_modules_in_order):
                if layer not in layer_expert_units:
                    continue
                moe_block = layer_moe_blocks.get(layer)
                if moe_block is None:
                    continue
                targets: list[torch.nn.Module] = []
                for prev_layer in reversed(wrapped_modules_in_order[max(0, i - num_prefetch) : i]):
                    targets.extend(layer_expert_units.get(prev_layer, []))
                    targets.append(prev_layer)
                if targets:
                    _register_backward_prefetch_on_moe_input(
                        moe_block,
                        targets,
                        wait_a2a=fsdp_plan.prefetch_wait_a2a,
                    )


def get_mixprecision_policy(fsdp_plan: FSDPPlanConfig):
    """Construct the MixedPrecisionPolicy object."""
    param_dtype = get_dtype(fsdp_plan.param_dtype) if fsdp_plan.param_dtype else None
    reduce_dtype = get_dtype(fsdp_plan.reduce_dtype) if fsdp_plan.reduce_dtype else None
    output_dtype = get_dtype(fsdp_plan.output_dtype) if fsdp_plan.output_dtype else None

    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        cast_forward_inputs=fsdp_plan.cast_forward_inputs,
    )


def get_offload_policy(fsdp_plan: FSDPPlanConfig):
    """Construct the CPUOffloadPolicy object, or OffloadPolicy() to disable offloading."""
    if not fsdp_plan.cpu_offload:
        return OffloadPolicy()
    return CPUOffloadPolicy(pin_memory=fsdp_plan.pin_memory)


def get_fsdp_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig, ignored_modules: Set[str]) -> dict[Any, Any]:
    fsdp_modules = {}
    for name, module in model.named_modules():
        for pattern, plan in fsdp_plan.apply_modules.items():
            if module_name_match(pattern, name) and name not in ignored_modules:
                print_rank(logger.debug, f'[FSDP2]: Apply fsdp2 to module <{name}>')
                if module not in fsdp_modules:
                    fsdp_modules[module] = {}
                fsdp_modules.get(module).update(plan)
    if len(fsdp_modules) == 0:
        raise RuntimeError(f'[FSDP2] No module named {fsdp_plan.apply_modules.keys()}.')
    return fsdp_modules


def _post_order_traverse(model: torch.nn.Module, parent_path: str = ""):
    """
    Perform post-order traversal of model submodules.

    Post-order traversal ensures child modules are visited before their parents,
    which is important for FSDP to properly handle nested modules.

    Args:
        model: The model to traverse.
        parent_path: The path to the current module in the hierarchy.

    Yields:
        Tuple of (module_path, module) for each module in the model.
    """
    for name, child in model.named_children():
        child_path = f"{parent_path}.{name}" if parent_path else name
        yield from _post_order_traverse(child, child_path)
    yield parent_path, model


def get_fsdp_hook_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig) -> List[Any]:
    fsdp_hook_modules = []
    if fsdp_plan.apply_modules is None:
        return fsdp_hook_modules

    # Traverse all modules in the model
    if fsdp_plan.hook_modules:
        for name, module in _post_order_traverse(model):
            # Check if module matches any pattern in the FSDP plan
            for pattern in fsdp_plan.hook_modules:
                if module_name_match(pattern, name):
                    print_rank(logger.debug, f'[FSDP2]: Apply fsdp2 hook to hook_module <{name}>')
                    fsdp_hook_modules.append(module)
        # Ensure at least one module matches the FSDP plan
        if len(fsdp_hook_modules) == 0:
            raise RuntimeError(f'[FSDP2] No module named {fsdp_plan.hook_modules}.')

    return fsdp_hook_modules


def find_hook_module(
    target_module: torch.nn.Module, hook_module_list: List[torch.nn.Module]
) -> Optional[torch.nn.Module]:
    for hook_module in hook_module_list:
        for _, sub_mod in hook_module.named_modules():
            if sub_mod is target_module:
                return hook_module
    return None


def get_ignored_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig):
    ignored_modules = set()
    ignored_params = set()
    for name, module in model.named_modules():
        for pattern in fsdp_plan.ignored_modules:
            if module_name_match(pattern, name):
                print_rank(logger.debug, f'[FSDP2]: Ignored module to apply fsdp2 <{name}>')
                ignored_modules.add(name)
                ignored_params.update(list(module.parameters(recurse=True)))
    return ignored_modules, ignored_params

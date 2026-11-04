# Copyright (c) 2024 Meta Inc.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
from typing import Set, List, Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from fsdp_turbo.fsdp_turbo_config import FSDPPlanConfig
from fsdp_turbo.distributed.fine_grained_fully_shard import get_fsdp_strategy
from fsdp_turbo.distributed.fully_shard_parallel.prefetch import disable_default_backward_prefetch
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match
from fsdp_turbo.utils.dtype import get_dtype


logger = logging.getLogger(__name__)


def fully_shard_parallel_modules(model: torch.nn.Module, fsdp_mesh: DeviceMesh, fsdp_plan: FSDPPlanConfig):
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

    for module, plan in fsdp_modules.items():
        module_config = config.copy()
        module_config.update(plan)
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
    fully_shard_fn(model, **config)
    set_modules_to_prefetch(model, fsdp_modules, fsdp_plan)
    if fsdp_plan.num_to_backward_prefetch == 0:
        # FSDP2's default backward prefetch is enabled when no explicit list
        # is set, so a configured depth of zero needs an explicit override.
        disable_default_backward_prefetch(model.modules())
    return model


def set_modules_to_prefetch(
    model: torch.nn.Module, fsdp_modules: dict[torch.nn.Module, Any], fsdp_plan: FSDPPlanConfig
):
    """Configure prefetching in explicit execution order or registration order."""
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

    if fsdp_plan.num_to_forward_prefetch > 0:
        for i, layer in enumerate(wrapped_modules_in_order):
            layer.set_modules_to_forward_prefetch(
                wrapped_modules_in_order[i + 1 : i + 1 + fsdp_plan.num_to_forward_prefetch]
            )

    if fsdp_plan.num_to_backward_prefetch > 0:
        rev_wrapped_modules_in_order = list(reversed(wrapped_modules_in_order))
        for i, layer in enumerate(rev_wrapped_modules_in_order):
            layer.set_modules_to_backward_prefetch(
                rev_wrapped_modules_in_order[i + 1 : i + 1 + fsdp_plan.num_to_backward_prefetch]
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

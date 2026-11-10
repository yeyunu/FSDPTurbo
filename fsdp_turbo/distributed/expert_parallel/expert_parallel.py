# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import types
from functools import partial
from typing import Callable, Optional
import os

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard, DTensor, Replicate, distribute_module
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from fsdp_turbo.distributed.expert_parallel.dispatcher import all2all_experts_forward_fn
from fsdp_turbo.distributed.expert_parallel.ep_balance import initialize_ep_balance
from fsdp_turbo.distributed.expert_parallel.dispatcher_mc2 import mc2_experts_forward_fn
from fsdp_turbo.distributed.expert_parallel.domino_dispatcher import get_domino_experts_forward_fn
from fsdp_turbo.distributed.expert_parallel.dispatcher_chunkmoe import chunkmoe_experts_forward_fn
from fsdp_turbo.distributed.expert_parallel.dispatcher_all_gather import all_gather_experts_forward_fn
from fsdp_turbo.distributed.expert_parallel.utils import (
    DISPATCH_SPLIT_POLICY_ATTR,
    DispatchSplitPolicy,
    fixed_router_for_debug,
    normalize_expert_args,
    record_last_expert_a2a_event,
)
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match

logger = logging.getLogger(__name__)

SKIP_MOE_PAD_TOKENS = os.environ.get('SKIP_MOE_PAD_TOKENS', '0') == '1'


def expert_parallelize_modules(
    modules: torch.nn.Module,
    ep_mesh: DeviceMesh,
    plan: EPPlanConfig,
    edp_mesh: Optional[DeviceMesh] = None,
    dispatch_split_policy: Optional[DispatchSplitPolicy] = None,
):
    ep_modules = get_ep_modules(modules, plan)

    ep_group = ep_mesh.get_group()
    ep_rank = torch.distributed.get_rank(ep_group)
    ep_size = torch.distributed.get_world_size(ep_group)

    for module in ep_modules:
        module.num_global_experts = len(module) if not hasattr(module, 'num_experts') else module.num_experts
        if module.num_global_experts % ep_size != 0:
            raise AssertionError(
                f'Number of experts({module.num_global_experts}) is not divisible by ep size({ep_size}).'
            )
        module.num_local_experts = module.num_global_experts // ep_size
        local_expert_indices_offset = ep_rank * module.num_local_experts
        module.local_expert_indices = [local_expert_indices_offset + i for i in range(module.num_local_experts)]
        module.expert_ids_per_ep_rank = torch.arange(
            module.num_global_experts,
            dtype=torch.int32,
            device=ep_mesh.device_type,
        ).remainder(module.num_local_experts)

        # Per-container a2a split-cache policy; left unset keeps the original paths.
        if dispatch_split_policy is not None:
            setattr(module, DISPATCH_SPLIT_POLICY_ATTR, dispatch_split_policy)

        if plan.enable_ep_balance:
            initialize_ep_balance(module, ep_group, plan.max_dup_experts_num)

    if plan.dispatcher == "moonep":
        from fsdp_turbo.distributed.expert_parallel.moonep import parallelize_moonep_model

        parallelize_moonep_model(modules, ep_modules, ep_mesh, plan, edp_mesh=edp_mesh)
        return modules

    for module in ep_modules:
        distribute_experts_module(module, ep_mesh)
        experts_forward_fn = get_experts_forward_fn(ep_group=ep_group, ep_plan=plan, dispatcher=plan.dispatcher)
        module.forward = types.MethodType(experts_forward_fn, module)
    return modules


def get_ep_modules(modules: torch.nn.Module, plan: EPPlanConfig):
    ep_modules = []
    for plan_name in plan.apply_modules:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                print_rank(logger.debug, f'[Expert Parallel]: Apply ep to module <{name}>')
                if module not in ep_modules:
                    ep_modules.append(module)
    if len(ep_modules) == 0:
        raise RuntimeError(f'[Expert Parallel] No module named {plan} or not be ModuleList')
    return ep_modules


def prepare_distribute_input_fn(module, inputs, device_mesh):
    inputs = list(inputs)
    for idx, input_tensor in enumerate(inputs):
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, (Replicate(),), run_check=False)
            inputs[idx] = input_tensor
    return (*inputs,)


def prepare_distribute_output_fn(module, outputs, device_mesh):
    return outputs.to_local()


def distribute_expert_weight(module_name, module, ep_mesh):
    placements = [Shard(0)]
    for name, param in module.named_parameters(recurse=False):
        local_shape, global_offset = compute_local_shape_and_global_offset(param.shape, ep_mesh, placements)
        slices = [slice(offset, offset + size) for offset, size in zip(global_offset, local_shape)]
        dist_param = torch.nn.Parameter(DTensor.from_local(param[tuple(slices)].detach().clone(), ep_mesh, placements))
        module.register_parameter(name, dist_param)

    for name, children_module in module.named_children():
        distribute_expert_weight(name, children_module, ep_mesh)


def distribute_experts_module(module: torch.nn.Module, ep_mesh: DeviceMesh):
    return distribute_module(
        module=module,
        device_mesh=ep_mesh,
        partition_fn=distribute_expert_weight,
    )


def get_experts_forward_fn(
    ep_group: torch.distributed.ProcessGroup,
    ep_plan: EPPlanConfig,
    dispatcher: str = "all2all",
):
    def experts_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        dispatch_fn: Callable = None,
        use_swiglu: bool = True,
        **kwargs,
    ):
        # Ensure correct parameter order and dimensions
        top_k_index, top_k_weights = normalize_expert_args(top_k_index, top_k_weights)
        hidden_states_shape = hidden_states.shape
        hidden_dim = getattr(self, "hidden_dim", None)
        hidden_dim = hidden_states.shape[-1] if hidden_dim is None else hidden_dim
        hidden_states = hidden_states.reshape(-1, hidden_dim)

        if ep_plan.fixed_router:
            top_k_index, top_k_weights = fixed_router_for_debug(top_k_index, top_k_weights, self.num_global_experts)

        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj
        act_limit = self.limit if hasattr(self, 'limit') else None
        act_alpha = self.alpha if hasattr(self, 'alpha') else None
        # the definition of situ of some models (e.g. kimi) is different from default, which should redefine this function.
        act_fn = None if use_swiglu else self.act_fn

        quant_config = getattr(self, '_quant_config', None) or getattr(self, 'config', None)
        if quant_config is not None and not hasattr(quant_config, 'get_key_dtype'):
            quant_config = None

        seq_mask = kwargs.get('seq_mask', None)
        if SKIP_MOE_PAD_TOKENS:
            if seq_mask is None:
                raise ValueError("seq_mask must be provided when skip_moe_pad_tokens is enabled.")
            seq_mask = seq_mask.view(-1)
            pad_indices = seq_mask == 0
            nonpad_indices = ~pad_indices
            original_hidden_states = hidden_states
            hidden_states = original_hidden_states[nonpad_indices, :]
            pad_hidden_states = original_hidden_states[pad_indices, :]
            top_k_index = top_k_index[nonpad_indices, :]
            top_k_weights = top_k_weights[nonpad_indices, :]

        hidden_states = dispatch_fn(
            hidden_states,
            top_k_index,
            top_k_weights,
            (gate_up_proj, down_proj),
            self.num_global_experts,
            self.expert_ids_per_ep_rank,
            ep_group,
            fused=ep_plan.use_fused_ops,
            act_fn=act_fn,
            act_limit=act_limit,
            act_alpha=act_alpha,
            ep_balance_strategy=getattr(self, "ep_balance_strategy", None),
            quant_config=quant_config,
            expert_module=self,
            ep_plan=ep_plan,
        )

        if SKIP_MOE_PAD_TOKENS:
            result = torch.zeros_like(original_hidden_states)
            result[nonpad_indices, :] = hidden_states
            result[pad_indices, :] = pad_hidden_states
            hidden_states = result
        return hidden_states.view(*hidden_states_shape)

    def get_dispatch_experts_forward_fn(dispatch_fn):
        return partial(experts_forward, dispatch_fn=dispatch_fn, use_swiglu=ep_plan.use_swiglu)

    forward_fn = None
    if isinstance(dispatcher, Callable):
        forward_fn = partial(dispatcher, ep_group)  # Note: keep original interface, should deprecate
    elif isinstance(dispatcher, str) and dispatcher.startswith('custom_'):
        method_name = dispatcher.removeprefix('custom_')

        def custom_ep_forward(self, hidden_states, top_k_index, top_k_weights):
            method = getattr(self, method_name, None)
            if not callable(method):
                raise RuntimeError(
                    f'EP dispatcher requires {type(self).__name__}.{method_name} to be a callable method.'
                )
            out = method(
                hidden_states,
                top_k_index,
                top_k_weights,
                ep_group=ep_group,
                fixed_router=ep_plan.fixed_router,
            )
            record_last_expert_a2a_event(self, torch.accelerator.current_stream().record_event())
            return out

        forward_fn = custom_ep_forward
    elif isinstance(dispatcher, str) and dispatcher == 'domino':
        forward_fn = get_domino_experts_forward_fn(ep_group, fixed_router=ep_plan.fixed_router)
    elif isinstance(dispatcher, str):
        dispatch_fn = None
        if dispatcher == 'all2all':
            dispatch_fn = all2all_experts_forward_fn
        elif dispatcher == 'mc2':
            dispatch_fn = mc2_experts_forward_fn
        elif dispatcher == 'chunkmoe':
            dispatch_fn = chunkmoe_experts_forward_fn
        elif dispatcher == 'allgather':
            dispatch_fn = all_gather_experts_forward_fn
        if dispatch_fn is not None:
            forward_fn = get_dispatch_experts_forward_fn(dispatch_fn)

    if forward_fn is None:
        raise RuntimeError(f'Unsupported dispatcher {dispatcher}.')

    return forward_fn

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Attach Kimi router-debug handles for the MoonEP dispatcher."""


def attach_gate_debug(modules, ep_modules, ep_group, plan) -> None:
    """Give each Kimi MoE gate the EP group, layer index, and router-debug config."""
    module_names = {id(module): name for name, module in modules.named_modules()}
    for layer_index, module in enumerate(ep_modules):
        name = module_names[id(module)]
        parent = modules.get_submodule(name.rpartition(".")[0])
        gate = getattr(parent, "gate", None)
        if gate is None:
            continue
        object.__setattr__(gate, "_expert_parallel_group", ep_group)
        object.__setattr__(gate, "_router_debug_layer", layer_index)
        object.__setattr__(gate, "_router_debug", plan.router_debug)

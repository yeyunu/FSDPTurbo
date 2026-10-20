# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Vocab-FSDP (vfsdp): apply ``fully_shard`` on top of the vocab-parallel (vp)
partition — the direct analogue of
``expert_parallel/expert_fully_shard_parallel.py``.

The embedding weight arriving here is already a DTensor with ``Shard(0)`` on
the ``vp`` mesh, which FSDP2 composes as the "TP" spec: with
``shard_placement_fn = Shard(0)`` torch turns the FSDP placement into a
strided shard so the vocab dim is split a SECOND time across the vfsdp group.
Per-rank storage is ``[vocab/(vp*vfsdp), dim]`` and the forward unshard only
reassembles this rank's contiguous ``[vocab/vp, dim]`` window along the vfsdp
group — the full table is NEVER materialized on any rank, which is what makes
the ~51B qwen4_exp ngram table trainable.
"""

import logging

import torch
from torch.distributed.tensor import Shard

from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import (
    set_gradient_divide_factor,
)
from fsdp_turbo.distributed.fine_grained_fully_shard import get_fsdp_strategy
from fsdp_turbo.distributed.fully_shard_parallel.fully_shard_parallel import (
    find_hook_module,
    get_fsdp_hook_modules,
    get_mixprecision_policy,
    get_offload_policy,
)
from fsdp_turbo.fsdp_turbo_config import FSDPPlanConfig, VPPlanConfig
from fsdp_turbo.utils.dtype import get_dtype
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match

logger = logging.getLogger(__name__)


def is_vfsdp_enabled(vfsdp_mesh) -> bool:
    """Return whether the vocabulary HSDP mesh spans multiple devices."""
    return int(vfsdp_mesh.mesh.numel()) > 1


def get_vfsdp_modules(modules: torch.nn.Module, plan: VPPlanConfig):
    vfsdp_modules = []
    for plan_name in plan.apply_vfsdp_modules:
        matched = []
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                print_rank(logger.debug, f"[Vocabulary Fully Shard]: Apply vfsdp to module <{name}>")
                matched.append(module)
        if not matched:
            raise RuntimeError(f"[Vocabulary Fully Shard] No module matched pattern <{plan_name}>.")
        vfsdp_modules.extend(matched)
    return list(dict.fromkeys(vfsdp_modules))


def vocab_fully_shard_modules(
    model: torch.nn.Module,
    vfsdp_mesh,
    vp_plan: VPPlanConfig,
    fsdp_plan: FSDPPlanConfig,
) -> torch.nn.Module:
    vfsdp_modules = get_vfsdp_modules(model, vp_plan)

    if not is_vfsdp_enabled(vfsdp_mesh):
        # ---- V-FSDP singleton (no FSDP sharding) branch ----
        # The pure-VP parameters stay outside ancestor FSDP units because
        # validate_vp_config added the patterns to fsdp_plan.ignored_modules;
        # mixed-precision casting and gradient division are applied in the
        # return all-to-all instead.
        output_dtype_name = fsdp_plan.output_dtype or fsdp_plan.param_dtype
        output_dtype = get_dtype(output_dtype_name) if output_dtype_name else None
        for embedding in vfsdp_modules:
            embedding._vp_gradient_divide_factor = float(vp_plan.gradient_divide_factor)
            embedding._vp_output_dtype = output_dtype

        print_rank(
            logger.info,
            "[Vocabulary Fully Shard]: Skip singleton V-FSDP; keep VP weights "
            "sharded, cast lookup outputs to the mixed-precision dtype, and "
            "apply gradient division in the return all-to-all.",
        )
        return model

    # ---- V-FSDP active (multi-device) branch ----
    hook_modules = get_fsdp_hook_modules(model, fsdp_plan)
    offload_policy = get_offload_policy(fsdp_plan)

    config = {
        "mesh": vfsdp_mesh,
        "mp_policy": get_mixprecision_policy(fsdp_plan),
        # Keep both VP and VFSDP on the vocabulary dimension. FSDP2 composes
        # this with the existing VP Shard(0) placement without gathering the
        # global embedding table.
        "shard_placement_fn": lambda _param: Shard(0),
        "offload_policy": offload_policy,
    }

    strategy = get_fsdp_strategy(fsdp_plan.fsdp_implementation)
    fully_shard_fn = strategy.get_unified_fully_shard_fn()
    for embedding in vfsdp_modules:
        hook_module = find_hook_module(embedding, hook_modules)
        if hook_module is None:
            fully_shard_fn(embedding, **config)
        else:
            fully_shard_fn(embedding, hook_module=hook_module, **config)
        set_gradient_divide_factor(embedding, vp_plan.gradient_divide_factor)
    return model

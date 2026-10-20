# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Vocabulary-parallel (VP) sharding for huge embedding tables.

The VP analogue of ``expert_parallel/expert_parallel.py``: a ``[vocab, dim]``
embedding weight is ``distribute_tensor``-sharded along the vocab dim
(dim 0, ``Shard(0)`` on the ``vp`` mesh) exactly like MoE experts are split
along the expert dim.  The vp group doubles as a data-parallel group (ranks
embed DIFFERENT tokens), so the module's ``forward`` is replaced with the
all-to-all dispatcher in ``vp_dispatcher``.
"""

import logging
import types

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from fsdp_turbo.fsdp_turbo_config import VPPlanConfig
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match

from .vp_dispatcher import vocab_parallel_embedding_forward

logger = logging.getLogger(__name__)


def get_vp_modules(model: torch.nn.Module, plan: VPPlanConfig):
    modules = []
    seen = set()
    for pattern in plan.apply_modules:
        matched = []
        for name, module in model.named_modules():
            if module_name_match(pattern, name):
                if not isinstance(module, torch.nn.Embedding):
                    raise TypeError(f"[Vocabulary Parallel] Module <{name}> must be nn.Embedding.")
                matched.append((name, module))
        if not matched:
            raise RuntimeError(f"[Vocabulary Parallel] No module matched pattern <{pattern}>.")
        for name, module in matched:
            if id(module) in seen:
                continue
            seen.add(id(module))
            print_rank(logger.info, f"[Vocabulary Parallel]: Apply VP to module <{name}>")
            modules.append(module)
    return modules


def _get_forward_fn(vp_group):
    def forward(module, input_ids):
        return vocab_parallel_embedding_forward(module, input_ids, vp_group)

    return forward


def vocab_parallelize_modules(
    model: torch.nn.Module,
    vp_mesh: DeviceMesh,
    plan: VPPlanConfig,
) -> torch.nn.Module:
    vp_group = vp_mesh.get_group()
    vp_rank = torch.distributed.get_rank(vp_group)
    vp_size = torch.distributed.get_world_size(vp_group)

    for module in get_vp_modules(model, plan):
        if isinstance(module.weight, DTensor):
            raise RuntimeError("[Vocabulary Parallel] Embedding weight is already distributed.")
        if module.max_norm is not None:
            raise ValueError("[Vocabulary Parallel] nn.Embedding.max_norm is not supported.")
        if module.scale_grad_by_freq:
            raise ValueError("[Vocabulary Parallel] nn.Embedding.scale_grad_by_freq is not supported.")
        if module.sparse:
            raise ValueError("[Vocabulary Parallel] Sparse embedding gradients are not supported.")
        if module.num_embeddings % vp_size != 0:
            raise ValueError(
                f"num_embeddings ({module.num_embeddings}) must be divisible by vocab parallel size ({vp_size})."
            )

        weight = module.weight.data
        if weight.device.type != vp_mesh.device_type and not weight.is_meta:
            # Host-resident table (e.g. the PLE n-gram table shared via
            # /dev/shm): move only this rank's shard to the accelerator.
            # ``distribute_tensor`` would stage the FULL table on the device
            # (it does ``tensor.to(mesh_device)``), which OOMs for ~51B tables.
            local_weight = weight.chunk(vp_size, dim=0)[vp_rank].to(vp_mesh.device_type)
            distributed_weight = DTensor.from_local(
                local_weight,
                vp_mesh,
                [Shard(0)],
                run_check=False,
                shape=torch.Size((module.num_embeddings, module.embedding_dim)),
                stride=(module.embedding_dim, 1),
            )
        else:
            distributed_weight = distribute_tensor(weight, vp_mesh, [Shard(0)])
        module.register_parameter(
            "weight",
            torch.nn.Parameter(distributed_weight, requires_grad=module.weight.requires_grad),
        )
        module._vp_vocab_start_index = vp_rank * (module.num_embeddings // vp_size)
        module._vp_vocab_end_index = (vp_rank + 1) * (module.num_embeddings // vp_size)
        # V-FSDP normally applies the global gradient divisor during its
        # reduce-scatter. The pure-VP path overrides this when V-FSDP is a
        # singleton and therefore skipped.
        module._vp_gradient_divide_factor = 1.0
        module._vp_output_dtype = None
        module.forward = types.MethodType(_get_forward_fn(vp_group), module)

        local_shape = tuple(module.weight.to_local().shape)
        print_rank(
            logger.info,
            f"[Vocabulary Parallel]: rank={vp_rank}/{vp_size}, local weight shape={local_shape}, "
            f"vocab range=[{module._vp_vocab_start_index}, {module._vp_vocab_end_index})",
        )
    return model

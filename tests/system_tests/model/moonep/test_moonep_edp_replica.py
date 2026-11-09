# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""D-2 probe: after replica sync, the two expert copies must hold the same home_grad.

world=4, EP=2, efsdp=1 → edp=2. This is not a 7.8 upgrade and must not be
written up as supporting edp>1. It only checks that sync_replica_gradients
made the two copies agree. Launch with run_test_moonep_edp2.sh.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from fsdp_turbo.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig, MoonEPConfig
from fsdp_turbo.utils.device import get_accelerator_name, set_accelerator_compatible
from fsdp_turbo.utils.log import set_log_level


def _load_st():
    path = Path(__file__).with_name("test_moonep.py")
    spec = importlib.util.spec_from_file_location("moonep_st_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_st = _load_st()
EXPERTS_PER_RANK = _st.EXPERTS_PER_RANK
FFN = _st.FFN
GRADIENT_TOLERANCE = _st.GRADIENT_TOLERANCE
HIDDEN = _st.HIDDEN
KNOBS = _st.KNOBS
SEED = _st.SEED
STEPS = _st.STEPS
TOKENS_PER_RANK = _st.TOKENS_PER_RANK
TOP_K = _st.TOP_K
ExpertsHolder = _st.ExpertsHolder
FusedExperts = _st.FusedExperts
_to_local = _st._to_local
build_routing = _st.build_routing
relative_error = _st.relative_error


def main() -> int:
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        if name not in os.environ:
            raise RuntimeError(f"This script must be launched with torchrun; {name} is missing.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    backend = set_accelerator_compatible()
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    set_log_level(os.environ.get("LOG_LEVEL", "INFO"))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError(f"D-2 probe expects world=4; got {world_size}.")

    device_type = get_accelerator_name()
    device = torch.device(device_type, local_rank)
    # (edp=2, efsdp=1, ep=2): ranks 0/1 are one EP group, 2/3 the other copy.
    mesh = init_device_mesh(device_type, (2, 1, 2), mesh_dim_names=("edp", "efsdp", "ep"))
    ep_mesh = mesh["ep"]
    edp_mesh = mesh["edp"]
    ep_size = ep_mesh.size()
    edp_size = edp_mesh.size()
    ep_rank = dist.get_rank(group=ep_mesh.get_group())
    edp_rank = dist.get_rank(group=edp_mesh.get_group())
    num_experts = EXPERTS_PER_RANK * ep_size

    plan = EPPlanConfig(
        apply_modules=["experts"],
        dispatcher="moonep",
        gradient_divide_factor=float(world_size),
        moonep_config=MoonEPConfig(
            tokens_per_rank=TOKENS_PER_RANK,
            top_k=TOP_K,
            **KNOBS,
        ),
    )

    torch.manual_seed(SEED)
    holder = ExpertsHolder(FusedExperts(num_experts, HIDDEN, FFN, device))
    for parameter in holder.parameters():
        dist.broadcast(parameter.data, src=0)

    runtime = None
    failures: list[str] = []
    try:
        expert_parallelize_modules(holder, ep_mesh, plan, edp_mesh=edp_mesh)
        runtime = holder._moonep_runtime
        if rank == 0:
            group = runtime.edp_group
            group_size = 0 if group is None else dist.get_world_size(group=group)
            print(
                f"D-2 probe: world={world_size} ep={ep_size} efsdp=1 edp={edp_size} "
                f"edp_group size={group_size} E={num_experts} "
                f"(replica-consistency check only; not a support claim)",
                flush=True,
            )
        if runtime.edp_group is None or dist.get_world_size(group=runtime.edp_group) != 2:
            raise RuntimeError("D-2 probe did not get an edp group of size 2.")

        generator = torch.Generator(device=device).manual_seed(SEED + rank)
        for step in range(STEPS):
            hidden = (
                torch.randn(TOKENS_PER_RANK, HIDDEN, device=device, generator=generator, dtype=torch.float32)
                .to(torch.bfloat16)
                .requires_grad_(True)
            )
            top_k_index, top_k_weights = build_routing(TOKENS_PER_RANK, num_experts, TOP_K, device, generator)
            top_k_weights = top_k_weights.clone().requires_grad_(True)
            loss_seed = torch.randn(
                TOKENS_PER_RANK, HIDDEN, device=device, generator=generator, dtype=torch.float32
            ).to(torch.bfloat16)

            holder.zero_grad(set_to_none=True)
            output = holder.experts(hidden, top_k_index, top_k_weights)
            (output.float() * loss_seed.float()).sum().backward()

            gate_up = _to_local(holder.experts.gate_up_proj.grad).detach().contiguous()
            down = _to_local(holder.experts.down_proj.grad).detach().contiguous()
            # Same EP rank, other edp rank: {0,2} and {1,3} on this mesh.
            gathered_gate = [torch.empty_like(gate_up) for _ in range(edp_size)]
            gathered_down = [torch.empty_like(down) for _ in range(edp_size)]
            dist.all_gather(gathered_gate, gate_up, group=runtime.edp_group)
            dist.all_gather(gathered_down, down, group=runtime.edp_group)
            peer_gate = gathered_gate[1 - edp_rank]
            peer_down = gathered_down[1 - edp_rank]

            for name, mine, peer in (
                ("grad(gate_up_proj)", gate_up, peer_gate),
                ("grad(down_proj)", down, peer_down),
            ):
                error = relative_error(mine, peer)
                if rank == 0:
                    print(
                        f"step {step}: {name} replica relerr {error:.4e} "
                        f"(edp {edp_rank}<->{1 - edp_rank}, ep={ep_rank})",
                        flush=True,
                    )
                if error > GRADIENT_TOLERANCE:
                    failures.append(f"step {step}: {name} replica relerr {error:.4e} > {GRADIENT_TOLERANCE:.1e}")

        failure_count = torch.tensor([len(failures)], device=device, dtype=torch.int32)
        dist.all_reduce(failure_count)
        if int(failure_count) and rank == 0:
            print(f"FAILED: {int(failure_count)} replica check(s) across all ranks", flush=True)
        for failure in failures:
            print(f"[rank {rank}] {failure}", flush=True)
        if rank == 0 and not int(failure_count):
            print(
                f"D-2 probe OK: replica home_grad agreed over {STEPS} steps "
                f"(world=4, EP=2, efsdp=1, edp=2). Not a support claim.",
                flush=True,
            )
        return 1 if int(failure_count) else 0
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())

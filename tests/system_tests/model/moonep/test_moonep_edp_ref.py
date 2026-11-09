# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Absolute-reference verification of the MoonEP replica reduction at ``edp>1``.

The D-2 probe compares the two expert copies with each other, which cannot see a
uniform scale error: if the replica reduction averages where ``gradient_divide_factor``
has already divided, both copies are wrong by the same ``1/edp`` and their relative
error is still exactly ``0.0000e+00``. This script compares against the replicated
reference instead, summed over every rank that contributed tokens, so the magnitude
is checked and not only the agreement.

One file rather than six, selected by environment variable:

  * ``MOONEP_NUM_EXPERTS`` pins E so ``(edp=1, ep=world)`` and ``(edp=2, ep=world/2)``
    describe the same model and can be compared across runs (V-1);
  * the default ``(edp=2, ep=2)`` run is the absolute reference check itself (V-2);
  * ``MOONEP_OPT_LR`` turns on a weight update and the replica drift check (V-3);
  * ``MOONEP_ACCUM_STEPS`` accumulates micro-batches before the check (V-5);
  * ``MOONEP_MEASURE`` with ``MOONEP_EDP_SYNC=0/1`` is the same-config A/B for the
    cost of the reduction (V-6, V-7);
  * ``MOONEP_EDP=4`` on 8 cards, and ``MOONEP_CKPT`` for a shard-level round trip (V-8).

A pass here does **not** upgrade the 7.8 matrix. It says the replica reduction has
the value the divide factor implies; it does not say ``edp>1`` is supported. The
tolerances come from ``test_moonep.py`` and must not be relaxed here.

This ST builds ``EPPlanConfig`` directly and never calls
``FSDPTurboConfig.validate_ep_config``, so the D-3 ``edp>1`` guard does not gate it
even on a branch that carries the guard.

Launch with ``run_test_moonep_edp_ref.sh``.
"""

from __future__ import annotations

import importlib.util
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from fsdp_turbo.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig, MoonEPConfig
from fsdp_turbo.utils.device import get_accelerator_name, set_accelerator_compatible
from fsdp_turbo.utils.log import set_log_level


def _load_st():
    """``test_moonep.py`` is a script, not a module; load it by path like D-2 does."""
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
OUTPUT_TOLERANCE = _st.OUTPUT_TOLERANCE
SEED = _st.SEED
STEPS = _st.STEPS
TOKENS_PER_RANK = _st.TOKENS_PER_RANK
TOP_K = _st.TOP_K
ExpertsHolder = _st.ExpertsHolder
FusedExperts = _st.FusedExperts
_to_local = _st._to_local
build_routing = _st.build_routing
reference_moe = _st.reference_moe
relative_error = _st.relative_error


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


EDP = int(os.environ.get("MOONEP_EDP", "2"))
EP = int(os.environ.get("MOONEP_EP", "2"))
# Off is a real configuration, not a debug switch: it is the no-sync arm of the
# V-6 / V-7 A/B, and on this branch it reproduces what 33bda71 does at edp>1.
SYNC = _flag("MOONEP_EDP_SYNC", "1")
ACCUM_STEPS = int(os.environ.get("MOONEP_ACCUM_STEPS", "1"))
OPT_LR = float(os.environ.get("MOONEP_OPT_LR", "0"))
DRIFT_EVERY = int(os.environ.get("MOONEP_DRIFT_EVERY", "10"))
MEASURE = _flag("MOONEP_MEASURE", "0")
# The reference is a naive per-expert loop with its own FP32 buffers and autograd
# graph. It costs more than the kernel under test and allocates more, so a timing or
# memory run has to have it off or it measures the reference instead. The launcher
# turns it off with MOONEP_MEASURE=1 unless asked otherwise.
REF_CHECK = _flag("MOONEP_REF_CHECK", "1")
WARMUP = int(os.environ.get("MOONEP_WARMUP", "2"))
DUMP_DIR = os.environ.get("MOONEP_DUMP_DIR", "")
CKPT = _flag("MOONEP_CKPT", "0")
_NUM_EXPERTS_ENV = os.environ.get("MOONEP_NUM_EXPERTS", "")


def resolve_num_experts(ep_size: int) -> int:
    """E must be identical across configs to compare runs, and E/ep a multiple of four.

    The multiple-of-four rule is the 2 MiB VMM granularity constraint documented in
    ``test_moonep.py``: at hidden == ffn == 512 one expert row is 512 KiB.
    """
    if _NUM_EXPERTS_ENV:
        num_experts = int(_NUM_EXPERTS_ENV)
    else:
        num_experts = EXPERTS_PER_RANK * ep_size
    if num_experts % ep_size:
        raise RuntimeError(f"MOONEP_NUM_EXPERTS={num_experts} is not divisible by ep={ep_size}.")
    per_rank = num_experts // ep_size
    if per_rank % 4:
        raise RuntimeError(
            f"E/ep = {per_rank} is not a multiple of four; the compact GEMM view needs the "
            f"per-rank expert block to be a multiple of the 2 MiB VMM granularity."
        )
    return num_experts


def local_shard(tensor: torch.Tensor, ep_rank: int, per_rank: int) -> torch.Tensor:
    """Accept either a sharded local tensor or a full one, so this works for grads and params."""
    if tensor.shape[0] == per_rank:
        return tensor
    return tensor[ep_rank * per_rank : (ep_rank + 1) * per_rank]


def gather_full_weights(holder, ep_group, ep_size: int, ep_rank: int, per_rank: int):
    """Rebuild the full [E, ...] weights from the EP shards.

    The reference has to be evaluated at MoonEP's *current* weights, otherwise a
    long run with ``MOONEP_OPT_LR`` compares two models that have drifted apart and
    the per-step error stops meaning anything.
    """
    full = []
    for parameter in (holder.experts.gate_up_proj, holder.experts.down_proj):
        shard = local_shard(_to_local(parameter).detach(), ep_rank, per_rank).contiguous()
        gathered = [torch.empty_like(shard) for _ in range(ep_size)]
        dist.all_gather(gathered, shard, group=ep_group)
        full.append(torch.cat(gathered, dim=0))
    return full[0], full[1]


def make_batch(device, generator, num_experts: int):
    """Same draw order as ``test_moonep.py`` so a pinned E gives an identical batch stream."""
    hidden = (
        torch.randn(TOKENS_PER_RANK, HIDDEN, device=device, generator=generator, dtype=torch.float32)
        .to(torch.bfloat16)
        .requires_grad_(True)
    )
    top_k_index, top_k_weights = build_routing(TOKENS_PER_RANK, num_experts, TOP_K, device, generator)
    top_k_weights = top_k_weights.clone().requires_grad_(True)
    loss_seed = torch.randn(TOKENS_PER_RANK, HIDDEN, device=device, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    return hidden, top_k_index, top_k_weights, loss_seed


def peak_memory_gib() -> float:
    try:
        return float(torch.accelerator.max_memory_allocated()) / (1024**3)
    except (AttributeError, RuntimeError):
        # torch_npu 2.10 registers no generic DeviceAllocator, so the
        # torch.accelerator memory-stat dispatch raises; the concrete device
        # module (torch.npu) still works, per the streams.py API-family rule.
        try:
            device_module = torch.get_device_module(get_accelerator_name())
            return float(device_module.max_memory_allocated()) / (1024**3)
        except (AttributeError, RuntimeError):
            return float("nan")


def reset_peak_memory() -> None:
    try:
        torch.accelerator.reset_peak_memory_stats()
    except (AttributeError, RuntimeError):
        try:
            torch.get_device_module(get_accelerator_name()).reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            pass


def main() -> int:  # noqa: C901 - one linear scenario, splitting it would hide the order
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
    if EDP * EP != world_size:
        raise RuntimeError(f"MOONEP_EDP({EDP}) * MOONEP_EP({EP}) must equal world size {world_size}.")

    device_type = get_accelerator_name()
    device = torch.device(device_type, local_rank)
    num_experts = resolve_num_experts(EP)
    per_rank = num_experts // EP

    # (edp, efsdp, ep): rank = edp_index * ep + ep_index, so an edp group is the
    # set of ranks holding the same expert slice in different copies.
    mesh = init_device_mesh(device_type, (EDP, 1, EP), mesh_dim_names=("edp", "efsdp", "ep"))
    ep_mesh = mesh["ep"]
    edp_mesh = mesh["edp"]
    ep_group = ep_mesh.get_group()
    ep_rank = dist.get_rank(group=ep_group)
    edp_rank = dist.get_rank(group=edp_mesh.get_group())

    # Fixed rather than left to default so both arms of every A/B divide by the same
    # number and a mismatch can only come from the reduction itself.
    factor = float(world_size)
    plan = EPPlanConfig(
        apply_modules=["experts"],
        dispatcher="moonep",
        gradient_divide_factor=factor,
        moonep_config=MoonEPConfig(tokens_per_rank=TOKENS_PER_RANK, top_k=TOP_K, **KNOBS),
    )

    torch.manual_seed(SEED)
    holder = ExpertsHolder(FusedExperts(num_experts, HIDDEN, FFN, device))
    for parameter in holder.parameters():
        dist.broadcast(parameter.data, src=0)
    activation = holder.experts.act_fn

    # With the reduction on, an expert's gradient collects tokens from the whole
    # world; with it off, only from its own EP group. The reference has to be summed
    # over exactly the ranks MoonEP claims to have covered.
    use_edp_mesh = SYNC and EDP > 1
    reference_group = None if use_edp_mesh else ep_group

    runtime = None
    failures: list[str] = []
    step_times: list[float] = []
    if DUMP_DIR:
        Path(DUMP_DIR).mkdir(parents=True, exist_ok=True)

    try:
        expert_parallelize_modules(holder, ep_mesh, plan, edp_mesh=edp_mesh if use_edp_mesh else None)
        runtime = holder._moonep_runtime
        edp_group = runtime.edp_group
        edp_group_size = 0 if edp_group is None else dist.get_world_size(group=edp_group)
        if rank == 0:
            print(
                f"EDPREF config: world={world_size} edp={EDP} efsdp=1 ep={EP} "
                f"edp_group size={edp_group_size} sync={'on' if use_edp_mesh else 'off'} "
                f"E={num_experts} E/ep={per_rank} factor={factor:g} "
                f"steps={STEPS} accum={ACCUM_STEPS} lr={OPT_LR:g} "
                f"ref_check={'on' if REF_CHECK else 'off'} "
                f"(absolute-reference check; not a support claim)",
                flush=True,
            )
        if use_edp_mesh and edp_group_size != EDP:
            raise RuntimeError(f"expected an edp group of size {EDP}; got {edp_group_size}.")
        if not use_edp_mesh and edp_group is not None and edp_group_size > 1:
            raise RuntimeError("sync was requested off but the runtime still has an edp group.")

        generator = torch.Generator(device=device).manual_seed(SEED + rank)
        checked_runtime_sees_updates = False

        for step in range(STEPS):
            if MEASURE and step == WARMUP:
                reset_peak_memory()
            measuring = MEASURE and step >= WARMUP
            if measuring:
                torch.accelerator.synchronize()
                started = time.perf_counter()

            full_gate_up = full_down = None
            reference_gate_up = reference_down = None
            if REF_CHECK:
                full_gate_up, full_down = gather_full_weights(holder, ep_group, EP, ep_rank, per_rank)
                reference_gate_up = torch.zeros(num_experts, 2 * FFN, HIDDEN, device=device, dtype=torch.float32)
                reference_down = torch.zeros(num_experts, HIDDEN, FFN, device=device, dtype=torch.float32)

            holder.zero_grad(set_to_none=True)
            last_batch = None
            for micro in range(ACCUM_STEPS):
                hidden, top_k_index, top_k_weights, loss_seed = make_batch(device, generator, num_experts)
                output = holder.experts(hidden, top_k_index, top_k_weights)
                (output.float() * loss_seed.float()).sum().backward()
                last_batch = (hidden, top_k_index, top_k_weights, loss_seed, output.detach().clone())
                if not REF_CHECK:
                    continue

                reference_hidden = hidden.detach().clone().requires_grad_(True)
                reference_weights = top_k_weights.detach().clone().requires_grad_(True)
                leaf_gate_up = full_gate_up.clone().requires_grad_(True)
                leaf_down = full_down.clone().requires_grad_(True)
                reference_output = reference_moe(
                    reference_hidden, top_k_index, reference_weights, leaf_gate_up, leaf_down, activation
                )
                (reference_output.float() * loss_seed.float()).sum().backward()
                # FP32 accumulation: with ACCUM_STEPS > 1 a BF16 running sum would
                # contribute more error than the kernel under test.
                reference_gate_up += leaf_gate_up.grad.float()
                reference_down += leaf_down.grad.float()

                # These three are local to the rank and independent of the replica
                # reduction, so they are the zero-regression half of the check.
                for name, actual, expected, tolerance in (
                    ("output", last_batch[4], reference_output, OUTPUT_TOLERANCE),
                    ("grad(hidden)", hidden.grad, reference_hidden.grad, GRADIENT_TOLERANCE),
                    ("grad(router probs)", top_k_weights.grad, reference_weights.grad, GRADIENT_TOLERANCE),
                ):
                    error = relative_error(actual, expected)
                    if math.isnan(error) or error > tolerance:
                        failures.append(f"step {step} micro {micro}: {name} relerr {error:.4e} > {tolerance:.1e}")
                    elif rank == 0 and micro == ACCUM_STEPS - 1:
                        print(f"step {step}: {name:<20s} relerr {error:.4e}", flush=True)

            moonep_gate_up = local_shard(_to_local(holder.experts.gate_up_proj.grad), ep_rank, per_rank)
            moonep_down = local_shard(_to_local(holder.experts.down_proj.grad), ep_rank, per_rank)
            expert_slice = slice(ep_rank * per_rank, (ep_rank + 1) * per_rank)

            if REF_CHECK:
                dist.all_reduce(reference_gate_up, group=reference_group)
                dist.all_reduce(reference_down, group=reference_group)
                reference_gate_up /= factor
                reference_down /= factor
                for name, actual, expected in (
                    ("grad(gate_up_proj)", moonep_gate_up, reference_gate_up[expert_slice]),
                    ("grad(down_proj)", moonep_down, reference_down[expert_slice]),
                ):
                    error = relative_error(actual, expected)
                    # The scale ratio is the tell for a divide-factor bug: a missing or
                    # doubled division shows up as 0.5 or 2.0 while relerr alone only
                    # says "wrong".
                    actual_norm = float(actual.detach().float().norm())
                    expected_norm = float(expected.detach().float().norm())
                    ratio = actual_norm / expected_norm if expected_norm else float("nan")
                    if math.isnan(error) or error > GRADIENT_TOLERANCE:
                        failures.append(
                            f"step {step}: {name} relerr {error:.4e} > {GRADIENT_TOLERANCE:.1e} "
                            f"(norm ratio {ratio:.6f}; 0.5 or 2.0 means the divide factor, "
                            f"not the reduction)"
                        )
                    elif rank == 0:
                        print(
                            f"step {step}: {name:<20s} relerr {error:.4e} norm ratio {ratio:.6f}",
                            flush=True,
                        )

            if DUMP_DIR and edp_rank == 0:
                import numpy as np

                target = Path(DUMP_DIR) / f"grads_edp{EDP}_ep{EP}_step{step}_eprank{ep_rank}.npz"
                np.savez(
                    target,
                    gate_up=moonep_gate_up.detach().float().cpu().numpy(),
                    down=moonep_down.detach().float().cpu().numpy(),
                    expert_start=np.int64(ep_rank * per_rank),
                    expert_stop=np.int64((ep_rank + 1) * per_rank),
                    num_experts=np.int64(num_experts),
                    factor=np.float64(factor),
                    edp=np.int64(EDP),
                    ep=np.int64(EP),
                    step=np.int64(step),
                )

            if OPT_LR > 0:
                # Plain SGD, applied by hand: identical gradients and a deterministic
                # rule are what make the replica comparison below a bitwise one.
                pre_update_output = last_batch[4]
                with torch.no_grad():
                    for parameter in (holder.experts.gate_up_proj, holder.experts.down_proj):
                        data = local_shard(_to_local(parameter).data, ep_rank, per_rank)
                        grad = local_shard(_to_local(parameter.grad), ep_rank, per_rank)
                        data.add_(grad.to(data.dtype), alpha=-OPT_LR)

                if not checked_runtime_sees_updates:
                    checked_runtime_sees_updates = True
                    with torch.no_grad():
                        recomputed = holder.experts(last_batch[0], last_batch[1], last_batch[2])
                    moved = relative_error(recomputed, pre_update_output)
                    if rank == 0:
                        print(f"step {step}: output moved {moved:.4e} after the weight update", flush=True)
                    if moved == 0.0:
                        failures.append(
                            "the forward did not change after the weight update; the runtime is "
                            "holding its own copy of the expert weights, so the drift check below "
                            "is vacuous and V-3 needs the full-model harness instead"
                        )

                if EDP > 1 and edp_group is not None and (step % DRIFT_EVERY == 0 or step == STEPS - 1):
                    for name, parameter in (
                        ("gate_up_proj", holder.experts.gate_up_proj),
                        ("down_proj", holder.experts.down_proj),
                    ):
                        mine = local_shard(_to_local(parameter).detach(), ep_rank, per_rank).contiguous()
                        gathered = [torch.empty_like(mine) for _ in range(EDP)]
                        dist.all_gather(gathered, mine, group=edp_group)
                        worst = 0.0
                        for other in range(EDP):
                            if other == edp_rank:
                                continue
                            worst = max(worst, float((mine.float() - gathered[other].float()).abs().max()))
                        if rank == 0:
                            print(f"step {step}: replica drift {name:<14s} max abs {worst:.3e}", flush=True)
                        # Exact, not toleranced: the replicas consume the same reduced
                        # gradient through the same update, so any nonzero bit is drift.
                        if worst != 0.0:
                            failures.append(f"step {step}: {name} replicas differ by {worst:.3e}, expected 0")

            if measuring:
                torch.accelerator.synchronize()
                step_times.append(time.perf_counter() - started)

        if CKPT:
            # Shard-level round trip only. A real resume test needs the checkpoint
            # harness; this one catches a MoonEP-managed parameter that cannot be
            # serialised or that comes back with the wrong slice.
            checkpoint_dir = Path(os.environ.get("MOONEP_CKPT_DIR", "./logs/edp_ckpt"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"shard_rank{rank}.pt"
            before = {
                name: local_shard(_to_local(parameter).detach(), ep_rank, per_rank).clone()
                for name, parameter in (
                    ("gate_up_proj", holder.experts.gate_up_proj),
                    ("down_proj", holder.experts.down_proj),
                )
            }
            torch.save({name: tensor.cpu() for name, tensor in before.items()}, path)
            restored = torch.load(path, map_location=device)
            for name, tensor in before.items():
                worst = float((tensor.float() - restored[name].float()).abs().max())
                if rank == 0:
                    print(f"ckpt: {name:<14s} round-trip max abs {worst:.3e}", flush=True)
                if worst != 0.0:
                    failures.append(f"ckpt: {name} changed by {worst:.3e} across save/load")

        failure_count = torch.tensor([len(failures)], device=device, dtype=torch.int32)
        dist.all_reduce(failure_count)
        for failure in failures:
            print(f"[rank {rank}] {failure}", flush=True)

        if MEASURE and step_times:
            # HCCL rejects FP64 (handover 7.11). These tensors only reduce
            # timing / memory scalars, not expert gradients.
            timings = torch.tensor(
                [statistics.median(step_times), sum(step_times) / len(step_times)],
                device=device,
                dtype=torch.float32,
            )
            dist.all_reduce(timings, op=dist.ReduceOp.MAX)
            memory = torch.tensor([peak_memory_gib()], device=device, dtype=torch.float32)
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(
                    f"EDPMEASURE edp={EDP} ep={EP} sync={'on' if use_edp_mesh else 'off'} "
                    f"measured_steps={len(step_times)} median_s={float(timings[0]):.6f} "
                    f"mean_s={float(timings[1]):.6f} peak_alloc_gib={float(memory[0]):.4f} "
                    f"(slowest rank; compare only against the other arm of the same A/B, "
                    f"and only a gap over the 7.7 noise floor of +-2% counts)",
                    flush=True,
                )

        if rank == 0:
            if int(failure_count):
                print(f"FAILED: {int(failure_count)} check(s) across all ranks", flush=True)
            elif REF_CHECK:
                print(
                    f"EDPREF OK: MoonEP matched the world-summed reference over {STEPS} steps "
                    f"(world={world_size}, edp={EDP}, ep={EP}, sync="
                    f"{'on' if use_edp_mesh else 'off'}, E={num_experts}, accum={ACCUM_STEPS}). "
                    f"Not a support claim; 7.8 stays as it is.",
                    flush=True,
                )
            else:
                print(
                    f"EDPREF cost run finished with the reference off, so it checked no values "
                    f"(world={world_size}, edp={EDP}, ep={EP}, sync="
                    f"{'on' if use_edp_mesh else 'off'}). Read EDPMEASURE only.",
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

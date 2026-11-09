# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""End-to-end numerical verification of the MoonEP expert-parallel dispatcher.

Compares the forward output and every gradient against a replicated reference in
which each rank owns all experts and computes the MoE naively. Comparing every
tensor rather than a loss is what makes dispatch, prefetch, combine and the
home-rank reduction verifiable separately.

That separation has already earned itself. When the home-rank reduction was
reading slot buffers before their peers had written them, this test's output,
hidden-state gradient and router-probability gradient were all still normal
(4.5e-4, 4.9e-3, 3.6e-3); only the two expert-weight gradients moved, to 4.7e-1.
Three of the five checks saw nothing, so a metric downstream of the output alone
would not have caught it.

Device agnostic; launch with ``run_test_moonep.sh`` or torchrun.
"""

from __future__ import annotations

import os
import sys
import math

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from fsdp_turbo.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig, MoonEPConfig
from fsdp_turbo.utils.device import get_accelerator_name, set_accelerator_compatible
from fsdp_turbo.utils.log import set_log_level

# MoonEP kernel constraints these shapes satisfy:
#   * every expert matrix dimension is a multiple of 128;
#   * the per-rank expert block (E/R * out * in * 2 bytes) is a multiple of the
#     2 MiB VMM allocation granularity, which the compact GEMM view requires.
# With hidden == ffn == 512 the per-expert row is 512 KiB, so E/R must be a
# multiple of four; E = 4 * R keeps that true for any EP size.
HIDDEN = int(os.environ.get("MOONEP_HIDDEN", "512"))
FFN = int(os.environ.get("MOONEP_FFN", "512"))
TOKENS_PER_RANK = int(os.environ.get("MOONEP_TOKENS", "512"))
TOP_K = int(os.environ.get("MOONEP_TOP_K", "4"))
EXPERTS_PER_RANK = int(os.environ.get("MOONEP_EXPERTS_PER_RANK", "4"))
STEPS = int(os.environ.get("MOONEP_STEPS", "3"))
SEED = int(os.environ.get("SEED", "42"))

# BF16 grouped GEMMs accumulate in a different order than the reference, so the
# comparison is on relative L2 error rather than element-wise closeness.
#
# The tolerance only has to separate correct from broken, and those are far
# apart: a correct run lands around 6e-3 on the gradients, while the run with
# the home-rank reduction reading unwritten peer slots produced 4.7e-1 (file
# header above). That is 77x, so these values sit between them with roughly 4x
# of headroom and are not a tight fit to any one measurement.
#
# A single tolerance covers every shape because the error does not grow with the
# reduction length: it comes from BF16 quantization of the operands and the
# output, which is length independent and matches the observed 2^-8 magnitude,
# rather than from accumulation order.
OUTPUT_TOLERANCE = float(os.environ.get("MOONEP_OUTPUT_TOLERANCE", "4e-3"))
GRADIENT_TOLERANCE = float(os.environ.get("MOONEP_GRADIENT_TOLERANCE", "2e-2"))


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


# The tuning knobs, under the same names the full-model harness uses. Comparing
# losses says a knob did not derail the run; comparing every tensor against the
# replicated reference says the dispatch, the prefetch, the combine and the
# home-rank reduction are each still right. Only the second is worth much for a
# knob that changes how ranks synchronize, so every knob has to be reachable
# here, not only from a full-model run.
KNOB_DEFAULTS = {
    "async_finish": False,
    "recompute_dispatch": False,
    "shmem_stream_barrier": True,
    # On here even though the harness leaves it off: this test is about
    # correctness, so the device-side plan assertions should run.
    "validate_plan_invariants": True,
    "npu_async_comm": False,
    # Diagnostics knobs, also settings-owned since the env.py removal: reachable
    # here so a run can flip them without editing the config constructor.
    "trace_enabled": False,
    "trace_sync_enabled": False,
    "vmm_probe_enabled": False,
}
KNOBS = {name: _flag(f"MOONEP_{name.upper()}", "1" if default else "0") for name, default in KNOB_DEFAULTS.items()}


class FusedExperts(torch.nn.Module):
    """The fused 3-D expert layout used by Kimi K3 and Qwen3.5-MoE."""

    def __init__(self, num_experts: int, hidden_dim: int, ffn_dim: int, device, dtype=torch.bfloat16):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.act_fn = torch.nn.SiLU()
        scale = hidden_dim**-0.5
        self.gate_up_proj = torch.nn.Parameter(
            torch.randn(num_experts, 2 * ffn_dim, hidden_dim, device=device, dtype=dtype) * scale
        )
        self.down_proj = torch.nn.Parameter(
            torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype) * scale
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Naive reference: every expert is local and applied under a mask."""
        return reference_moe(hidden_states, top_k_index, top_k_weights, self.gate_up_proj, self.down_proj, self.act_fn)


class ExpertsHolder(torch.nn.Module):  # pylint: disable=abstract-method
    """Named wrapper so ``ep_plan.apply_modules`` has a module path to match."""

    def __init__(self, experts: FusedExperts):
        super().__init__()
        self.experts = experts


def reference_moe(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj, act_fn):
    """Dense masked MoE over all experts, in the caller's dtype."""
    tokens = hidden_states.shape[0]
    output = torch.zeros(tokens, hidden_states.shape[-1], dtype=torch.float32, device=hidden_states.device)
    for expert in range(gate_up_proj.shape[0]):
        token_ids, choice_ids = (top_k_index == expert).nonzero(as_tuple=True)
        if token_ids.numel() == 0:
            continue
        rows = hidden_states.index_select(0, token_ids)
        fc1 = rows @ gate_up_proj[expert].transpose(0, 1)
        gate, up = fc1.chunk(2, dim=-1)
        expert_out = (act_fn(gate) * up) @ down_proj[expert].transpose(0, 1)
        weights = top_k_weights[token_ids, choice_ids].to(expert_out.dtype).unsqueeze(-1)
        output.index_add_(0, token_ids, (expert_out * weights).float())
    return output.to(hidden_states.dtype)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative L2 error, computed in FP32 to keep the metric meaningful."""
    actual = actual.detach().float()
    expected = expected.detach().float()
    denominator = expected.norm()
    if denominator == 0:
        return float(actual.norm())
    return float((actual - expected).norm() / denominator)


def build_routing(tokens: int, num_experts: int, top_k: int, device, generator):
    """Deliberately skewed routing so the planner has to migrate experts."""
    logits = torch.randn(tokens, num_experts, device=device, generator=generator, dtype=torch.float32)
    # Bias a quarter of the experts upward: an even router would need no
    # redundant experts at all and would not exercise the prefetch path.
    hot = torch.zeros(num_experts, device=device, dtype=torch.float32)
    hot[: max(1, num_experts // 4)] = 4.0
    probabilities = torch.softmax(logits + hot, dim=-1)
    top_k_weights, top_k_index = torch.topk(probabilities, top_k, dim=-1)
    return top_k_index.to(torch.int32).contiguous(), top_k_weights.contiguous()


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
    device = torch.device(get_accelerator_name(), local_rank)
    num_experts = EXPERTS_PER_RANK * world_size
    torch.manual_seed(SEED)

    ep_mesh = init_device_mesh(get_accelerator_name(), (world_size,), mesh_dim_names=("ep",))
    plan = EPPlanConfig(
        apply_modules=["experts"],
        dispatcher="moonep",
        moonep_config=MoonEPConfig(
            tokens_per_rank=TOKENS_PER_RANK,
            top_k=TOP_K,
            **KNOBS,
        ),
    )

    holder = ExpertsHolder(FusedExperts(num_experts, HIDDEN, FFN, device))
    # Every rank starts from identical expert weights so the replicated
    # reference and the EP-sharded MoonEP path describe the same model.
    for parameter in holder.parameters():
        dist.broadcast(parameter.data, src=0)
    reference_gate_up = holder.experts.gate_up_proj.detach().clone()
    reference_down = holder.experts.down_proj.detach().clone()
    activation = holder.experts.act_fn

    runtime = None
    failures: list[str] = []
    try:
        expert_parallelize_modules(holder, ep_mesh, plan)
        runtime = holder._moonep_runtime
        # D-1 evidence: this ST never passes edp_mesh, so the replica
        # all-reduce must stay a no-op. Print it so a machine log can
        # prove that without opening the runtime.
        if rank == 0:
            edp_group = runtime.edp_group
            edp_size = 0 if edp_group is None else dist.get_world_size(group=edp_group)
            print(
                f"D-1 probe: edp_group is None={edp_group is None} size={edp_size} "
                f"world={world_size} ep={world_size} (sync_replica_gradients no-op when None or size<=1)",
                flush=True,
            )

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

            # --- MoonEP path -------------------------------------------------
            holder.zero_grad(set_to_none=True)
            output = holder.experts(hidden, top_k_index, top_k_weights)
            (output.float() * loss_seed.float()).sum().backward()
            moonep_output = output.detach().clone()
            moonep_hidden_grad = hidden.grad.detach().clone()
            moonep_weight_grad = top_k_weights.grad.detach().clone()
            gate_up_grad = holder.experts.gate_up_proj.grad
            down_grad = holder.experts.down_proj.grad
            moonep_gate_up_grad = _to_local(gate_up_grad).detach().clone()
            moonep_down_grad = _to_local(down_grad).detach().clone()

            # --- replicated reference ----------------------------------------
            ref_hidden = hidden.detach().clone().requires_grad_(True)
            ref_weights = top_k_weights.detach().clone().requires_grad_(True)
            ref_gate_up = reference_gate_up.clone().requires_grad_(True)
            ref_down = reference_down.clone().requires_grad_(True)
            ref_output = reference_moe(ref_hidden, top_k_index, ref_weights, ref_gate_up, ref_down, activation)
            (ref_output.float() * loss_seed.float()).sum().backward()
            # An expert's true gradient collects tokens from every rank, so the
            # reference must be summed across the EP group before it can be
            # compared with MoonEP's home-rank result.
            dist.all_reduce(ref_gate_up.grad, group=ep_mesh.get_group())
            dist.all_reduce(ref_down.grad, group=ep_mesh.get_group())
            experts_per_rank = num_experts // world_size
            expert_slice = slice(rank * experts_per_rank, (rank + 1) * experts_per_rank)

            checks = (
                ("output", moonep_output, ref_output, OUTPUT_TOLERANCE),
                ("grad(hidden)", moonep_hidden_grad, ref_hidden.grad, GRADIENT_TOLERANCE),
                ("grad(router probs)", moonep_weight_grad, ref_weights.grad, GRADIENT_TOLERANCE),
                (
                    "grad(gate_up_proj)",
                    moonep_gate_up_grad,
                    ref_gate_up.grad[expert_slice],
                    GRADIENT_TOLERANCE,
                ),
                ("grad(down_proj)", moonep_down_grad, ref_down.grad[expert_slice], GRADIENT_TOLERANCE),
            )
            for name, actual, expected, tolerance in checks:
                error = relative_error(actual, expected)
                if math.isnan(error) or error > tolerance:
                    failures.append(f"step {step}: {name} relative error {error:.4e} > {tolerance:.1e}")
                elif rank == 0:
                    print(f"step {step}: {name:<20s} relative error {error:.4e}", flush=True)

        # Surface a failure on any rank as a failure of the whole job.
        failure_count = torch.tensor([len(failures)], device=device, dtype=torch.int32)
        dist.all_reduce(failure_count)
        if int(failure_count) and rank == 0:
            print(f"FAILED: {int(failure_count)} check(s) across all ranks", flush=True)
        for failure in failures:
            print(f"[rank {rank}] {failure}", flush=True)
        if rank == 0 and not int(failure_count):
            # Printed with the result, not just the shape: a pass means nothing
            # unless the reader knows which configuration produced it.
            changed = ", ".join(f"{name}={value}" for name, value in KNOBS.items() if value != KNOB_DEFAULTS[name])
            print(
                f"OK: MoonEP matches the replicated reference over {STEPS} steps "
                f"(E={num_experts}, R={world_size}, S={TOKENS_PER_RANK}, K={TOP_K}; "
                f"knobs: {changed or 'all default'}).",
                flush=True,
            )
        return 1 if int(failure_count) else 0
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _to_local(tensor):
    from torch.distributed.tensor import DTensor

    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


if __name__ == "__main__":
    sys.exit(main())

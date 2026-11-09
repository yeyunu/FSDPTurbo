# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Fused-expert forward implementation backed by MoonEP.

The schedule of report S5.2.1: dispatch on the comm stream, queue both
projections' prefetches behind it, run the first GEMM as soon as the gate_up
slots are ready so the down prefetch overlaps it, then the second GEMM, the
router weighting and combine. Static shapes throughout, no host reads.
"""

from __future__ import annotations

import types

import torch

from fsdp_turbo.distributed.expert_parallel.moonep.autograd import (
    moonep_dispatch,
    moonep_down_and_combine,
    moonep_gate_up_gmm,
    moonep_weights_for_step,
)
from fsdp_turbo.distributed.expert_parallel.moonep.plan import verify_routing_range
from fsdp_turbo.distributed.expert_parallel.moonep.runtime import MoonEPRuntime
from fsdp_turbo.distributed.expert_parallel.utils import (
    fixed_router_for_debug,
    normalize_expert_args,
)


_REQUIRED_EXPERT_ATTRIBUTES = ("gate_up_proj", "down_proj", "act_fn", "hidden_dim", "num_global_experts")


def _validate_expert_module(module: torch.nn.Module) -> bool:
    """Check the fused 3-D expert contract and report the weights' orientation.

    Two layouts are accepted, and the return value says which one this module
    uses:

    * ``False`` -- ``gate_up_proj=[E, 2*Hp, H]`` and ``down_proj=[E, H, Hp]``,
      the linear-layer convention with the output dimension first. Qwen3-MoE and
      Qwen3.5-MoE store them this way.
    * ``True`` -- the transpose of both, ``[E, H, 2*Hp]`` and ``[E, Hp, H]``,
      which is how ``PatchKimiMoeExperts`` stores them.

    MoonEP's buffers mirror whichever layout the model brought, so the parameter
    keeps its shape and the model's state-dict ABI is untouched. Only the three
    grouped GEMMs care, and for them the orientation is one flag.

    Both projections must agree. A module that mixed the two would need two flags
    threaded separately, and no such model exists.
    """
    missing = [name for name in _REQUIRED_EXPERT_ATTRIBUTES if not hasattr(module, name)]
    if missing:
        # ``num_global_experts`` is set by ``expert_parallelize_modules`` on the
        # normal path; checking it here keeps a direct ``parallelize_moonep_module``
        # call from failing with a bare ``AttributeError`` further down.
        raise RuntimeError(
            "MoonEP currently supports fused expert modules with gate_up_proj, down_proj, "
            f"act_fn, hidden_dim and num_global_experts; missing {missing}."
        )
    if not callable(module.act_fn):
        raise RuntimeError("MoonEP requires the expert module to provide a callable act_fn.")
    for name in ("gate_up_proj", "down_proj"):
        value = getattr(module, name)
        if not isinstance(value, torch.nn.Parameter):
            raise RuntimeError(f"MoonEP requires {name} to be an unquantized nn.Parameter, got {type(value).__name__}.")
    gate_up = module.gate_up_proj
    down = module.down_proj
    expected_experts = int(module.num_global_experts)
    if gate_up.ndim != 3 or down.ndim != 3:
        raise RuntimeError("MoonEP fused expert weights must both be rank-3 tensors.")
    if gate_up.shape[0] != expected_experts or down.shape[0] != expected_experts:
        raise RuntimeError("MoonEP projection expert dimensions must match num_global_experts.")
    hidden = module.hidden_dim
    output_first = gate_up.shape[-1] == hidden and down.shape[-2] == hidden and gate_up.shape[-2] == 2 * down.shape[-1]
    input_first = gate_up.shape[-2] == hidden and down.shape[-1] == hidden and gate_up.shape[-1] == 2 * down.shape[-2]
    if not output_first and not input_first:
        raise RuntimeError(
            "MoonEP expects the two expert matrices to be gate_up_proj=[E,2*Hp,H] "
            "with down_proj=[E,H,Hp], or the transpose of both; got "
            f"{tuple(gate_up.shape)} and {tuple(down.shape)} with H={hidden}."
        )
    if output_first and input_first:
        # Only when 2*Hp == H == Hp, which cannot hold for a positive Hp. Kept as
        # an assertion because the flag would otherwise be silently ambiguous.
        raise RuntimeError(
            "MoonEP cannot tell which orientation these expert matrices use: "
            f"{tuple(gate_up.shape)} and {tuple(down.shape)} satisfy both."
        )
    return input_first


def _consumes_concatenated_gate_up(act_fn) -> bool:
    """True when ``act_fn`` splits ``[gate, up]`` itself instead of seeing the gate half.

    Kimi's ``SituAndMul`` (and the fused SiTU-GLU it wraps) take the concatenated
    first-projection output. The generic SwiGLU path is ``act_fn(gate) * up``.
    Matched across the MRO so a subclass keeps the SiTU path; the class is not
    imported because ``distributed/`` must not depend on model code.
    """
    return any(klass.__name__ == "SituAndMul" for klass in type(act_fn).__mro__)


def _activate_expert_hidden(module, gate_up: torch.Tensor) -> torch.Tensor:
    """Apply the expert activation without assuming a SwiGLU-shaped ``act_fn``.

    Prefer the module's own ``_apply_activation`` when present -- that is what
    ``PatchKimiMoeExperts`` uses to choose SiTU vs SiLU. Otherwise detect SiTU
    by the activation class, and only then fall back to the Qwen-style split.
    """
    apply = getattr(module, "_apply_activation", None)
    if callable(apply):
        return apply(gate_up).contiguous()
    if _consumes_concatenated_gate_up(module.act_fn):
        return module.act_fn(gate_up).contiguous()
    gate, up = gate_up.chunk(2, dim=-1)
    act_limit = getattr(module, "limit", None)
    if act_limit is not None:
        gate = gate.clamp(max=act_limit)
        up = up.clamp(min=-act_limit, max=act_limit)
    return (module.act_fn(gate) * up).contiguous()


def parallelize_moonep_module(module, runtime: MoonEPRuntime, ep_mesh, fixed_router=False):
    """Replace the expert weights with symmetric views and bind the forward."""
    transposed = _validate_expert_module(module)
    runtime.register_hidden_size(module.hidden_dim)
    gate_up = runtime.distribute_projection("gate_up", module.gate_up_proj, ep_mesh, transposed)
    down = runtime.distribute_projection("down", module.down_proj, ep_mesh, transposed)

    module.register_parameter("gate_up_proj", gate_up.parameter)
    module.register_parameter("down_proj", down.parameter)
    # Resource holders are deliberately neither modules nor buffers, so they do
    # not alter state_dict content or optimizer parameter discovery.
    object.__setattr__(module, "_moonep_runtime", runtime)
    object.__setattr__(module, "_moonep_gate_up", gate_up)
    object.__setattr__(module, "_moonep_down", down)
    module.forward = types.MethodType(get_moonep_experts_forward_fn(runtime, fixed_router=fixed_router), module)


def _fit_tokens_to_static_rank(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    tokens_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a microbatch to MoonEP's reserved ``[S, H]`` / ``[S, K]`` buffers.

    ``tokens_per_rank`` is the capacity reserved at heap init
    (``batch_size * max_seq_length / ulysses_size``), not a value the caller
    types into the EP plan. A shorter live prefix is padded: hidden and route
    weights are zero, expert ids are zero. Those pad rows do not change the
    live prefix of a grouped GEMM (zero activation, zero route weight). A
    longer microbatch cannot be addressed; raise ``data.max_seq_length``.
    """
    live = hidden_states.shape[0]
    if live == tokens_per_rank:
        return hidden_states, top_k_index, top_k_weights
    if live > tokens_per_rank:
        raise RuntimeError(
            "MoonEP reserved S="
            f"{tokens_per_rank} tokens/rank from data.batch_size * data.max_seq_length "
            f"(divided by Ulysses if enabled), but this step has S={live}. "
            "Raise data.max_seq_length to cover this step; do not set tokens_per_rank in the EP plan."
        )
    pad = tokens_per_rank - live
    hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad))
    top_k_index = torch.nn.functional.pad(top_k_index, (0, 0, 0, pad))
    top_k_weights = torch.nn.functional.pad(top_k_weights, (0, 0, 0, pad))
    return hidden_states, top_k_index, top_k_weights


def _tokens_per_expert(top_k_index: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Histogram the routed expert ids without a device-to-host synchronization.

    ``torch.bincount`` sizes its output from the data and therefore reads a
    device scalar back to the host, which would reintroduce the per-layer stall
    that MoonEP's static shapes exist to remove.
    """
    flat = top_k_index.reshape(-1).to(torch.int64)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=top_k_index.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    return counts


def get_moonep_experts_forward_fn(runtime: MoonEPRuntime, fixed_router=False):
    def moonep_experts_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ):
        top_k_index, top_k_weights = normalize_expert_args(top_k_index, top_k_weights)
        if top_k_index.ndim == 1:
            top_k_index = top_k_index.unsqueeze(-1)
            top_k_weights = top_k_weights.unsqueeze(-1)
        if fixed_router:
            top_k_index, top_k_weights = fixed_router_for_debug(top_k_index, top_k_weights, self.num_global_experts)

        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim).contiguous()
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(f"MoonEP expert input must be BF16, got {hidden_states.dtype}.")

        live_tokens = hidden_states.shape[0]
        top_k = top_k_index.shape[-1]
        if top_k_index.shape[0] != live_tokens or top_k_weights.shape != top_k_index.shape:
            raise RuntimeError(
                "MoonEP expects hidden [S,H], expert indices [S,K], and route weights [S,K]; got "
                f"{tuple(hidden_states.shape)}, {tuple(top_k_index.shape)}, {tuple(top_k_weights.shape)}."
            )

        top_k_index = top_k_index.to(dtype=torch.int32).contiguous()
        top_k_weights = top_k_weights.float().contiguous()
        reserved = runtime.configured_tokens_per_rank
        if reserved is not None:
            hidden_states, top_k_index, top_k_weights = _fit_tokens_to_static_rank(
                hidden_states, top_k_index, top_k_weights, reserved
            )
        tokens_per_rank = hidden_states.shape[0]
        tokens_per_expert = _tokens_per_expert(top_k_index, self.num_global_experts)

        call = runtime.new_call(tokens_per_rank, self.hidden_dim, top_k)
        if call.validate:
            verify_routing_range(top_k_index, self.num_global_experts)
        dispatched, route_weights_nvs = moonep_dispatch(
            call, hidden_states, top_k_weights, top_k_index, tokens_per_expert
        )
        # Both prefetches are queued behind the dispatch on the communication
        # stream; only the gate_up copy is waited on before the first GEMM.
        call.start_prefetch(self._moonep_gate_up)
        call.start_prefetch(self._moonep_down)

        # The bridge returns ordering markers, not weights: each expert GEMM
        # consumes one so the weight-gradient reduction runs after both of them.
        gate_up_marker, down_marker = moonep_weights_for_step(
            self.gate_up_proj, self.down_proj, self._moonep_gate_up, self._moonep_down, call
        )

        call.await_prefetch(self._moonep_gate_up)
        fc1 = moonep_gate_up_gmm(call, dispatched, gate_up_marker, hidden_states, self._moonep_gate_up)
        activated = _activate_expert_hidden(self, fc1)

        call.await_prefetch(self._moonep_down)
        output = moonep_down_and_combine(
            call, activated, down_marker, route_weights_nvs, self._moonep_down, self._moonep_gate_up
        )
        return output[:live_tokens].view(*hidden_shape)

    return moonep_experts_forward


def parallelize_moonep_model(modules, ep_modules, ep_mesh, plan, edp_mesh=None):
    """Parallelize every matched expert module with one shared MoonEP runtime.

    The runtime is created on the first module, reused for the rest, attached to
    the model root, and finalized once every symmetric allocation is registered.
    ``ep_modules`` must already carry ``num_global_experts``.
    """
    from fsdp_turbo.distributed.expert_parallel.moonep.gate_debug import attach_gate_debug

    attach_gate_debug(modules, ep_modules, ep_mesh.get_group(), plan)
    runtime = None
    for module in ep_modules:
        if runtime is None:
            runtime = MoonEPRuntime(
                ep_mesh.get_group(),
                ep_mesh,
                module.num_global_experts,
                plan.moonep_config,
                # The other dispatchers get this from expert FSDP's reduce-scatter,
                # which MoonEP replaces, so it has to be applied to the home-rank
                # reduction instead.
                gradient_divide_factor=plan.gradient_divide_factor,
                # Candidate (C-3): 1-D ``edp`` mesh, the replicate axis of fused
                # HSDP's ``edp_efsdp``. None keeps the verified EP==world path.
                edp_group=edp_mesh.get_group() if edp_mesh is not None else None,
            )
            object.__setattr__(modules, "_moonep_runtime", runtime)
        elif runtime.num_experts != module.num_global_experts:
            raise RuntimeError(
                "All MoonEP modules sharing a runtime must have the same number of experts; "
                f"expected {runtime.num_experts}, got {module.num_global_experts}."
            )
        parallelize_moonep_module(module, runtime, ep_mesh, fixed_router=plan.fixed_router)
    if runtime is not None:
        # Every symmetric allocation is now registered, so the backend can bring
        # up its communication buffer and (on Ascend) its symmetric heap. This
        # must happen before the first forward.
        runtime.finalize()
    return modules

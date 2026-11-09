# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Opt-in Kimi routing diagnostics and synthetic load for EP benchmarks.

Both entry points are default-off. With no ``gate._router_debug``, or with
``apply_custom_router`` / ``record_load_balance`` left false, ``custom_routing``
returns ``None`` and ``record_load`` returns immediately, so Kimi's real top-k
path is untouched. They serve every EP dispatcher, not only MoonEP.
"""

import logging
import math

import torch
import torch.distributed as dist

from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


def _exact_counts(tokens, experts, top_k, target):
    if experts == top_k:
        return [tokens] * experts
    total = tokens * top_k
    # The discrete maximum is at least ceil(mean), even for target=0.
    peak = min(tokens, max(math.ceil(total / experts), round(total / experts * (1 + target))))
    base, remainder = divmod(total - peak, experts - 1)
    return [peak] + [base + (index < remainder) for index in range(experts - 1)]


def _synthetic_indices(gate, tokens, target, mode, device):
    experts, top_k = gate.num_experts, gate.top_k
    target = min(target, (experts - top_k) / top_k)
    if mode == "exact":
        quota = torch.tensor(_exact_counts(tokens, experts, top_k, target), device=device)
        selected = torch.empty((tokens, top_k), dtype=torch.int64, device=device)
        for row in range(tokens):
            chosen = torch.argsort(quota, descending=True, stable=True)[:top_k]
            selected[row] = chosen
            quota[chosen] -= 1
        return selected

    layer = int(getattr(gate, "_router_debug_layer", 0))
    generator = torch.Generator(device=device).manual_seed(1701 + layer)
    normal = torch.randn(experts, generator=generator, device=device)

    def sample(sigma, rows, rank=0):
        # Keep a positive mass even at the theoretical load limit.
        weights = torch.softmax(normal * sigma, dim=0).clamp_min(1e-30)
        sampler = torch.Generator(device=device).manual_seed(1907 + layer * 100003 + rank)
        return torch.multinomial(weights.expand(rows, -1), top_k, replacement=False, generator=sampler)

    calibration_rows = min(max(tokens, 1), 4096)
    low, high, best_sigma, best_error = 0.0, 128.0, 0.0, float("inf")
    for _ in range(20):
        sigma = (low + high) / 2
        indices = sample(sigma, calibration_rows)
        counts = torch.bincount(indices.flatten(), minlength=experts).float()
        maxvio = (counts.max() / counts.mean() - 1).item()
        error = abs(maxvio - target)
        if error < best_error:
            best_sigma, best_error = sigma, error
        if error <= max(0.005, target * 0.02):
            break
        if maxvio < target:
            low = sigma
        else:
            high = sigma
    rank = dist.get_rank() if dist.is_initialized() else 0
    return sample(best_sigma, tokens, rank)


def _router_debug(gate):
    return getattr(gate, "_router_debug", None)


def custom_routing(gate, hidden_states):
    """Return synthetic routing only when explicitly enabled for benchmarking."""
    debug = _router_debug(gate)
    if debug is None or not debug.apply_custom_router:
        return None
    target = float(debug.target_max_vio)
    if not math.isfinite(target) or target < 0:
        raise ValueError("apply_custom_router requires a finite target_max_vio >= 0.")
    mode = debug.router_mode
    if mode not in ("exact", "lognormal"):
        raise ValueError("router_mode must be exact or lognormal.")
    if not 0 < gate.top_k <= gate.num_experts:
        raise ValueError("Synthetic routing requires 0 < top_k <= num_experts.")
    tokens = hidden_states.numel() // hidden_states.shape[-1]
    indices = _synthetic_indices(gate, tokens, target, mode, hidden_states.device)
    weights = torch.full(
        indices.shape,
        gate.routed_scaling_factor / gate.top_k,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    if not getattr(gate, "_router_debug_warned", False):
        print_rank(logger.warning, "Synthetic Kimi routing enabled for benchmarking; real router scores are bypassed.")
        gate._router_debug_warned = True
    return indices, weights


def record_load(gate, indices):
    """Log local counts and EP-reduced global maxvio, excluding static padding."""
    debug = _router_debug(gate)
    if debug is None or not debug.record_load_balance:
        return
    counts = torch.bincount(indices.flatten().long(), minlength=gate.num_experts)
    rank = dist.get_rank() if dist.is_initialized() else 0
    layer = getattr(gate, "_router_debug_layer", "unknown")
    print_rank(logger.info, f"[Layer {layer}] tokens_per_expert={counts.tolist()}", ranks=rank)
    group = getattr(gate, "_expert_parallel_group", None)
    if dist.is_initialized() and group is not None:
        dist.all_reduce(counts, group=group)
    total = counts.sum().item()
    maxvio = counts.max().item() * gate.num_experts / total - 1 if total else 0.0
    print_rank(logger.info, f"[Layer {layer}] global maxvio={maxvio:.6f}")

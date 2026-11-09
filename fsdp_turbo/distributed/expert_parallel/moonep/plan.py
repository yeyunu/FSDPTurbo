# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Static-shape helpers for MoonEP's group layout and plan invariants.

Notation: ``S`` tokens/rank, ``K`` top-k, ``E`` routed experts, ``R`` EP ranks,
``epn = E/R``, ``B = epn`` prefetch slots/rank, ``NvS`` dispatched rows/rank.
"""

from __future__ import annotations

from typing import Any

import torch


_TOP_K_CONFIG_ATTRS = ("num_experts_per_token", "num_experts_per_tok")


def infer_moe_top_k(model: Any) -> int | None:
    """Read the routed top-k from a loaded MoE model, if the config exposes it.

    MoonEP's NPU backend must lock ``K`` before the symmetric heap is created,
    which is before the first expert forward. The value lives on the model
    config (HuggingFace ``num_experts_per_tok`` or Kimi ``num_experts_per_token``),
    not in the EP plan.
    """
    configs = []
    root = getattr(model, "config", None)
    if root is not None:
        configs.append(root)
        text = getattr(root, "text_config", None)
        if text is not None:
            configs.append(text)
    for cfg in configs:
        for name in _TOP_K_CONFIG_ATTRS:
            value = getattr(cfg, name, None)
            if value is not None and int(value) > 0:
                return int(value)
    for module in model.modules():
        top_k = getattr(module, "top_k", None)
        num_experts = getattr(module, "num_experts", None)
        if isinstance(top_k, int) and top_k > 0 and isinstance(num_experts, int) and num_experts > 0:
            return top_k
    return None


def fill_moonep_static_shape(
    plan: Any,
    model: Any,
    *,
    batch_size: int | None = None,
    max_seq_length: int | None = None,
    ulysses_size: int = 1,
) -> None:
    """Derive the reserved ``S`` / ``K`` so the NPU heap can initialize.

    Neither field is meant to be typed into the EP plan. ``tokens_per_rank`` is
    ``batch_size * max_seq_length / ulysses_size``; ``top_k`` comes from the
    model. An explicit override is left alone. A still-missing field stays
    unset so ``MoonEPRuntime.finalize`` can say derivation failed.
    """
    from fsdp_turbo.fsdp_turbo_config import MoonEPConfig

    moon = plan.moonep_config or MoonEPConfig()
    if moon.tokens_per_rank is None and batch_size and max_seq_length:
        if ulysses_size < 1:
            raise ValueError(f"MoonEP ulysses_size must be positive, got {ulysses_size}.")
        tokens = batch_size * max_seq_length
        if tokens % ulysses_size:
            raise RuntimeError(
                "MoonEP tokens_per_rank is batch_size * max_seq_length split across "
                f"Ulysses ranks; {tokens} is not divisible by ulysses_size={ulysses_size}."
            )
        moon.tokens_per_rank = tokens // ulysses_size
    if moon.top_k is None:
        moon.top_k = infer_moe_top_k(model)
    plan.moonep_config = moon


def compact_group_ends(
    cu_seqlens: torch.Tensor,
    rank: int,
    experts_per_rank: int,
    num_experts: int,
    prefetch_slots: int,
    num_rows: int,
) -> torch.Tensor:
    """Project MoonEP's ``[E+B]`` group ends onto the ``epn+B`` groups this rank computes.

    Valid because a rank only receives tokens for its own home experts and for
    the slots it prefetched, so every other group is empty and
    ``[home | slots]`` is already a monotone partition of the live prefix.
    Report Appendix E proves ``B = E/R`` always suffices for that.

    The last offset is extended to ``num_rows`` so the GEMM covers the whole
    static buffer without a host read, which folds MoonEP's reserved tail into
    the final slot group.
    """
    start = rank * experts_per_rank
    ends = torch.cat(
        (
            cu_seqlens[start : start + experts_per_rank],
            cu_seqlens[num_experts : num_experts + prefetch_slots],
        )
    ).clone()
    ends[-1] = num_rows
    return ends


def full_group_ends(cu_seqlens: torch.Tensor, num_rows: int) -> torch.Tensor:
    """MoonEP's own ``[E+B]`` group ends, used where the whole mapping is readable."""
    ends = cu_seqlens.clone()
    ends[-1] = num_rows
    return ends


def zero_static_tail_(rows: torch.Tensor, last_group_end: torch.Tensor) -> torch.Tensor:
    """Zero the rows past the last live group, in place.

    MoonEP leaves its static tail undefined. The ``[E+B]`` group set has no
    spare weight row to absorb it, so it would contaminate the last prefetch
    slot's weight gradient.
    """
    active = torch.arange(rows.shape[0], device=rows.device) < last_group_end
    return rows.mul_(active.unsqueeze(-1))


def verify_plan_locality(cu_seqlens: torch.Tensor, rank: int, experts_per_rank: int, num_experts: int) -> None:
    """Check that no tokens land outside this rank's group blocks.

    The invariant :func:`compact_group_ends` relies on. Checked on the host
    rather than with ``torch._assert_async``, whose failure poisons the CUDA
    context and surfaces as a crash with no message.
    """
    start = rank * experts_per_rank
    ends = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    leading = ends[start - 1] if start > 0 else 0
    local_end = ends[start + experts_per_rank - 1]
    experts_end = ends[num_experts - 1]
    if leading == 0 and experts_end == local_end:
        return

    stray = [
        group
        for group in range(num_experts)
        if not start <= group < start + experts_per_rank and ends[group] > (ends[group - 1] if group else 0)
    ]
    raise RuntimeError(
        "MoonEP planned tokens for expert groups this rank does not own: groups "
        f"{stray[:16]}{'...' if len(stray) > 16 else ''}. Every routed token must land "
        f"on one of this rank's [{start}, {start + experts_per_rank}) home experts or in "
        f"one of its {cu_seqlens.numel() - num_experts} prefetch slots. This is a planner "
        "problem, not a configuration one: on Ascend the forward's weight operand maps "
        "only those groups, so there is nothing to fall back to."
    )


def verify_routing_range(top_k_index: torch.Tensor, num_experts: int) -> None:
    """Check that every routed expert id is in ``[0, E)``.

    An out-of-range id corrupts the planning kernel in ways that are hard to
    attribute, so it is worth one synchronization during bring-up.
    """
    lowest = int(top_k_index.min())
    highest = int(top_k_index.max())
    if 0 <= lowest and highest < num_experts:
        return
    raise RuntimeError(f"MoonEP expert ids must be in [0, {num_experts}); got [{lowest}, {highest}].")

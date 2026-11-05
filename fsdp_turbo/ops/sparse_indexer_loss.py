# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Independent sparse indexer gradient and its detached attention teacher."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from fsdp_turbo.ops.registry import dispatch_op

_terms: ContextVar[list[torch.Tensor] | None] = ContextVar("slig_terms", default=None)


@contextmanager
def collect_indexer_gradients() -> Iterator[list[torch.Tensor]]:
    """Collect unnormalized gradient carriers during the original model forward."""
    terms = []
    token = _terms.set(terms)
    try:
        yield terms
    finally:
        _terms.reset(token)


@torch.no_grad()
def _teacher(query, kv, compressed, window, indices, sinks, scale):
    # Bound temporary [B,S,H,K] scores and [B,S,K,D] gathers for long contexts.
    batch = torch.arange(query.size(0), device=query.device)[:, None, None]
    parts = []

    def scores(q, keys, positions):
        values = keys[batch, positions.clamp_min(0).long()]
        logits = torch.einsum("bshd,bskd->bshk", q.float(), values.float()) * scale
        return logits.masked_fill_(positions.unsqueeze(2) < 0, -torch.inf)

    for q, win, selected in zip(query.split(32, 1), window.split(32, 1), indices.split(32, 1)):
        local = scores(q, kv, win)
        sparse = scores(q, compressed, selected)
        sink = sinks.float().view(1, 1, -1, 1).expand_as(local[..., :1])
        denom = torch.logsumexp(torch.cat((local, sparse, sink), dim=-1), dim=-1, keepdim=True)
        parts.append((sparse - denom).exp().sum(dim=2))
    return torch.cat(parts, dim=1)


def record_indexer_gradient(
    query: torch.Tensor,
    kv: torch.Tensor,
    compressed: torch.Tensor,
    window: torch.Tensor,
    indices: torch.Tensor,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    sinks: torch.Tensor,
    scale: float,
    ratio: int,
    fused: bool,
    query_mask: torch.Tensor | None,
) -> None:
    """Build the local gradient sum using cache-relative indices; replay never appends to the original loss."""
    if query_mask is not None:
        indices = indices.masked_fill(~query_mask.bool().unsqueeze(-1), -1)
    target = _teacher(query, kv, compressed, window, indices, sinks, scale)
    term = dispatch_op(
        "sparse_indexer_gradient",
        *inputs,
        indices,
        target,
        ratio,
        device_type=query.device.type if fused else "cpu",
    )
    terms = _terms.get()
    if terms is not None:
        terms.append(term)

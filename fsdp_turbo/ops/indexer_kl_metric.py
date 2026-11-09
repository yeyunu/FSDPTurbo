"""Detached SLIG loss reporting; this module never supplies a training gradient."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

from fsdp_turbo.ops.registry import dispatch_op


@dataclass
class IndexerKLMetrics:
    sums: list[torch.Tensor] = field(default_factory=list)
    scale: float | torch.Tensor = 1.0
    use_fused: bool = True

    def value(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.stack(self.sums).sum() * self.scale if self.sums else reference.new_zeros(())


_active_metrics: ContextVar[IndexerKLMetrics | None] = ContextVar("slig_metrics", default=None)


@contextmanager
def collect_indexer_kl_metrics(use_fused: bool = True) -> Iterator[IndexerKLMetrics]:
    """Collect detached per-layer scalar losses for one forward/backward pair."""
    values = IndexerKLMetrics(use_fused=use_fused)
    token = _active_metrics.set(values)
    try:
        yield values
    finally:
        _active_metrics.reset(token)


def current_indexer_kl_metrics() -> IndexerKLMetrics | None:
    """Return the active collector in forward, or None when logging is disabled."""
    return _active_metrics.get()


@torch.no_grad()
def indexer_kl_metric(target: torch.Tensor, probs: torch.Tensor, *, use_fused: bool = True) -> torch.Tensor:
    """Return summed KL; use_fused=False selects portable PyTorch on the input device."""
    if target.numel() == 0 and target.shape == probs.shape:
        return target.new_zeros((), dtype=torch.float32)
    if target.shape != probs.shape or target.device != probs.device:
        raise ValueError("KL target and index probabilities must have identical shape and device")
    if target.ndim == 4 and target.shape[-2] == 1:
        target, probs = target.squeeze(-2), probs.squeeze(-2)
    if target.ndim not in (2, 3) or target.numel() == 0:
        raise ValueError("Expected nonempty [B,S,1,K], [B,S,K], or [T,K] KL inputs")
    # Match the caller's F.normalize(cmp_softmax_l1, p=1, dim=-1), then
    # _compute_indexer_loss's logits weighting. The CANN scalar is a SUM.
    teacher = torch.nn.functional.normalize(target.detach().float(), p=1, dim=-1).contiguous()
    student = probs.detach().float().contiguous()
    total = dispatch_op("indexer_kl_sum", teacher, student, device_type=teacher.device.type if use_fused else "cpu")
    return total.reshape(())

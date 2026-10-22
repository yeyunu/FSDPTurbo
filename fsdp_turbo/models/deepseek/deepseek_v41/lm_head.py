"""Causal-LM output head with a consistent logits/loss interface."""

import torch
import torch.nn.functional as F

from fsdp_turbo.memory.chunk_loss.chunk_loss import chunk_loss, fixed_cross_entropy
from fsdp_turbo.ops.hf32_linear import hf32_linear

from .model import ParallelHead


def _calculate_head_loss(hidden_states, head_weight, head_bias, *, shift_labels, alpha, use_hf32):
    projection = hf32_linear if use_hf32 else F.linear
    logits = projection(hidden_states, head_weight).float()
    loss = fixed_cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shift_labels.reshape(-1),
        alpha=alpha,
    )
    return loss, logits


class ChunkedLMHead(ParallelHead):
    """Return ``(logits, loss)`` while preserving the head parameter path."""

    def __init__(self, head: ParallelHead, *, chunk_loss_size: int | None = None) -> None:
        with torch.device("meta"):
            super().__init__(head.vocab_size, head.dim, head.norm_eps, head.hc_eps)
        self.weight = head.weight
        self.compute_dtype = head.compute_dtype
        self.use_hf32 = head.use_hf32
        self.chunk_loss_size = chunk_loss_size
        self.train(head.training)

    # This training interface accepts shifted targets and returns (logits, loss).
    def forward(  # pylint: disable=arguments-differ
        self, hidden_states: torch.Tensor, *, shift_labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return full logits without targets, or chunked loss for shifted targets."""
        if shift_labels is None or self.chunk_loss_size is None:
            logits = super().forward(hidden_states, full_logits=True)
            loss = None
            if shift_labels is not None:
                loss = F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
            return logits, loss

        hidden_states = hidden_states[:, :-1].to(self.compute_dtype)
        head_weight = self.weight.to(self.compute_dtype)
        num_items = (shift_labels != -100).sum().clamp(min=1)
        loss_kwargs_chunks = [
            dict(shift_labels=labels, alpha=num_items, use_hf32=self.use_hf32)
            for labels in torch.split(shift_labels, self.chunk_loss_size, dim=1)
        ]
        loss = chunk_loss(
            hidden_states,
            head_weight,
            None,
            _calculate_head_loss,
            loss_kwargs_chunks,
            self.chunk_loss_size,
        )
        return None, loss

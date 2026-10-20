"""Shared sequence and dtype helpers for DeepSeek-V4.1."""

from contextlib import contextmanager
from itertools import accumulate

import torch


def _packed_sequence_lengths(total_tokens: int, cu_seqlens: torch.Tensor | None = None) -> list[int] | None:
    """Return logical document lengths from explicit packed sequence boundaries."""
    if cu_seqlens is None:
        return None
    boundaries = cu_seqlens.tolist()
    if len(boundaries) <= 2 or boundaries[0] != 0 or boundaries[-1] != total_tokens:
        return None
    return [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]


def _packed_freqs(freqs_cis: torch.Tensor, lengths: list[int], stride: int = 1):
    """Build RoPE frequencies with positions restarted for every packed document."""
    return torch.cat([freqs_cis[:length:stride] for length in lengths], dim=0)


def _sequence_slices(lengths: list[int]):
    boundaries = list(accumulate(lengths, initial=0))
    return zip(boundaries, boundaries[1:])


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch's default dtype, restoring it even if the body raises."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

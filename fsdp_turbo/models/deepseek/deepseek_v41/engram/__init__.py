"""Engram residual module and n-gram hashing for DeepSeek-V4.1."""

from .engram import (
    Engram,
    EngramLayout,
    NgramHashState,
    find_next_prime,
)

__all__ = [
    "Engram",
    "EngramLayout",
    "NgramHashState",
    "find_next_prime",
]

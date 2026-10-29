"""Shared cross-layer attention cache for DeepSeek-V4.1."""

import torch
from typing import NamedTuple


class ContextParallelAttentionState(NamedTuple):
    """Pass cross-layer CP tensors explicitly across activation-checkpoint boundaries.

    Each layer returns a new state; replay never reads a later layer's mutable
    cache. Floating tensors keep their autograd edges to the source layers.
    """

    compress_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    topk_idxs: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


class SharedAttentionRuntime:
    """What attention layers hand down the stack instead of recomputing. Layers run in order and
    every source writes before its consumers read, so one slot each is enough. Training resets the
    slots before every full-sequence forward so tensors cannot retain a previous autograd graph.
    Sources: compress_kv and index_k from kv_source_layers, topk_idxs from index_source_layers,
    candidates from candidate_source_layer.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None
        self.packed_candidates: list[torch.Tensor] | None = None
        self.candidate_block_indices: torch.Tensor | None = None
        self.candidate_block_length: torch.Tensor | None = None


shared_attn = SharedAttentionRuntime()

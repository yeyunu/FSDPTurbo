"""Shared cross-layer attention cache for DeepSeek-V4.1."""

from __future__ import annotations

from typing import Any

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


_SLOTS = (
    "compress_kv",
    "index_k",
    "topk_idxs",
    "candidates",
    "packed_candidates",
    "candidate_block_indices",
    "candidate_block_length",
)


class _ReplayWindow:
    """Private copy of the shared slots for one checkpoint replay.

    torch's non-reentrant checkpoint calls the context factory once during the
    original forward and re-enters the returned replay context on every
    recompute, so this object has to support repeated enter/exit: each entry
    reinstalls the captured state and each exit puts the live state back.
    """

    def __init__(self, runtime: "SharedAttentionRuntime", captured: dict[str, Any]):
        self._runtime = runtime
        self._captured = captured
        self._outer: dict[str, Any] | None = None

    def __enter__(self) -> "_ReplayWindow":
        self._outer = self._runtime.snapshot()
        self._runtime.restore(self._captured)
        return self

    def __exit__(self, *exc_info) -> bool:
        outer, self._outer = self._outer, None
        if outer is not None:
            self._runtime.restore(outer)
        return False


class SharedAttentionRuntime:
    """What attention layers hand down the stack instead of recomputing.

    Layers run in order and every source writes before its consumers read, so one slot each is
    enough. Training resets the slots before every full-sequence forward so tensors cannot retain
    a previous autograd graph. Sources: compress_kv and index_k from kv_source_layers, topk_idxs
    from index_source_layers, candidates from candidate_source_layer.

    Recompute interplay: a checkpoint replays a source layer's forward during backward. The slots
    then hold whatever the *last* forward pass published (a later source group), so a replay that
    merely stopped publishing would still let its consumers read the wrong version. Instead every
    checkpoint captures :meth:`snapshot` when it is wired (i.e. at the start of that module's
    forward, when the slots still belong to the preceding sources) and re-enters
    :meth:`replay_window` on each recompute: the replay reads and writes its own copy, and the
    live state is restored when it leaves, so earlier layers' backward keeps the tensors it was
    consuming. ``publish`` is the only write path so that every slot update can be isolated and
    returned in one place.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear every slot at the start of a full-sequence forward.

        Declared explicitly (rather than through ``_SLOTS``) so static analysers know the members.
        """
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None
        self.packed_candidates: list[torch.Tensor] | None = None
        self.candidate_block_indices: torch.Tensor | None = None
        self.candidate_block_length: torch.Tensor | None = None

    def snapshot(self) -> dict[str, Any]:
        """Return the current slot values (references, no tensor copies)."""
        return {slot: getattr(self, slot, None) for slot in _SLOTS}

    def restore(self, state: dict[str, Any]) -> None:
        """Install ``state`` (as produced by :meth:`snapshot`) into the slots."""
        for slot in _SLOTS:
            setattr(self, slot, state.get(slot))

    def replay_window(self, captured: dict[str, Any]) -> _ReplayWindow:
        """Context manager that isolates one checkpoint replay from the live slots."""
        return _ReplayWindow(self, captured)

    def publish(self, slot: str, value: Any) -> Any:
        """Store ``value`` in ``slot`` and return it.

        Returning the value lets a source use its own write (including a recomputed one) without
        re-reading the slot, which is what the replay needs.
        """
        setattr(self, slot, value)
        return value


shared_attn = SharedAttentionRuntime()

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Variable-length TND context parallelism for packed DeepSeek-V4 attention.

The stream is cut into ``cp`` contiguous, compression-LCM-aligned segments;
each rank keeps its segment as the query and gathers every touched document's
KV prefix with one differentiable all-gather per layer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_gather_tensor_autograd

from fsdp_turbo.distributed.context_parallel.loss import (
    cp_all_reduce_forward_backward,
    cp_fixed_cross_entropy,
    shift_labels_globally,
)
from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)

EXCHANGE_STYLES = ("full_prefix", "halo_prefix")


def active_lcm(compress_ratios) -> int:
    """Smallest common multiple of the active compression ratios (> 1)."""
    values = [int(ratio) for ratio in compress_ratios if int(ratio) > 1]
    if not values:
        return 1
    lcm = values[0]
    for value in values[1:]:
        lcm = lcm * value // math.gcd(lcm, value)
    return lcm


def variable_cuts(total_len: int, cp_size: int, lcm: int) -> list[int]:
    """``cp_size + 1`` cut points: contiguous, unpadded, ``lcm``-aligned."""
    if total_len % lcm:
        raise ValueError(f"packed TND length {total_len} is not a multiple of the compression LCM {lcm}")
    units = total_len // lcm
    cuts = [lcm * (k * units // cp_size) for k in range(cp_size + 1)]
    cuts[-1] = total_len
    return cuts


@dataclass(frozen=True)
class DocumentSegment:
    """One rank's query segment of one document, in document-local coordinates."""

    doc_id: int
    q_lo: int
    q_hi: int


def document_segments(cu_ends: list[int], cuts: list[int]) -> list[list[tuple[int, int, int]]]:
    """``doc_id -> [(rank, q_lo, q_hi), ...]`` in rank order."""
    segments = []
    for doc_start, doc_end in zip(cu_ends[:-1], cu_ends[1:]):
        per_doc = []
        for rank, (cut_lo, cut_hi) in enumerate(zip(cuts[:-1], cuts[1:])):
            lo, hi = max(doc_start, cut_lo), min(doc_end, cut_hi)
            if lo < hi:
                per_doc.append((rank, lo - doc_start, hi - doc_start))
        segments.append(per_doc)
    return segments


def local_segments(cu_ends: list[int], cuts: list[int], rank: int) -> list[DocumentSegment]:
    """Segments of ``rank`` in stream order."""
    segments = []
    for doc_id, per_doc in enumerate(document_segments(cu_ends, cuts)):
        for seg_rank, q_lo, q_hi in per_doc:
            if seg_rank == rank:
                segments.append(DocumentSegment(doc_id, q_lo, q_hi))
    return segments


def exchange_row_ranges(
    cu_ends: list[int], cuts: list[int], window: int, style: str, ratio: int
) -> dict[int, tuple[int, int]]:
    """``doc_id -> (row_lo, row_hi)``: rows any receiver of the document needs.

    Rows are ``ratio`` tokens; ``ratio=1`` is the raw stream. The interval is
    the union of every receiver's suffix-anchored buffer
    ``[max(0, q_lo - (window - 1)), q_hi)`` (full prefix: ``[0, q_hi)``).
    """
    if style not in EXCHANGE_STYLES:
        raise ValueError(f"unknown exchange style {style!r}")
    ranges = {}
    for doc_id, segments in enumerate(document_segments(cu_ends, cuts)):
        row_hi = max(q_hi for _, _, q_hi in segments) // ratio
        if ratio > 1 or style == "full_prefix":
            ranges[doc_id] = (0, row_hi)
        else:
            ranges[doc_id] = (min(max(0, q_lo - (window - 1)) for _, q_lo, _ in segments), row_hi)
    return ranges


def send_row_ranges(
    cu_ends: list[int],
    cuts: list[int],
    window: int,
    ratio: int,
    rank: int,
    *,
    prefix: bool = False,
) -> dict[int, tuple[int, int]]:
    """``doc_id -> (row_lo, row_hi)`` this rank contributes to the all-gather.

    Only later segments read a rank's rows, so a rank sends just the suffix a
    successor's window reaches (all rows for the compressed channel); the last
    segment sends nothing.
    """
    ranges = {}
    for doc_id, segments in enumerate(document_segments(cu_ends, cuts)):
        current = next((segment for segment in segments if segment[0] == rank), None)
        successors = [segment for segment in segments if segment[0] > rank]
        if current is None or not successors:
            continue
        _, q_lo, q_hi = current
        if prefix or ratio > 1:
            send_lo, send_hi = q_lo // ratio, q_hi // ratio
        else:
            send_lo = max(q_lo, min(max(0, lo - (window - 1)) for _, lo, _ in successors))
            send_hi = q_hi
        if send_lo < send_hi:
            ranges[doc_id] = (send_lo, send_hi)
    return ranges


def all_gather_var_len(local: torch.Tensor, group: dist.ProcessGroup, block_len: int) -> tuple[torch.Tensor, int]:
    """All-gather tensors of different lengths along dim 1, differentiable.

    ``block_len`` is the padded block width. The runtime precomputes it from the
    batch plan, which removes a blocking MAX all-reduce and a device sync from
    every layer.
    """
    local = local.contiguous()
    if dist.get_world_size(group) == 1:
        return local, local.shape[1]
    padded = F.pad(local, (0, 0, 0, block_len - local.shape[1]))
    return all_gather_tensor_autograd(padded, gather_dim=1, group=group), block_len


def shifted_targets(labels: torch.Tensor, cu_ends: list[int], cuts: list[int], rank: int) -> torch.Tensor:
    """Targets of ``rank``'s query segment under the global one-token shift."""
    shifted = shift_labels_globally(labels, cu_ends[1:-1])
    return shifted[:, cuts[rank] : cuts[rank + 1]]


@dataclass(frozen=True)
class TNDPlan:
    """Geometry of one packed TND batch. Identical on every CP rank.

    The model reads this object instead of individual runtime attributes, so a
    test double only needs a plan plus the two exchange methods.
    """

    docs: tuple[DocumentSegment, ...]
    cu_ends: tuple[int, ...]
    cuts: tuple[int, ...]
    local_start: int
    local_end: int
    segment_lengths: tuple[int, ...]
    exchange_style: str
    window: int
    rank: int
    world_size: int

    def row_ranges(self, ratio: int) -> dict[int, tuple[int, int]]:
        """Rows every receiver of each document needs (receive-side buffers)."""
        return exchange_row_ranges(list(self.cu_ends), list(self.cuts), self.window, self.exchange_style, ratio)

    def prefix_start(self, doc_id: int, ratio: int) -> int:
        """Global compressed-row offset of ``doc_id``."""
        return sum((end - start) // ratio for start, end in zip(self.cu_ends[:doc_id], self.cu_ends[1 : doc_id + 1]))

    def buffer_start(self, segment: DocumentSegment) -> int:
        """First raw row this document's local query reads."""
        if self.exchange_style == "full_prefix":
            return 0
        return max(0, segment.q_lo - (self.window - 1))

    def stream_block_len(self, ratio: int = 1) -> int:
        """Padded block width for a whole-stream all-gather of ``ratio``-row units."""
        widths = [
            sum(
                (segment.q_hi - segment.q_lo) // ratio
                for segment in local_segments(list(self.cu_ends), list(self.cuts), rank)
            )
            for rank in range(self.world_size)
        ]
        return max(widths, default=0)

    def stream_lengths(self, ratio: int = 1) -> tuple[int, ...]:
        """Per-rank width of a whole-stream all-gather (tokens or compressed rows)."""
        return tuple(length // ratio for length in self.segment_lengths)

    def send_block_len(self, ratio: int, *, prefix: bool) -> int:
        """Padded block width for a per-document row all-gather."""
        widths = [
            sum(
                high - low
                for low, high in send_row_ranges(
                    list(self.cu_ends), list(self.cuts), self.window, ratio, rank, prefix=prefix
                ).values()
            )
            for rank in range(self.world_size)
        ]
        return max(widths, default=0)

    def cu_seqlens(self, ratio: int) -> tuple[list[int], list[int], list[int]]:
        """Operator metadata: cumulative local query, raw KV and compressed KV rows."""
        cu_q, cu_kv, cu_cmp = [0], [0], [0]
        for segment in self.docs:
            cu_q.append(cu_q[-1] + segment.q_hi - segment.q_lo)
            cu_kv.append(cu_kv[-1] + segment.q_hi - self.buffer_start(segment))
            if ratio:
                cu_cmp.append(cu_cmp[-1] + segment.q_hi // ratio)
        return cu_q, cu_kv, cu_cmp


class TNDRuntime:
    """Per-rank runtime state for one packed TND batch."""

    def __init__(self, group: dist.ProcessGroup, exchange_style: str = "full_prefix"):
        if exchange_style not in EXCHANGE_STYLES:
            raise ValueError(
                f"kv_allgather_tnd exchange_style must be one of {EXCHANGE_STYLES}, got {exchange_style!r}"
            )
        self.group = group
        self.exchange_style = exchange_style
        self.window = 128
        self.lcm = 1
        self.plan: TNDPlan | None = None
        self.shared: dict = {}
        self._cu_seqlens: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
        self._window_positions: torch.Tensor | None = None

    def configure_model(self, modules) -> None:
        """Read the window size and compression ratios from the model's TND attention.

        ``modules`` is the list the model returned from
        ``set_tnd_context_parallel``; scanning the whole model would let an
        unrelated module (e.g. the vision tower's ``head_dim``) leak in.
        """
        ratios = []
        windows = set()
        for module in modules:
            ratio = getattr(module, "compress_ratio", None)
            if isinstance(ratio, int):
                ratios.append(ratio)
            window = getattr(module, "window_size", None)
            if isinstance(window, int):
                windows.add(window)
        if len(windows) > 1:
            raise ValueError(f"packed-TND attention layers must share one window size; got {sorted(windows)}")
        self.window = windows.pop() if windows else self.window
        self.lcm = active_lcm(ratios)

    def start_batch(self, cu_ends: list[int]) -> None:
        """Cut the complete packed stream and derive this rank's local plan."""
        rank, world_size = dist.get_rank(self.group), dist.get_world_size(self.group)
        total_len = cu_ends[-1]
        units = total_len // self.lcm
        if units < world_size:
            raise ValueError(
                f"cp size {world_size} exceeds the packed stream's {units} compression "
                "units; the empty-segment path is not implemented."
            )
        cuts = variable_cuts(total_len, world_size, self.lcm)
        self.plan = TNDPlan(
            docs=tuple(local_segments(cu_ends, cuts, rank)),
            cu_ends=tuple(cu_ends),
            cuts=tuple(cuts),
            local_start=cuts[rank],
            local_end=cuts[rank + 1],
            segment_lengths=tuple(cuts[index + 1] - cuts[index] for index in range(world_size)),
            exchange_style=self.exchange_style,
            window=self.window,
            rank=rank,
            world_size=world_size,
        )
        self.shared = {}
        self._cu_seqlens = {}
        self._window_positions = None

    def state_for(self, kind: str, ratio: int | None = None) -> dict:
        """Per-batch state shared between layers, keyed by purpose and ratio."""
        key = (kind, ratio) if ratio is not None else (kind,)
        return self.shared.setdefault(key, {})

    def split_batch(self, batch: dict) -> dict:
        """Keep labels complete; slice the model inputs to this rank's cut."""
        input_ids = batch["input_ids"]
        total_len = input_ids.shape[-1]
        cu_ends = [0, *batch["actual_seq_len"].reshape(-1).tolist()]
        if cu_ends[-1] != total_len:
            raise ValueError(f"actual_seq_len must finish at the packed length {total_len}; got {cu_ends[-1]}")
        # Image metadata is redundant (like cu_ends); the merge derives cuts locally.
        self.start_batch(cu_ends)
        plan = self.plan

        keep = slice(plan.local_start, plan.local_end)
        sliced = dict(batch)
        sliced["input_ids"] = input_ids[:, keep]
        sliced["attention_mask"] = batch["attention_mask"][:, keep]
        if "token_types" in batch:
            sliced["token_types"] = batch["token_types"][:, keep]
        # Positions are document-local and re-rotated; stream-wide ids do not apply.
        sliced["position_ids"] = None

        local_lengths = [
            min(doc_end, plan.local_end) - max(doc_start, plan.local_start)
            for doc_start, doc_end in zip(plan.cu_ends[:-1], plan.cu_ends[1:])
            if min(doc_end, plan.local_end) > max(doc_start, plan.local_start)
        ]
        sliced["actual_seq_len"] = torch.tensor(local_lengths, dtype=torch.int32).cumsum(0)
        return sliced

    def gather_stream(self, local: torch.Tensor, ratio: int = 1) -> torch.Tensor:
        """Gather the local pre-RoPE stream and return it in stream order.

        ``ratio`` is the width of one local row in tokens: ``1`` for the raw KV
        stream, the compress ratio for the compressed channel. Slicing must use
        the same unit as the local tensor, otherwise uneven cuts mix ranks.
        """
        plan = self.plan
        lengths = plan.stream_lengths(ratio)
        gathered, block_len = all_gather_var_len(local, self.group, max(lengths))
        parts = []
        offset = 0
        for length in lengths:
            parts.append(gathered[:, offset : offset + length])
            offset += block_len
        return torch.cat(parts, dim=1)

    def gather_document_rows(
        self, local_rows_by_doc: dict[int, torch.Tensor], ratio: int, kind: str
    ) -> dict[int, torch.Tensor]:
        """Assemble each local document's buffer from one narrow all-gather.

        Each rank sends only the rows a later segment can reach and the receiver
        adds its own segment locally. Every output keeps a zero-weight link to
        the gathered tensor so ranks consuming none still join the backward.
        """
        plan = self.plan
        prefix = kind == "cmp"
        doc_segments = document_segments(list(plan.cu_ends), list(plan.cuts))
        if prefix:
            receive_ranges = {
                doc_id: (0, max(q_hi for _, _, q_hi in segments) // ratio)
                for doc_id, segments in enumerate(doc_segments)
            }
        else:
            receive_ranges = plan.row_ranges(ratio)
        send_ranges = [
            send_row_ranges(list(plan.cu_ends), list(plan.cuts), plan.window, ratio, rank, prefix=prefix)
            for rank in range(plan.world_size)
        ]
        rank_layouts = []
        for rank in range(plan.world_size):
            layout = []
            for doc_id, segments in enumerate(doc_segments):
                send = send_ranges[rank].get(doc_id)
                if send is None:
                    continue
                segment = next(item for item in segments if item[0] == rank)
                local_lo = segment[1] // ratio if ratio > 1 else segment[1]
                layout.append((doc_id, send[1] - send[0], send[0], local_lo))
            rank_layouts.append(layout)

        sample = local_rows_by_doc[plan.docs[0].doc_id]
        pieces = [
            local_rows_by_doc[doc_id][:, send_lo - local_lo : send_lo - local_lo + width]
            for doc_id, width, send_lo, local_lo in rank_layouts[plan.rank]
        ]
        local_buffer = torch.cat(pieces, dim=1) if pieces else sample[:, :0]
        if any(rank_layouts):
            gathered, block_len = all_gather_var_len(
                local_buffer, self.group, plan.send_block_len(ratio, prefix=prefix)
            )
        else:
            gathered, block_len = None, 0

        outputs = {}
        for segment in plan.docs:
            doc_id = segment.doc_id
            row_lo, row_hi = receive_ranges[doc_id]
            buffer = sample.new_zeros(1, row_hi - row_lo, sample.shape[-1])
            local_lo = segment.q_lo // ratio if ratio > 1 else segment.q_lo
            local = local_rows_by_doc[doc_id]
            buffer[:, local_lo - row_lo : local_lo - row_lo + local.shape[1]] = local
            for rank, _, _ in doc_segments[doc_id]:
                if rank >= plan.rank or gathered is None:
                    continue
                within = 0
                for other_doc, width, send_lo, _ in rank_layouts[rank]:
                    if other_doc == doc_id:
                        buffer[:, send_lo - row_lo : send_lo - row_lo + width] = gathered[
                            :, block_len * rank + within : block_len * rank + within + width
                        ]
                        break
                    within += width
            outputs[doc_id] = buffer

        if gathered is not None:
            # Every rank must join the all-gather backward: first segments read
            # no gathered rows, so autograd would prune them and hang the rest.
            keep_alive = 0.0 * gathered.sum()
            outputs = {doc_id: buffer + keep_alive for doc_id, buffer in outputs.items()}

        return outputs

    def cu_seqlens(self, ratio: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Cached packed operator metadata for the current batch."""
        cached = self._cu_seqlens.get(ratio)
        if cached is None or cached[0].device != device:
            cu_q, cu_kv, cu_cmp = self.plan.cu_seqlens(ratio)
            cached = (
                torch.tensor(cu_q, dtype=torch.int32, device=device),
                torch.tensor(cu_kv, dtype=torch.int32, device=device),
                torch.tensor(cu_cmp, dtype=torch.int32, device=device) if ratio else None,
            )
            self._cu_seqlens[ratio] = cached
        return cached

    def window_positions(self, plan: TNDPlan, device) -> torch.Tensor:
        """Cached sliding-window index rows for every local document."""
        cached = self._window_positions
        if cached is None or cached.device != device:
            parts = []
            for segment in plan.docs:
                length = segment.q_hi - segment.q_lo
                buffer_start = plan.buffer_start(segment)
                row_end = (segment.q_lo + torch.arange(length, device=device)).unsqueeze(1)
                positions = (row_end - plan.window + 1).clamp(min=0) + torch.arange(plan.window, device=device)
                positions = torch.where(positions > row_end, -1, positions) - buffer_start
                parts.append(positions.int().unsqueeze(0))
            cached = torch.cat(parts, dim=1)
            self._window_positions = cached
        return cached

    def engram_hashes(self, hash_state, input_ids: torch.Tensor, engram_mask) -> torch.Tensor:
        """Hash ids for the local segments with a per-document n-gram halo.

        Engram hashing carries no gradient, so this exchange is a plain
        all-gather of ``[input_ids, valid_mask]``; the receiver rebuilds the
        complete stream and slices each document's halo.
        """
        plan = self.plan
        mask = engram_mask if engram_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        packed = torch.stack([input_ids, mask.to(torch.int64)], dim=1)  # [1, 2, T_local]
        if plan.world_size > 1:
            # Same padded width the differentiable stream gather uses.
            block_len = plan.stream_block_len()
            padded = F.pad(packed, (0, block_len - packed.shape[2]))
            gathered = [torch.empty_like(padded) for _ in range(plan.world_size)]
            dist.all_gather(gathered, padded, group=self.group)
        else:
            gathered = [packed]
        ids = torch.cat([block[:, 0, :length] for block, length in zip(gathered, plan.segment_lengths)], dim=1)
        valid = torch.cat(
            [block[:, 1, :length].bool() for block, length in zip(gathered, plan.segment_lengths)],
            dim=1,
        )

        halo_size = hash_state.layout.max_ngram_size - 1
        outputs = []
        for segment in plan.docs:
            doc_start = plan.cu_ends[segment.doc_id]
            local_start = doc_start + segment.q_lo
            halo_lo = max(0, segment.q_lo - halo_size)
            local_mask = valid[:, local_start : doc_start + segment.q_hi] if engram_mask is not None else None
            outputs.append(
                hash_state(
                    ids[:, local_start : doc_start + segment.q_hi],
                    token_mask=local_mask,
                    halo_ids=ids[:, doc_start + halo_lo : local_start],
                    halo_mask=(valid[:, doc_start + halo_lo : local_start] if engram_mask is not None else None),
                    q_lo=segment.q_lo,
                )
            )
        return torch.cat(outputs, dim=1)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Globally shifted, document-masked, globally normalised causal-LM loss."""
        plan = self.plan
        targets = shifted_targets(labels, list(plan.cu_ends), list(plan.cuts), plan.rank)
        return cp_fixed_cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            targets.reshape(-1).to(logits.device),
            group=self.group,
        )

    def reduce_loss(self, local_loss: torch.Tensor) -> torch.Tensor:
        """Sum a per-rank loss share over the CP domain and compensate the gradient.

        The backward is scaled by the group size because FSDP averages gradients
        across the CP ranks that share a shard group; without it the CP arms would
        train with a 1/cp-scaled gradient while the LM loss is unscaled.
        """
        return cp_all_reduce_forward_backward(local_loss, self.group)


def _batch_layout(batch) -> str | None:
    """Return a batch's ``sfa_layout``; collation may wrap it in a list."""
    layout = batch.get("sfa_layout") if isinstance(batch, Mapping) else None
    if isinstance(layout, (list, tuple)):
        layout = layout[0] if layout else None
    return layout


def is_tnd_batch(batch) -> bool:
    """Whether ``batch`` is a packed-TND batch."""
    return _batch_layout(batch) == "TND"


def _runtime_of(model):
    """Find the packed-TND runtime on ``model`` or one of its wrappers."""
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        runtime = getattr(model, "tnd_context_parallel", None)
        if runtime is not None:
            return runtime
        model = getattr(model, "model", None) or getattr(model, "module", None)
    return None


def install_tnd_context_parallel(model, cp_group, cp_plan=None):
    """Attach the packed-TND runtime to ``model`` and return the same model.

    The model supplies ``set_tnd_context_parallel(runtime)`` to wire the runtime
    into its attention layers and return them; this module stays model-agnostic
    and configures the runtime from that set.
    """
    if getattr(cp_plan, "kv_allgather_function_patches", None):
        raise ValueError("packed-TND CP is native; remove kv_allgather_function_patches.")
    configure = getattr(model, "set_tnd_context_parallel", None)
    if not callable(configure):
        raise RuntimeError(
            "packed-TND CP requires a model built with context_parallel=True "
            "(its adapter must expose set_tnd_context_parallel)."
        )
    runtime = TNDRuntime(cp_group, getattr(cp_plan, "tnd_exchange_style", "full_prefix"))
    driven = configure(runtime)
    if not driven:
        raise RuntimeError("packed-TND CP wired no attention layers.")
    runtime.configure_model(driven)
    model.tnd_context_parallel = runtime
    print_rank(
        logger.info,
        f"[Context Parallel/KVAllGather] Installed packed TND "
        f"(cp={dist.get_world_size(cp_group) if cp_group is not None else 1}, "
        f"window={runtime.window}, lcm={runtime.lcm}, style={runtime.exchange_style}).",
    )
    return model


def split_tnd_batch(model, batch):
    """Return the packed-TND split of ``batch``, or ``None`` when it is not TND."""
    if not is_tnd_batch(batch):
        return None
    runtime = _runtime_of(model)
    if runtime is None:
        raise RuntimeError("packed-TND batch has no installed runtime.")
    return runtime.split_batch(batch)

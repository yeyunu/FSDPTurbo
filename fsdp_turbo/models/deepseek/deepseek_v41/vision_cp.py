# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Packed-TND image merge: each rank writes only its local intersection.

Image metadata is redundant on every rank (like ``cu_ends``), so the plan needs
no exchange. Every rank still calls the vision tower once per step, and without
a runtime the vendored merge runs unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .image_processor import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
)
from . import model as reference_model


@dataclass(frozen=True)
class LocalImage:
    """One image's intersection with one rank's TND segment."""

    image_id: int  # global image order (position in the records list)
    lo: int  # intersection lower bound, local segment coordinates
    hi: int  # intersection upper bound, local segment coordinates
    owner: bool  # global span start inside this rank's cut
    row_lo: int  # first IMAGE row of the intersection, in image row order
    n_rows_local: int  # IMAGE rows inside the intersection
    types_slice: torch.Tensor  # the rank's view of the span types


def local_image_plan(images: list, cuts: list[int], rank: int) -> list[LocalImage]:
    """Intersections of ``images`` with rank ``rank``'s segment, zero communication."""
    cut_lo, cut_hi = cuts[rank], cuts[rank + 1]
    plan = []
    for image_id, image in enumerate(images):
        span_lo = int(image.start)
        span_hi = span_lo + int(image.types.numel())
        global_lo, global_hi = max(span_lo, cut_lo), min(span_hi, cut_hi)
        if global_lo >= global_hi:
            continue
        offset = global_lo - span_lo
        width = global_hi - global_lo
        plan.append(
            LocalImage(
                image_id=image_id,
                lo=global_lo - cut_lo,
                hi=global_hi - cut_lo,
                owner=cut_lo <= span_lo < cut_hi,
                row_lo=int((image.types[:offset] == IMAGE).sum()),
                n_rows_local=int((image.types[offset : offset + width] == IMAGE).sum()),
                types_slice=image.types[offset : offset + width],
            )
        )
    return plan


def check_span_legality(images: list, cu_ends: list[int]) -> None:
    """A span must live inside exactly one document (vendored semantics)."""
    for image_id, image in enumerate(images):
        span_lo = int(image.start)
        span_hi = span_lo + int(image.types.numel())
        inside = any(cu_ends[doc] <= span_lo and span_hi <= cu_ends[doc + 1] for doc in range(len(cu_ends) - 1))
        if not inside:
            raise ValueError(
                f"image {image_id} span [{span_lo}, {span_hi}) crosses a document boundary (cu_ends={cu_ends})."
            )


def _dummy_vision_call(net, h: torch.Tensor, records: list) -> None:
    """Call the vision tower once per rank, even without local images.

    A shrunken gradient set breaks the FSDP reduce-scatter, so image-free ranks
    reuse the first record's grid and keep the features and delimiter embeddings
    alive with an exactly-zero term.
    """
    if records:
        image = records[0]
        features = net.aligner(
            net.vision([image.patches.to(h.device)], [(image.n_vit_h, image.n_vit_w)]),
            [(image.n_vit_h, image.n_vit_w)],
        )
    else:
        # No images anywhere: every rank takes the same dummy branch.
        grid = net.aligner.downsample_ratio
        patch = net.vision.patch_size
        dtype = net.vision.patch_embed.proj.weight.dtype
        dummy = h.new_zeros((grid * grid, 3, patch, patch), dtype=dtype)
        features = net.aligner(net.vision([dummy], [(grid, grid)]), [(grid, grid)])
    zero = sum(feature.sum() for feature in features)
    zero = zero + net.image_start.sum() + net.image_end.sum() + net.image_newline.sum()
    h.add_(zero.to(h.dtype).mul(0.0))


def merge_image_embeddings_tnd(images, h: torch.Tensor, net, runtime) -> torch.Tensor:
    """Merge image spans locally (decision 11): no exchange, in-place writes.

    Only the local plan may write; image-free ranks keep the single dummy vision
    call and never run the vendored merge, which writes at global offsets.
    """
    records = [image for sample in images for image in (sample or ())]
    if records:
        check_span_legality(records, runtime.plan.cu_ends)
    plan = local_image_plan(records, runtime.plan.cuts, runtime.plan.rank)
    if not plan:
        # Deliberately NOT an early return: an image-free rank must still run
        # the tower and span embeddings or its reduce-scatter count differs.
        _dummy_vision_call(net, h, records)
        return h

    patches = [records[item.image_id].patches.to(h.device) for item in plan]
    grid_sizes = [(records[item.image_id].n_vit_h, records[item.image_id].n_vit_w) for item in plan]
    aligned_features = net.aligner(net.vision(patches, grid_sizes), grid_sizes)
    # Every rank keeps the tower and span embeddings in the graph: a rank with
    # no IMAGE row would otherwise post fewer reduces and stall HCCL.
    keep_alive = sum(feature.sum() for feature in aligned_features)
    keep_alive = keep_alive + net.image_start.sum() + net.image_end.sum() + net.image_newline.sum()
    h.add_(keep_alive.to(h.dtype).mul(0.0))
    rows_of = {item.image_id: feats for item, feats in zip(plan, aligned_features)}
    for item in plan:
        span = h[0, item.lo : item.hi]
        span[item.types_slice == IMAGE_START] = net.image_start.to(h.dtype)
        span[item.types_slice == IMAGE_END] = net.image_end.to(h.dtype)
        span[item.types_slice == IMAGE_NEW_LINE] = net.image_newline.to(h.dtype)
        if item.n_rows_local:
            rows = rows_of[item.image_id][item.row_lo : item.row_lo + item.n_rows_local]
            span[item.types_slice == IMAGE] = rows.to(h.dtype)
    return h


class TNDVisionTransformer(reference_model.Transformer):
    """Transformer whose image merge understands TND context-parallel segments."""

    def merge_image_embeddings(self, images, h: torch.Tensor):
        runtime = getattr(self, "cp_runtime", None)
        if runtime is None:
            return super().merge_image_embeddings(images, h)
        return merge_image_embeddings_tnd(images, h, self, runtime)

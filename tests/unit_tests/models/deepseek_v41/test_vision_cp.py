# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CP merge of image spans: planning, equivalence, and gradient wiring.

Pins the per-rank image plan, the merge reproducing the full-stream result
(split, mixed grids, edge images), and the D23 keep-alive that keeps image-free
ranks in the tower's autograd graph.
"""

import types

import torch

from fsdp_turbo.distributed.context_parallel.kv_allgather.tnd import TNDPlan
from fsdp_turbo.models.deepseek.deepseek_v41 import model as reference_model
from fsdp_turbo.models.deepseek.deepseek_v41.image_processor import image_token_types
from fsdp_turbo.models.deepseek.deepseek_v41.vision_cp import (
    TNDVisionTransformer,
    local_image_plan,
)

_TINY_ARGS = dict(
    dim=64,
    n_layers=2,
    vocab_size=256,
    n_heads=4,
    q_lora_rank=16,
    head_dim=16,
    rope_head_dim=8,
    o_groups=2,
    o_lora_rank=8,
    moe_inter_dim=32,
    n_routed_experts=4,
    n_activated_experts=1,
    window_size=8,
    compress_ratios=(0, 0, 0),
    kv_source_layers=(),
    index_source_layers=(),
    n_mtp_layers=0,
    dspark_block_size=0,
    engram_layer_ids=(),
    engram_num_embeddings=(),
    vision_n_layers=1,
    vision_dim=32,
    vision_n_heads=4,
    vision_inter_dim=32,
    vision_patch_size=4,
    vision_downsample_ratio=1,
    vision_max_n_token=64,
    max_seq_len=64,
)


def _record(start, rows=2, cols=2):
    """Minimal image record for the plan-only cases."""
    return types.SimpleNamespace(start=start, types=image_token_types(rows, cols))


def _image_record(start, image_id):
    """One 2x2 image at *start*: patches, ViT grid and the 8-token span."""
    return types.SimpleNamespace(
        start=start,
        patches=torch.randn(4, 3, 4, 4),
        n_vit_h=2,
        n_vit_w=2,
        types=image_token_types(2, 2),
        image_id=image_id,
    )


def _runtime_plan(cuts, cu_ends, rank):
    return TNDPlan(
        docs=(),
        cu_ends=tuple(cu_ends),
        cuts=tuple(cuts),
        local_start=cuts[rank],
        local_end=cuts[rank + 1],
        segment_lengths=tuple(cuts[index + 1] - cuts[index] for index in range(len(cuts) - 1)),
        exchange_style="full_prefix",
        window=128,
        rank=rank,
        world_size=len(cuts) - 1,
    )


def _merge_as_rank(net, images, h_local, plan):
    net.cp_runtime = types.SimpleNamespace(plan=plan)
    try:
        return net.merge_image_embeddings(images, h_local)
    finally:
        net.cp_runtime = None


# --- D23 regressions: every rank must keep the tower and the span embeddings ---


def test_delimiter_only_rank_keeps_the_vision_tower_in_the_graph():
    """A rank with only span delimiters must still backprop through the tower.

    Otherwise its FSDP reduce set differs from its peers' and the first backward
    hangs on an HCCL timeout; the keep-alive must leave ``h`` bit-identical.
    """

    net = TNDVisionTransformer(reference_model.ModelArgs(**_TINY_ARGS))
    total, cut = 32, 18
    cu_ends = [0, total]
    cuts = [0, cut, total]

    torch.manual_seed(11)
    # A 2x2 span cut at 18 leaves rank 1 only NEW_LINE/END, no IMAGE row.
    span = _image_record(12, 0)
    records = [span]
    plan_rank1 = local_image_plan(records, cuts, 1)
    assert plan_rank1 and sum(item.n_rows_local for item in plan_rank1) == 0

    # ``h`` is the non-leaf embedding output; the leaf only carries the gradient.
    leaf = torch.randn(1, total - cut, net.layers[0].attn.dim, dtype=torch.bfloat16, requires_grad=True)
    h = leaf * 1.0
    before = h.detach().clone()
    merged = _merge_as_rank(net, [records], h, _runtime_plan(cuts, cu_ends, 1))
    span_lo, span_hi = plan_rank1[0].lo, plan_rank1[0].hi
    assert torch.equal(merged.detach()[:, :span_lo], before[:, :span_lo])
    assert torch.equal(merged.detach()[:, span_hi:], before[:, span_hi:])

    merged.sum().backward()
    tower_grads = [param.grad for name, param in net.named_parameters() if name.startswith(("vision.", "aligner."))]
    assert tower_grads, "tower parameters not found"
    assert all(grad is not None for grad in tower_grads), "tower lost its gradient path"
    assert all(torch.count_nonzero(grad) == 0 for grad in tower_grads), "gradient must be exactly zero"

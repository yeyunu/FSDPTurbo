# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Packed-TND geometry: cuts, document segments and the exchange row ranges.

One coarse case over every layout the acceptance data uses. The finer-grained
originals (variable_cuts rejects, shifted_targets masking, send-side ranges,
runtime wiring) live in ``agent-workspace/dsv4-tnd-cp/50-verify/unit_tests``.
"""

import pytest
import torch

from fsdp_turbo.distributed.context_parallel.kv_allgather.tnd import (
    document_segments,
    exchange_row_ranges,
    variable_cuts,
)

LAYOUTS = {
    "single_long": [64],
    "uneven_total": [70],
    "multidoc": [30, 34, 16],
    "multidoc_uneven": [30, 34, 14],
    "doc_at_cut": [20, 20, 40],
    "short_docs": [2] * 20,
    "degenerate": [6],
}
WINDOW = 4


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_geometry_covers_the_stream_for_every_layout(name):
    lengths = LAYOUTS[name]
    cu_ends = [0, *list(torch.tensor(lengths).cumsum(0).tolist())]
    total = cu_ends[-1]
    for cp in (2, 4, 8):
        if total // 2 < cp:
            continue
        cuts = variable_cuts(total, cp, 2)
        # Cuts tile the stream, stay lcm-aligned, and match an even split within a unit.
        assert cuts[0] == 0 and cuts[-1] == total
        assert all(cut % 2 == 0 for cut in cuts)
        widths = [cuts[i + 1] - cuts[i] for i in range(cp)]
        assert max(widths) - min(widths) <= 2

        segments = document_segments(cu_ends, cuts)
        full = exchange_row_ranges(cu_ends, cuts, WINDOW, "full_prefix", 1)
        halo = exchange_row_ranges(cu_ends, cuts, WINDOW, "halo_prefix", 1)
        for doc_id, per_doc in enumerate(segments):
            doc_len = cu_ends[doc_id + 1] - cu_ends[doc_id]
            # Segments of one document tile it without gaps.
            assert per_doc[0][1] == 0 and per_doc[-1][2] == doc_len
            for (_, _, previous_end), (_, next_start, _) in zip(per_doc, per_doc[1:]):
                assert previous_end == next_start
            # full_prefix starts at row 0; halo starts at the receiver's window.
            assert full[doc_id] == (0, doc_len)
            lo, hi = halo[doc_id]
            assert hi == doc_len
            for _, q_lo, q_hi in per_doc:
                assert lo <= max(0, q_lo - (WINDOW - 1))
                assert hi >= q_hi


def test_tnd_cp_rejects_attention_recompute():
    """packed-TND shares one KV-source state slot per compression ratio.

    A replay would read the state a later source layer overwrote, so the
    combination is rejected at config time instead of producing wrong gradients.
    """
    from fsdp_turbo.fsdp_turbo_config import DataConfig, FSDPTurboConfig, MemoryConfig

    with pytest.raises(ValueError, match="memory.recompute"):
        FSDPTurboConfig(data=DataConfig(sfa_layout="TND"), memory=MemoryConfig(recompute=True))


def test_tnd_cp_rejects_chunk_loss():
    """packed-TND loss is shifted and normalised over the whole packed stream.

    The chunked head normalises each chunk locally, so the two loss paths cannot
    describe the same stream; the combination is rejected at config time.
    """
    from fsdp_turbo.fsdp_turbo_config import DataConfig, FSDPTurboConfig, MemoryConfig

    with pytest.raises(ValueError, match="memory.chunk_loss"):
        FSDPTurboConfig(data=DataConfig(sfa_layout="TND"), memory=MemoryConfig(chunk_loss=True))

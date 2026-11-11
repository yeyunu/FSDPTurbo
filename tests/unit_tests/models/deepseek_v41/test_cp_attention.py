# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU geometry of the packed-TND attention.

The real ``TNDRuntime`` builds the plan and buffers; only the collective is
replayed. Assertions pin the operator metadata, per-document compressed indices,
and equality with the same rows of a cp=1 run.
"""

import pytest
import torch

from fsdp_turbo.distributed.context_parallel.kv_allgather import tnd
from fsdp_turbo.distributed.context_parallel.kv_allgather.tnd import TNDRuntime
from fsdp_turbo.models.deepseek.deepseek_v41 import cp_attention
from fsdp_turbo.models.deepseek.deepseek_v41.adapter import initialize_deepseek_v41_model
from fsdp_turbo.models.deepseek.deepseek_v41.cp_attention import CPAttention
from fsdp_turbo.models.deepseek.deepseek_v41.model import ModelArgs, set_dtype

WINDOW = 4
RATIO = 2
CU_ENDS = [0, 8, 24]  # two documents: 0..8 and 8..24
TOTAL = CU_ENDS[-1]


@pytest.fixture(autouse=True)
def cpu_operator_dispatch(monkeypatch):
    """Keep these cases on the CPU operator path.

    On an NPU host ``dispatch_op`` would otherwise select the device kernel,
    which rejects CPU tensors (and vice versa). Upstream's sparse-attention
    tests pin the same switch.
    """
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cpu"))


class _Collective:
    """Stand-in for the CP all-gather on a single node.

    ``record`` keeps each rank's send block; ``replay`` hands every rank the
    concatenation of both ranks' blocks in rank order, which is exactly what
    ``all_gather_var_len`` produces over a two-rank group.
    """

    WORLD_SIZE = 2

    def __init__(self):
        self.blocks = {}
        self.counter = 0

    def patch(self, monkeypatch, rank, replay=False):
        def fake_all_gather(local, group=None, block_len=None):
            index = self.counter
            self.counter += 1
            self.blocks[(rank, index)] = local.detach().clone()
            if not replay:
                # The real all-gather returns one block per rank.
                return local.new_zeros(1, block_len * self.WORLD_SIZE, local.shape[-1]), block_len
            first, second = self.blocks[(0, index)], self.blocks[(1, index)]
            width = max(first.shape[1], second.shape[1])
            parts = [torch.nn.functional.pad(block, (0, 0, 0, width - block.shape[1])) for block in (first, second)]
            return torch.cat(parts, dim=1), width

        monkeypatch.setattr(tnd, "all_gather_var_len", fake_all_gather)

    def restart(self):
        self.counter = 0


def _runtime(monkeypatch, rank=1, world_size=2, style="full_prefix"):
    """A real runtime whose plan belongs to *rank* of a *world_size* group."""
    monkeypatch.setattr(tnd.dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(tnd.dist, "get_world_size", lambda group=None: world_size)
    runtime = TNDRuntime(None)
    runtime.exchange_style = style
    runtime.window = WINDOW
    runtime.lcm = RATIO
    runtime.start_batch(CU_ENDS)
    return runtime


def build_attention():
    args = ModelArgs(
        max_batch_size=1,
        max_seq_len=64,
        dim=32,
        n_layers=5,
        n_heads=4,
        head_dim=16,
        rope_head_dim=4,
        q_lora_rank=8,
        o_groups=2,
        o_lora_rank=8,
        window_size=WINDOW,
        compress_ratios=(0, RATIO, 1, 0, 0),
        kv_source_layers=(1,),
        index_source_layers=(1,),
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        use_sparse_flash_attn=True,
    )
    with set_dtype(torch.bfloat16):
        attention = CPAttention(1, args)
    initialize_deepseek_v41_model(attention)
    return attention


def capture_operator(monkeypatch, op_output):
    """Fixed-output operator: records the call and returns *op_output*."""
    captured = {}

    def fake_operator(query, key_value, attention_sink, topk_indices, softmax_scale, **kwargs):
        captured.update(kwargs)
        captured["query"] = query
        captured["key_value"] = key_value
        captured["topk_indices"] = topk_indices
        return op_output.to(query.dtype)

    monkeypatch.setattr(cp_attention, "indexed_sparse_attention", fake_operator)
    return captured


def _local_rows(runtime):
    return sum(segment.q_hi - segment.q_lo for segment in runtime.plan.docs)


@pytest.mark.parametrize("style", ["full_prefix", "halo_prefix"])
def test_operator_receives_the_plan_metadata(monkeypatch, style):
    attention = build_attention()
    collective = _Collective()
    collective.patch(monkeypatch, rank=1)
    runtime = _runtime(monkeypatch, rank=1, style=style)
    attention.cp_runtime = runtime

    rows = _local_rows(runtime)
    x = torch.randn(1, rows, attention.dim, dtype=torch.bfloat16)
    captured = capture_operator(monkeypatch, torch.randn(rows, attention.n_local_heads, attention.head_dim))

    output = attention.forward_tnd(x)

    cu_q, cu_kv, cu_cmp = runtime.plan.cu_seqlens(RATIO)
    assert output.shape == x.shape
    assert captured["cu_seqlens"].tolist() == cu_q
    assert captured["cu_seqlens_ori_kv"].tolist() == cu_kv
    assert captured["cu_seqlens_cmp_kv"].tolist() == cu_cmp
    assert captured["key_value"].shape[0] == cu_kv[-1]
    assert captured["compressed_key_value"].shape[0] == cu_cmp[-1]
    assert captured["topk_indices"].shape == (rows, WINDOW)
    assert captured["compressed_topk_indices"].shape == (rows, 2)
    assert captured["compress_ratio"] == RATIO
    assert captured["layout"] == "TND"

    # full_prefix starts at row 0; halo starts at the window and holds fewer rows.
    own_prefix = sum(segment.q_hi for segment in runtime.plan.docs)
    if style == "full_prefix":
        assert cu_kv[-1] == own_prefix
    else:
        assert cu_kv[-1] < own_prefix


def test_cp_output_matches_the_single_stream(monkeypatch):
    """Rows a rank owns must equal the same rows of a cp=1 run.

    The operator is replaced by a row-causal function of the query and the KV
    row it reads, so the comparison exercises the gathered prefix (and the
    inverse RoPE on the way out) rather than the kernel.
    """
    attention = build_attention()
    torch.manual_seed(5)
    x_full = torch.randn(1, TOTAL, attention.dim, dtype=torch.bfloat16)
    cuts = (0, 12, TOTAL)
    local_x = [x_full[:, cuts[rank] : cuts[rank + 1]] for rank in range(2)]

    def kv_operator(monkeypatch):
        def fake_operator(query, key_value, attention_sink, topk_indices, softmax_scale, **kwargs):
            # A query row reads its own KV row, so a short prefix matches the document.
            parts = []
            q_bounds = kwargs["cu_seqlens"].tolist()
            kv_bounds = kwargs["cu_seqlens_ori_kv"].tolist()
            for (q_lo, q_hi), (kv_lo, kv_hi) in zip(
                zip(q_bounds[:-1], q_bounds[1:]), zip(kv_bounds[:-1], kv_bounds[1:])
            ):
                rows = q_hi - q_lo
                # Query rows are the document tail; a CP rank agrees row-wise with the whole.
                assert kv_hi - kv_lo >= rows
                parts.append(query[q_lo:q_hi].float() + key_value[kv_hi - rows : kv_hi].float().unsqueeze(1))
            return torch.cat(parts).to(query.dtype)

        monkeypatch.setattr(cp_attention, "indexed_sparse_attention", fake_operator)

    # reference: one rank owns the whole stream
    kv_operator(monkeypatch)
    # world_size=1 skips the all-gather, so the reference sees the real KV.
    reference_runtime = _runtime(monkeypatch, rank=0, world_size=1)
    assert reference_runtime.plan.cuts == (0, TOTAL)
    attention.cp_runtime = reference_runtime
    full_output = attention.forward_tnd(x_full)

    # pass 1: record what each rank sends (the operator is not exercised here)
    collective = _Collective()
    for rank in range(2):
        capture_operator(
            monkeypatch,
            torch.zeros(local_x[rank].shape[1], attention.n_local_heads, attention.head_dim),
        )
        collective.patch(monkeypatch, rank=rank)
        attention.cp_runtime = _runtime(monkeypatch, rank=rank)
        attention.forward_tnd(local_x[rank])
        collective.restart()

    # pass 2: replay the two-rank gather and compare rows with the reference
    kv_operator(monkeypatch)
    for rank in range(2):
        collective.patch(monkeypatch, rank=rank, replay=True)
        attention.cp_runtime = _runtime(monkeypatch, rank=rank)
        output = attention.forward_tnd(local_x[rank])
        lo, hi = cuts[rank], cuts[rank + 1]
        assert torch.equal(output, full_output[:, lo:hi]), f"rank {rank} diverged from the single stream"
        collective.restart()


def test_halo_first_segment_keeps_the_gather_in_its_backward(monkeypatch):
    """The first segment of a document reads no gathered rows.

    Its output must still keep the all-gather in the autograd graph: a pruned
    node makes this rank skip the collective's reduce-scatter while the ranks
    that do read gathered rows run it, which deadlocks the backward.
    """
    runtime = _runtime(monkeypatch, rank=0, style="halo_prefix")
    backward_calls = []

    class _Gather(torch.autograd.Function):
        @staticmethod
        def forward(ctx, local, width):
            ctx.width = width
            return local.detach().repeat(1, 2, 1)

        @staticmethod
        def backward(ctx, grad_output):
            backward_calls.append(1)
            return grad_output[:, : ctx.width], None

    def fake_all_gather(local, group=None, block_len=None):
        return _Gather.apply(local, local.shape[1]), block_len

    monkeypatch.setattr(tnd, "all_gather_var_len", fake_all_gather)

    local_rows = {
        segment.doc_id: torch.randn(1, segment.q_hi - segment.q_lo, 4, requires_grad=True)
        for segment in runtime.plan.docs
    }
    outputs = runtime.gather_document_rows(local_rows, 1, "raw")
    sum(output.sum() for output in outputs.values()).backward()

    assert backward_calls

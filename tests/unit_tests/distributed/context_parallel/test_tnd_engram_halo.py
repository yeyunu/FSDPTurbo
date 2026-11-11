# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Engram n-gram halo: CP slicing must reproduce the whole-document hash ids."""

import torch

from fsdp_turbo.distributed.context_parallel.kv_allgather import tnd
from fsdp_turbo.distributed.context_parallel.kv_allgather.tnd import TNDRuntime
from fsdp_turbo.models.deepseek.deepseek_v41.engram import CPNgramHashState, EngramLayout

MAX_NGRAM = 3
LAYOUT = EngramLayout(
    max_ngram_size=MAX_NGRAM,
    layer_ids=(0,),
    num_embeddings=(64,),
    primes=(((3, 5), (7, 11)),),
    n_heads=2,
    head_dim=4,
)


def build_state():
    # Skip __init__: the vendored constructor needs the tokenizer and args.
    state = CPNgramHashState.__new__(CPNgramHashState)  # pylint: disable=no-value-for-parameter
    torch.nn.Module.__init__(state)  # pylint: disable=unnecessary-dunder-call
    state.layout = LAYOUT
    state.pad_id = 1
    flat = [[prime for per_ngram in layer for prime in per_ngram] for layer in LAYOUT.primes]
    offsets = [[0, *torch.tensor(sizes[:-1]).cumsum(0).tolist()] for sizes in flat]
    state.register_buffer("primes", torch.tensor(LAYOUT.primes), persistent=False)
    state.register_buffer("offsets", torch.tensor(offsets), persistent=False)
    state.register_buffer("multipliers", torch.tensor([[3, 5, 7]]), persistent=False)
    state.register_buffer("token_map", torch.arange(16) % 7, persistent=False)
    state.register_buffer("cache", torch.empty(1, 64, dtype=torch.int64), persistent=False)
    return state


def test_runtime_gathers_the_halo_across_ranks(monkeypatch):
    state = build_state()
    tokens = torch.arange(16).reshape(1, 16)
    ones = torch.ones(1, 8, dtype=torch.int64)
    padded = [
        torch.stack([tokens[:, :8], ones], dim=1),
        torch.stack([tokens[:, 8:], ones], dim=1),
    ]

    def fake_all_gather(output, tensor, group=None):
        for slot, block in zip(output, padded):
            slot.copy_(block)

    monkeypatch.setattr(tnd.dist, "all_gather", fake_all_gather)
    monkeypatch.setattr(tnd.dist, "get_world_size", lambda group=None: 2)

    reference = state(tokens, 0)
    for rank in (0, 1):
        monkeypatch.setattr(tnd.dist, "get_rank", lambda group=None, rank=rank: rank)
        runtime = TNDRuntime(group=None)
        runtime.window = 4
        runtime.start_batch([0, 16])
        local = tokens[:, rank * 8 : (rank + 1) * 8]
        hashes = runtime.engram_hashes(state, local, None)
        assert torch.equal(hashes, reference[:, rank * 8 : (rank + 1) * 8])

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MTP local-embedding equivalence under context parallelism.

``mtp_local_embedding`` (default) embeds only this rank's id shard instead of
the full sequence before the CP split.  This A/B test runs the same stubbed
MTP block twice on a two-rank gloo CP group -- once with the legacy
full-length embedding path, once with the local path -- and requires:

* bit-identical losses and downstream-parameter gradients per rank, and
* the legacy embedding-weight gradient (the CP-summed gradient every rank
  received through the all-gather split) to equal the SUM over ranks of the
  local path's partial gradients.  After FSDP's mean reduction the two
  coincide; the local path additionally matches the main-model path's
  partial-gradient semantics, which the CP loss scaling compensates for.

Three prediction steps also check that alignment padding does not leak between steps.
The sequence length (10) is deliberately unaligned (cp 2 x compress_ratio 4
pads to 16) so the alignment-tail zeroing is exercised.
"""

import os
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from fsdp_turbo.models.qwen import modeling_qwen4_exp as mtp_mod
from fsdp_turbo.models.qwen.modeling_qwen4_exp import (
    Qwen4ExpMultiTokenPredictionBlock as MultiTokenPredictionBlock,
)

HIDDEN = 8
VOCAB = 16
SEQ = 10
RATIO = 4
WORLD = 2


class _StubConfig:
    mtp_num_layers = 3
    hidden_size = HIDDEN
    rms_norm_eps = 1e-6
    hc_count = None
    indexer_compress_ratio = RATIO
    vocab_size = VOCAB
    enable_chunk_loss = False
    mtp_local_embedding = True


class _Norm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

    def forward(self, x):
        return x


class _Decoder(nn.Module):
    layer_type = "linear_attention"

    def __init__(self, config, layer_idx=0):
        super().__init__()

    def forward(self, hidden_states, **kwargs):
        return hidden_states * 1.5 + 0.25


class _Rotary(nn.Module):
    def forward(self, hidden_states, position_ids):
        table = torch.ones(1, 1, 1)
        return table, table


class _State:
    def __init__(self, group, size):
        self._group = group
        self._size = size

    def is_group_enable(self, name):
        return name == "cp" and self._size > 1

    def get_cp_group_size(self):
        return self._size

    def get_cp_group(self):
        return self._group


def _worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ["_MTP_EQUIV_PORT"]
    dist.init_process_group("gloo", rank=rank, world_size=world)

    torch.manual_seed(11)
    embed = nn.Embedding(VOCAB, HIDDEN)
    head = nn.Linear(HIDDEN, VOCAB, bias=False)
    torch.manual_seed(12)
    block = MultiTokenPredictionBlock(_StubConfig(), _Decoder, _Norm)

    torch.manual_seed(13)
    input_ids = torch.randint(0, VOCAB, (1, SEQ))
    labels = torch.randint(0, VOCAB, (1, SEQ))
    full_hidden = torch.randn(1, SEQ, HIDDEN)
    validity = torch.ones(1, SEQ, dtype=torch.bool)
    validity[0, SEQ - 3 :] = False

    pad_multiple = world * RATIO
    pad_len = (pad_multiple - SEQ % pad_multiple) % pad_multiple
    full_len = SEQ + pad_len
    shard = full_len // world
    hidden_local = F.pad(full_hidden, (0, 0, 0, pad_len))[:, rank * shard : (rank + 1) * shard].contiguous()

    def run(local_embedding):
        block.config.mtp_local_embedding = local_embedding
        for module in (embed, head, block):
            for param in module.parameters():
                param.grad = None
        state = _State(dist.group.WORLD, world)
        with mock.patch.object(mtp_mod, "get_parallel_state", return_value=state):
            losses = block.forward(
                hidden_local,
                input_ids=input_ids.clone(),
                labels=labels.clone(),
                attention_mask=validity.clone(),
                embed_tokens=embed,
                rotary_emb=_Rotary(),
                output_layer=head,
                loss_function=lambda logits, lab, vocab_size: logits.float().sum(),
            )
        sum(losses).backward()
        grads = {"embed": embed.weight.grad.clone(), "head": head.weight.grad.clone()}
        grads.update({f"block.{name}": p.grad.clone() for name, p in block.named_parameters()})
        return sum(losses).detach(), grads

    loss_old, grads_old = run(False)
    loss_new, grads_new = run(True)

    assert torch.equal(loss_old, loss_new), f"loss mismatch: {loss_old} vs {loss_new}"

    # Downstream parameters sit on the local graph in both paths: bitwise.
    for name, grad_old in grads_old.items():
        if name == "embed":
            continue
        assert torch.equal(grad_old, grads_new[name]), f"{name} gradient mismatch"

    # The legacy path delivered the CP-summed embedding gradient on every
    # rank; the local path delivers the rank partial.  Their relationship
    # (sum of partials == legacy) is what makes the two coincide after FSDP's
    # mean reduction.  Tolerance: the two sum the same contributions in a
    # different accumulation order.
    partials = [torch.empty_like(grads_new["embed"]) for _ in range(world)]
    dist.all_gather(partials, grads_new["embed"].contiguous())
    partial_sum = torch.stack(partials).sum(0)
    assert torch.allclose(grads_old["embed"], partial_sum, rtol=1e-5, atol=1e-6), (
        "embedding gradient mismatch: legacy (CP-summed) vs sum of local partials"
    )
    # The legacy gradient is identical on every rank by construction.
    legacy = [torch.empty_like(grads_old["embed"]) for _ in range(world)]
    dist.all_gather(legacy, grads_old["embed"].contiguous())
    for other in legacy:
        assert torch.equal(grads_old["embed"], other)

    dist.barrier()
    if rank == 0:
        print("MTP local-embedding equivalence PASSED", flush=True)
    dist.destroy_process_group()


class MTPLocalEmbeddingEquivalenceTest(unittest.TestCase):
    def test_two_rank_equivalence(self):
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ["_MTP_EQUIV_PORT"] = str(port)
        mp.spawn(_worker, args=(WORLD,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    unittest.main()

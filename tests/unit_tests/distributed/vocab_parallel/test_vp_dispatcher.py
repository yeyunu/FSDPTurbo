# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Numerical correctness test for the vocabulary-parallel (VP) embedding path.

Verifies, against a plain ``nn.Embedding`` reference, that
``vocab_parallelize_modules`` + ``vocab_parallel_embedding_forward``:

  1. forward: dispatcher output is bitwise identical to a full-table lookup,
     including ids at exact shard boundaries;
  2. backward: with per-rank identical inputs and
     ``gradient_divide_factor = world_size`` the local weight gradient equals
     the reference gradient slice of that rank (the same contract FSDP
     gradient averaging gives dense parameters);
  3. backward: with per-rank distinct inputs and ``gradient_divide_factor
     = 1.0`` the local gradient equals the sum of all ranks' reference
     gradients restricted to that rank's vocab range;
  4. the host-resident (CPU) weight path used by the PLE n-gram table: only
      this rank's shard is moved to the accelerator (``DTensor.from_local``),
      never the full table;
  5. the Qwen3.8-Flash-Next (qwen4_exp) PLE layout of examples/qwen3_8: the
      production wildcard pattern shards exactly the per-layer PLE tables
      (independent, CPU-materialized, trainable), each numerically correct
      against its own full-table reference.

Launch (requires a real collective backend; gloo has no all-to-all):

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    torchrun --nproc_per_node=8 \
        tests/unit_tests/distributed/vocab_parallel/test_vp_dispatcher.py

Any failing rank propagates the failure to all ranks, so the launcher fails
loudly instead of hanging.
"""

import os
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from fsdp_turbo.distributed.vocab_parallel import vocab_parallelize_modules
from fsdp_turbo.fsdp_turbo_config import VPPlanConfig

VOCAB = 4096
DIM = 32
BATCH = 2
SEQ = 257  # odd on purpose: exercises the uneven all-to-all tail
NUM_LAYERS = 5
PLE_LAYER_IDS = (1, 3)  # only some layers carry a PLE block, like qwen4_exp ple_layer_ids
QWEN4_EXP_VP_PATTERN = "model.language_model.layers.{*}.ple.ple_embedding.ngram_embedding"

_FAILURES = []


def _check(cond, msg, rank):
    if not cond:
        print(f"[rank {rank}] FAIL: {msg}", flush=True)
        _FAILURES.append(msg)


class ToyPLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ngram_embedding = nn.Embedding(VOCAB, DIM)

    def forward(self, ids):
        return self.ngram_embedding(ids)


class Qwen4ExpLikeNgramEmbedding(nn.Module):
    """Holder mirroring the VP-mode state of ``Qwen4ExpTextNGramEmbeddingCpu``.

    In the examples/qwen3_8 flow ``materialize_ngram_embedding_on_cpu`` runs
    with ``freeze=False`` before ``FSDPTurbo``, leaving a full trainable
    CPU-resident table in ``ngram_embedding`` for VP to shard.  Structure-only
    stubs keep this test free of the transformers qwen4_exp dependency.
    """

    def __init__(self):
        super().__init__()
        self.ngram_embedding = nn.Embedding(VOCAB, DIM)


class Qwen4ExpLikePleLayer(nn.Module):
    """The ``ple.ple_embedding.ngram_embedding`` chain of a qwen4_exp PLE layer."""

    def __init__(self):
        super().__init__()
        self.ple_embedding = Qwen4ExpLikeNgramEmbedding()


class Qwen4ExpLikeModel(nn.Module):
    """Structure-only stub of Qwen3.8-Flash-Next down to the PLE ngram tables.

    ``named_modules()`` yields exactly the names the production vp_plan
    pattern (examples/qwen3_8/config.yaml) matches; only ``PLE_LAYER_IDS``
    layers carry a PLE block, each with an INDEPENDENT table like the real
    per-layer PLE tables.
    """

    def __init__(self):
        super().__init__()
        language_model = nn.Module()
        language_model.layers = nn.ModuleList(
            Qwen4ExpLikePleLayer() if layer_idx in PLE_LAYER_IDS else nn.Module() for layer_idx in range(NUM_LAYERS)
        )
        model = nn.Module()
        model.language_model = language_model
        self.model = model

    def ngram_embedding(self, layer_idx):
        return self.model.language_model.layers[layer_idx].ple.ple_embedding.ngram_embedding


def check_qwen4_exp_ple_hierarchy(rank, world, vp_mesh, vp_rank, device, local_vocab):
    """VP over the Qwen3.8-Flash-Next PLE layout, as in examples/qwen3_8.

    Per-layer tables are materialized identically on every rank (seeded init
    plays the role of the /dev/shm sharing) and stay CPU-resident until VP
    moves only the local shard to the accelerator; the production wildcard
    pattern must shard exactly the PLE tables, each numerically correct
    against its own full-table reference.
    """
    model = Qwen4ExpLikeModel()
    tables = {}
    for layer_idx in PLE_LAYER_IDS:
        table = torch.randn(VOCAB, DIM, generator=torch.Generator().manual_seed(900 + layer_idx))
        tables[layer_idx] = table
        with torch.no_grad():
            model.ngram_embedding(layer_idx).weight.copy_(table)

    vocab_parallelize_modules(model, vp_mesh, VPPlanConfig(apply_modules=[QWEN4_EXP_VP_PATTERN]))

    sharded_count = sum(
        isinstance(module, nn.Embedding) and isinstance(module.weight, DTensor) for module in model.modules()
    )
    _check(
        sharded_count == len(PLE_LAYER_IDS),
        f"wildcard pattern sharded {sharded_count} tables, expected {len(PLE_LAYER_IDS)}",
        rank,
    )

    for layer_idx in PLE_LAYER_IDS:
        emb = model.ngram_embedding(layer_idx)
        _check(isinstance(emb.weight, DTensor), f"layer {layer_idx}: weight is not a DTensor", rank)
        _check(
            tuple(emb.weight.to_local().shape) == (local_vocab, DIM),
            f"layer {layer_idx}: unexpected local shard shape",
            rank,
        )
        _check(emb.weight.to_local().device.type == device, f"layer {layer_idx}: local shard not on accelerator", rank)
        _check(emb.weight.requires_grad, f"layer {layer_idx}: trainable table became frozen", rank)
        _check(
            emb._vp_vocab_start_index == vp_rank * local_vocab
            and emb._vp_vocab_end_index == (vp_rank + 1) * local_vocab,
            f"layer {layer_idx}: bad vocab range [{emb._vp_vocab_start_index}, {emb._vp_vocab_end_index})",
            rank,
        )

        ref = nn.Embedding(VOCAB, DIM).to(device)
        with torch.no_grad():
            ref.weight.copy_(tables[layer_idx])
        ref.weight.requires_grad_(True)

        # Identical ids on all ranks: random plus the global shard-boundary
        # rows, so owner routing hits every layer's table edges.
        generator = torch.Generator().manual_seed(1000 + layer_idx)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator)
        edges = torch.arange(world + 1) * local_vocab
        ids[0, 0] = 0
        ids[0, 1 : edges.numel()] = edges[1:] - 1
        ids[0, edges.numel()] = local_vocab
        ids = ids.to(device)
        grad_out = torch.randn((BATCH, SEQ, DIM), generator=generator).to(device)

        _check(torch.equal(emb(ids), ref(ids)), f"layer {layer_idx}: forward output mismatch", rank)

        # Same contract as backward A above: identical inputs and divide
        # factor = world make the local grad equal the reference slice.
        emb._vp_gradient_divide_factor = float(world)
        (emb(ids) * grad_out).sum().backward()
        (ref(ids) * grad_out).sum().backward()
        local_grad = emb.weight.grad.to_local()
        ref_slice = ref.weight.grad[vp_rank * local_vocab : (vp_rank + 1) * local_vocab]
        _check(
            torch.allclose(local_grad, ref_slice, atol=1e-5, rtol=1e-5),
            f"layer {layer_idx}: local grad != reference slice",
            rank,
        )


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    from fsdp_turbo.utils.device import get_accelerator_name, set_accelerator_compatible

    backend = set_accelerator_compatible()
    device = get_accelerator_name()
    dist.init_process_group(backend=f"cpu:gloo,{device}:{backend}")
    if device == "npu":
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    vp_mesh = init_device_mesh(device, (world,), mesh_dim_names=("vp",))
    vp_rank = dist.get_rank(vp_mesh.get_group())
    local_vocab = VOCAB // world

    # Same full table on every rank (the PLE flow materializes it identically
    # via /dev/shm sharing; broadcast plays that role here).
    full_weight = torch.randn(VOCAB, DIM, dtype=torch.float32)
    dist.broadcast(full_weight, src=0)
    full_weight = full_weight.to(device)

    # ---- VP path: host-resident weight, exactly like the PLE table -------
    model = ToyPLE()
    with torch.no_grad():
        model.ngram_embedding.weight.copy_(full_weight.cpu())
    model.ngram_embedding.weight.requires_grad_(True)

    plan = VPPlanConfig(apply_modules=["ngram_embedding"])
    vocab_parallelize_modules(model, vp_mesh, plan)

    emb = model.ngram_embedding
    _check(isinstance(emb.weight, DTensor), "weight is not a DTensor", rank)
    _check(tuple(emb.weight.to_local().shape) == (local_vocab, DIM), "unexpected local shard shape", rank)
    _check(emb._vp_vocab_start_index == vp_rank * local_vocab, "bad _vp_vocab_start_index", rank)
    _check(emb._vp_vocab_end_index == (vp_rank + 1) * local_vocab, "bad _vp_vocab_end_index", rank)
    _check(emb.weight.to_local().device.type == device, "local shard not on accelerator", rank)
    _check(emb.weight.requires_grad, "trainable table became frozen", rank)

    # ---- reference path --------------------------------------------------
    ref = nn.Embedding(VOCAB, DIM).to(device)
    with torch.no_grad():
        ref.weight.copy_(full_weight)
    ref.weight.requires_grad_(True)

    # Ids: random plus shard boundaries (local_vocab-1, local_vocab, ...),
    # the global first/last rows — so owner routing is exercised on edges.
    generator = torch.Generator().manual_seed(1234)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator)
    edges = torch.arange(world + 1) * local_vocab
    ids[0, 0] = 0
    ids[0, 1 : edges.numel()] = edges[1:] - 1
    ids[0, edges.numel()] = local_vocab  # first row of rank 1's shard
    dist.broadcast(ids, src=0)
    ids = ids.to(device)

    # ---- forward ---------------------------------------------------------
    out_vp = emb(ids)
    out_ref = ref(ids)
    _check(out_vp.shape == out_ref.shape, f"shape {tuple(out_vp.shape)} != {tuple(out_ref.shape)}", rank)
    _check(torch.equal(out_vp, out_ref), "forward output mismatch vs full-table lookup", rank)

    # ---- backward A: identical inputs, divide factor = world -------------
    # The factor is read at FORWARD time and stashed in the autograd context,
    # so it must be set before the forward call.
    grad_out = torch.randn(out_ref.shape, dtype=torch.float32)
    dist.broadcast(grad_out, src=0)
    grad_out = grad_out.to(device)

    emb._vp_gradient_divide_factor = float(world)
    out_vp = emb(ids)
    (out_vp * grad_out).sum().backward()
    (out_ref * grad_out).sum().backward()

    local_grad = emb.weight.grad.to_local()
    ref_slice = ref.weight.grad[vp_rank * local_vocab : (vp_rank + 1) * local_vocab]
    # Bitwise equality is impossible: the dispatcher accumulates the world_size
    # per-rank contributions before dividing, so summation order differs from
    # the single-rank reference. Tight allclose is the correct contract.
    _check(
        torch.allclose(local_grad, ref_slice, atol=1e-5, rtol=1e-5),
        "backward A: local grad != reference slice",
        rank,
    )

    # ---- backward B: distinct inputs per rank, divide factor = 1.0 -------
    emb.weight.grad = None
    ref.weight.grad = None
    ids_rank = (ids + rank * 37) % VOCAB
    emb._vp_gradient_divide_factor = 1.0
    out_vp = emb(ids_rank)
    out_ref = ref(ids_rank)
    grad_out = torch.randn(out_ref.shape, dtype=torch.float32, device=device)

    (out_vp * grad_out).sum().backward()
    (out_ref * grad_out).sum().backward()

    # Expected local grad = sum over ranks of each rank's reference grad,
    # restricted to this rank's vocab range.
    summed_ref = ref.weight.grad.clone()
    dist.all_reduce(summed_ref, op=dist.ReduceOp.SUM)
    local_grad = emb.weight.grad.to_local()
    ref_slice = summed_ref[vp_rank * local_vocab : (vp_rank + 1) * local_vocab]
    _check(
        torch.allclose(local_grad, ref_slice, atol=1e-4, rtol=1e-5),
        "backward B: local grad != summed reference slice",
        rank,
    )

    # ---- Qwen3.8-Flash-Next (qwen4_exp) PLE hierarchy ---------------------
    check_qwen4_exp_ple_hierarchy(rank, world, vp_mesh, vp_rank, device, local_vocab)

    # ---- aggregate verdict so no rank hangs on a peer's failure ----------
    failure_count = torch.tensor([len(_FAILURES)], dtype=torch.int64)
    dist.all_reduce(failure_count, op=dist.ReduceOp.SUM)
    if failure_count.item() > 0:
        if rank == 0:
            print(f"FAILED: {failure_count.item()} check failure(s) across ranks", flush=True)
        dist.destroy_process_group()
        sys.exit(1)

    if rank == 0:
        print(
            f"PASS: VP dispatcher forward+backward correct (toy + qwen4_exp PLE, world={world}, vocab={VOCAB}, dim={DIM}, device={device})",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

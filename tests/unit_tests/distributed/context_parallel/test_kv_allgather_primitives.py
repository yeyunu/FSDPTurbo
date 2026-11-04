# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Multi-process (gloo) tests for the KV-AllGather CP primitives.

``sparse_gather_rows`` exchanges data-dependent amounts between every pair of
ranks; ``conv_tail_exchange`` moves convolution context between neighbors.
Both run here over gloo, which supports pairwise-consistent unequal all-to-all
splits and point-to-point send/recv.
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fsdp_turbo.distributed.context_parallel.kv_allgather.conv_tail import conv_tail_exchange
from fsdp_turbo.distributed.context_parallel.kv_allgather.sparse_gather import sparse_gather_rows


def _init_gloo(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)


def _sparse_gather_worker(rank, world, port, shard_len, head_count, head_dim, seed):
    _init_gloo(rank, world, port)
    try:
        torch.manual_seed(seed)
        full_k = torch.randn(1, head_count, world * shard_len, head_dim)
        full_v = torch.randn(1, head_count, world * shard_len, head_dim)

        # Deterministic per-rank request sets: unbalanced sizes, overlapping
        # tokens, and every rank's first token requested by everyone so no
        # rank ends up serving nothing.
        request_sets = []
        for r in range(world):
            generator = torch.Generator().manual_seed(seed + 100 + r)
            count = int(torch.randint(1, 12, (1,), generator=generator))
            random_ids = torch.randint(0, world * shard_len, (count,), generator=generator)
            coverage = torch.arange(0, world * shard_len, shard_len)
            request_sets.append(torch.sort(torch.unique(torch.cat([random_ids, coverage])))[0])
        indices = request_sets[rank].to(torch.int64)

        local_k = full_k[:, :, rank * shard_len : (rank + 1) * shard_len].clone().requires_grad_(True)
        local_v = full_v[:, :, rank * shard_len : (rank + 1) * shard_len].clone().requires_grad_(True)
        gathered_k, gathered_v = sparse_gather_rows((local_k, local_v), indices, shard_len, dist.group.WORLD)

        assert torch.equal(gathered_k, full_k.index_select(2, indices)), "forward mismatch (k)"
        assert torch.equal(gathered_v, full_v.index_select(2, indices)), "forward mismatch (v)"

        # Per-rank loss on the gathered rows; the reference gradient of every
        # shard is the sum of all ranks' contributions through the full tensor.
        weights = []
        for r in range(world):
            generator = torch.Generator().manual_seed(seed + 500 + r)
            n = request_sets[r].numel()
            weights.append(
                (
                    torch.randn(1, head_count, n, head_dim, generator=generator),
                    torch.randn(1, head_count, n, head_dim, generator=generator),
                )
            )
        loss = (gathered_k * weights[rank][0]).sum() + (gathered_v * weights[rank][1]).sum()
        loss.backward()

        ref_k = full_k.clone().requires_grad_(True)
        ref_v = full_v.clone().requires_grad_(True)
        total = 0
        for r in range(world):
            ids = request_sets[r].to(torch.int64)
            total = total + (ref_k.index_select(2, ids) * weights[r][0]).sum()
            total = total + (ref_v.index_select(2, ids) * weights[r][1]).sum()
        total.backward()
        assert torch.allclose(local_k.grad, ref_k.grad[:, :, rank * shard_len : (rank + 1) * shard_len], atol=1e-5), (
            "backward mismatch (k)"
        )
        assert torch.allclose(local_v.grad, ref_v.grad[:, :, rank * shard_len : (rank + 1) * shard_len], atol=1e-5), (
            "backward mismatch (v)"
        )
    finally:
        dist.destroy_process_group()


def _conv_tail_worker(rank, world, port, seg_len, channels, width, seed):
    _init_gloo(rank, world, port)
    try:
        torch.manual_seed(seed)
        full = torch.randn(1, world * seg_len, channels)
        grads = []
        for r in range(world):
            generator = torch.Generator().manual_seed(seed + 100 + r)
            length = seg_len + (width if r > 0 else 0)
            grads.append(torch.randn(1, length, channels, generator=generator))

        local = full[:, rank * seg_len : (rank + 1) * seg_len].clone().requires_grad_(True)
        extended, borrowed = conv_tail_exchange(local, dist.group.WORLD, width)

        if rank == 0:
            assert not borrowed
            assert torch.equal(extended, local.detach()), "rank 0 output changed"
        else:
            assert borrowed
            expected_prefix = full[:, rank * seg_len - width : rank * seg_len]
            assert torch.equal(extended[:, :width], expected_prefix), "borrowed prefix mismatch"
            assert torch.equal(extended[:, width:], local.detach()), "local part mismatch"

        (extended * grads[rank]).sum().backward()

        # Expected local gradient: the local part of this rank's grad plus the
        # borrowed-prefix grad arriving from the next rank.
        expected = grads[rank][:, width:] if rank > 0 else grads[rank].clone()
        if rank < world - 1:
            incoming = grads[rank + 1][:, :width]
            expected = expected.clone()
            expected[:, -width:] += incoming
        assert torch.allclose(local.grad, expected, atol=1e-6), "gradient routing mismatch"
    finally:
        dist.destroy_process_group()


class SparseGatherTests(unittest.TestCase):
    def test_forward_and_backward(self):
        for world in (2, 4):
            with self.subTest(world=world):
                mp.spawn(
                    _sparse_gather_worker,
                    args=(world, 29611, 8, 3, 5, 1234 + world),
                    nprocs=world,
                    join=True,
                )


class ConvTailExchangeTests(unittest.TestCase):
    def test_forward_and_backward(self):
        mp.spawn(
            _conv_tail_worker,
            args=(4, 29612, 16, 6, 3, 4321),
            nprocs=4,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()

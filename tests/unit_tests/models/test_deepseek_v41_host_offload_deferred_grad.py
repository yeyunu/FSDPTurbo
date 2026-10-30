# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import unittest

import torch

from fsdp_turbo.models.deepseek.deepseek_v41.engram.host_offload_engram import (
    HostOffloadEngram,
)


def _build_table() -> HostOffloadEngram:
    # No shard_(): the buffer is created only when an EP mesh is attached, so
    # gradient bookkeeping can be exercised on CPU without CANN operators.
    return HostOffloadEngram(2048, 128, layer_id=1, num_max_tokens_per_rank=16)


def _eager_reduce(parts, weight_shape):
    """The pre-deferral accumulation: coalesce each part as soon as it arrives."""
    pending = None
    for ids, values in parts:
        gradient = torch.sparse_coo_tensor(
            ids.unsqueeze(0),
            values,
            weight_shape,
            dtype=torch.float32,
            check_invariants=False,
        ).coalesce()
        pending = gradient if pending is None else (pending + gradient).coalesce()
    return pending


class DeferredSparseGradientTests(unittest.TestCase):
    def test_accumulate_defers_d2h_and_coalescing_to_consumer(self):
        table = _build_table()
        table.accumulate_sparse_gradient(
            torch.tensor([1, 3], dtype=torch.int64),
            torch.ones(2, 128, dtype=torch.float32),
        )
        self.assertIsNone(table._pending_sparse_grad)
        self.assertEqual(len(table._deferred_sparse_parts), 1)

    def test_materialization_replays_parts_in_arrival_order_bit_for_bit(self):
        table = _build_table()
        parts = [
            (
                torch.tensor([3, 1], dtype=torch.int64),
                torch.tensor([[1.0] * 128, [2.0] * 128], dtype=torch.float32),
            ),
            (
                torch.tensor([1, 4], dtype=torch.int64),
                torch.tensor([[0.5] * 128, [4.0] * 128], dtype=torch.float32),
            ),
        ]
        for ids, values in parts:
            table.accumulate_sparse_gradient(ids, values)

        pending = table.pending_sparse_grad().coalesce()
        expected = _eager_reduce(parts, table.weight.shape).coalesce()
        self.assertTrue(torch.equal(pending.to_dense(), expected.to_dense()))
        # A second consumer call must not replay (and double-count) the parts.
        self.assertEqual(len(table._deferred_sparse_parts), 0)
        self.assertTrue(torch.equal(table.pending_sparse_grad().to_dense(), expected.to_dense()))

    def test_bounds_check_is_deferred_until_materialization(self):
        table = _build_table()
        table.accumulate_sparse_gradient(
            torch.tensor([2048], dtype=torch.int64),
            torch.ones(1, 128, dtype=torch.float32),
        )
        self.assertEqual(len(table._deferred_sparse_parts), 1)
        with self.assertRaises(IndexError):
            table.pending_sparse_grad()

    def test_scale_consumes_deferred_parts(self):
        table = _build_table()
        table.accumulate_sparse_gradient(
            torch.tensor([2], dtype=torch.int64),
            torch.ones(1, 128, dtype=torch.float32),
        )
        table.scale_pending_sparse_grad(0.25)
        dense = table.pending_sparse_grad().to_dense()
        self.assertTrue(torch.equal(dense[2], torch.full((128,), 0.25)))
        self.assertTrue(torch.equal(dense[0], torch.zeros(128)))

    def test_clear_and_step_complete_drop_unmaterialized_parts(self):
        table = _build_table()
        table.accumulate_sparse_gradient(
            torch.tensor([1], dtype=torch.int64),
            torch.ones(1, 128, dtype=torch.float32),
        )
        table.clear_sparse_gradient()
        self.assertEqual(len(table._deferred_sparse_parts), 0)
        self.assertIsNone(table.pending_sparse_grad())

        table.accumulate_sparse_gradient(
            torch.tensor([1], dtype=torch.int64),
            torch.ones(1, 128, dtype=torch.float32),
        )
        table.mark_sparse_step_complete()
        self.assertEqual(len(table._deferred_sparse_parts), 0)
        self.assertIsNone(table.pending_sparse_grad())


if __name__ == "__main__":
    unittest.main()

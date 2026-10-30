# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import unittest
from unittest import mock

import torch

from fsdp_turbo.models.deepseek.deepseek_v41.context_parallel import (
    exchange_cp_hash_inputs_halo,
)


class CpHaloExchangeTests(unittest.TestCase):
    """The halo exchange mirrors a contiguous ring: rank r sends its tail to
    rank r+1; torch.distributed is emulated because only split bookkeeping and
    tensor assembly are under test.
    """

    def _run_rank(self, rank: int, world_size: int, shards: list[list[int]], pad_id: int = 99):
        halo_size = 3
        local = torch.tensor([shards[rank]], dtype=torch.int64)

        def fake_all_to_all_single(output, boundary, output_split_sizes, input_split_sizes, group):
            expected_send = [0] * world_size
            expected_recv = [0] * world_size
            if rank + 1 < world_size:
                expected_send[rank + 1] = halo_size
                self.assertEqual(boundary.tolist(), shards[rank][-halo_size:])
            if rank > 0:
                expected_recv[rank - 1] = halo_size
                output.copy_(torch.tensor(shards[rank - 1][-halo_size:], dtype=output.dtype))
            self.assertEqual(input_split_sizes, expected_send)
            self.assertEqual(output_split_sizes, expected_recv)

        with (
            mock.patch("torch.distributed.get_rank", return_value=rank),
            mock.patch("torch.distributed.get_world_size", return_value=world_size),
            mock.patch("torch.distributed.all_to_all_single", side_effect=fake_all_to_all_single),
        ):
            hash_input_ids, local_slice = exchange_cp_hash_inputs_halo(local, halo_size, pad_id, group=object())
        return local, hash_input_ids, local_slice

    def test_rank0_fills_halo_with_pad_and_keeps_full_sequence_prefix(self):
        shards = [list(range(6)), list(range(6, 12)), list(range(12, 18))]
        local, hash_input_ids, local_slice = self._run_rank(0, 3, shards)
        self.assertEqual(hash_input_ids.shape, (1, 9))
        self.assertEqual(hash_input_ids[0, :3].tolist(), [99, 99, 99])
        torch.testing.assert_close(hash_input_ids[0, local_slice], local[0])
        self.assertEqual(hash_input_ids[0].tolist(), [99, 99, 99, *range(6)])

    def test_last_rank_prepends_left_neighbour_tail(self):
        shards = [list(range(6)), list(range(6, 12)), list(range(12, 18))]
        local, hash_input_ids, local_slice = self._run_rank(2, 3, shards)
        self.assertEqual(hash_input_ids[0, :3].tolist(), [9, 10, 11])
        torch.testing.assert_close(hash_input_ids[0, local_slice], local[0])
        self.assertEqual(hash_input_ids[0].tolist(), [9, 10, 11, *range(12, 18)])

    def test_middle_rank_sends_and_receives_one_halo(self):
        shards = [list(range(6)), list(range(6, 12)), list(range(12, 18))]
        local, hash_input_ids, local_slice = self._run_rank(1, 3, shards)
        self.assertEqual(hash_input_ids[0, :3].tolist(), [3, 4, 5])
        torch.testing.assert_close(hash_input_ids[0, local_slice], local[0])
        self.assertEqual(hash_input_ids[0].tolist(), [3, 4, 5, *range(6, 12)])


if __name__ == "__main__":
    unittest.main()

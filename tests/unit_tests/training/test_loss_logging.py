# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import unittest
from unittest import mock

import torch

from fsdp_turbo.training.trainer import BaseTrainer


class TrainerLossLoggingTests(unittest.TestCase):
    @mock.patch("torch.distributed.is_initialized", return_value=False)
    def test_accumulation_uses_fp32_without_dividing_twice(self, _initialized):
        losses = [
            torch.tensor(12.1875 / 2, dtype=torch.bfloat16, requires_grad=True),
            torch.tensor(12.375 / 2, dtype=torch.bfloat16, requires_grad=True),
        ]
        result = BaseTrainer.sync_loss({"loss": losses, "aux_loss": []})
        self.assertEqual(result["loss"].item(), 12.28125)
        self.assertEqual(result["loss"].dtype, torch.float32)
        self.assertFalse(result["loss"].requires_grad)
        self.assertIsNone(result["aux_loss"])
        self.assertEqual([x.item() for x in losses], [6.09375, 6.1875])
        sum(losses).backward()
        for loss in losses:
            self.assertEqual(loss.grad.item(), 1.0)

    @mock.patch("torch.distributed.is_initialized", return_value=False)
    def test_single_micro_step(self, _initialized):
        result = BaseTrainer.sync_loss({"loss": [torch.tensor(3.25)]})
        self.assertEqual(result["loss"].item(), 3.25)

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig
from fsdp_turbo.training.trainer import BaseTrainer


class _ParallelState:
    def __init__(self, cp_size):
        self.cp_size = cp_size

    def get_cp_group_size(self):
        return self.cp_size

    def get_cp_group(self):
        return object()


class TrainerContextParallelTests(unittest.TestCase):
    def _trainer(self, cp_size=2, vision_config=None, wrap_model=False):
        trainer = BaseTrainer()
        model = SimpleNamespace(config=SimpleNamespace(vision_config=vision_config))
        if wrap_model:
            model = SimpleNamespace(config=SimpleNamespace(), model=model)
        trainer.model = SimpleNamespace(
            **vars(model),
            parallel_state=_ParallelState(cp_size),
        )
        return trainer

    @mock.patch("torch.distributed.get_rank", return_value=1)
    @mock.patch("torch.distributed.get_world_size", return_value=2)
    def test_text_batch_is_split_once_and_labels_remain_global(self, _world_size, _rank):
        trainer = self._trainer()
        batch = {
            "input_ids": torch.arange(5).view(1, 5),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "labels": torch.arange(5).view(1, 5),
        }

        output = trainer._split_batch_for_context_parallel(batch)

        torch.testing.assert_close(output["input_ids"], torch.tensor([[3, 4, 0]]))
        torch.testing.assert_close(output["position_ids"], torch.tensor([[3, 4, 0]]))
        torch.testing.assert_close(output["labels"], batch["labels"])

    def test_vlm_batch_is_deferred_to_post_fusion(self):
        trainer = self._trainer(vision_config=SimpleNamespace())
        batch = {
            "input_ids": torch.arange(4).view(1, 4),
            "pixel_values": torch.ones(1, 3, 2, 2),
        }

        output = trainer._split_batch_for_context_parallel(batch)

        self.assertIs(output, batch)

    def test_pure_text_vlm_is_deferred_without_visual_fields(self):
        trainer = self._trainer(vision_config=SimpleNamespace())
        batch = {"input_ids": torch.arange(4).view(1, 4)}

        output = trainer._split_batch_for_context_parallel(batch)

        self.assertIs(output, batch)

    def test_wrapped_vlm_is_deferred_without_pre_slicing(self):
        trainer = self._trainer(vision_config=SimpleNamespace(), wrap_model=True)
        batch = {
            "input_ids": torch.arange(4).view(1, 4),
            "pixel_values": torch.ones(1, 3, 2, 2),
        }

        output = trainer._split_batch_for_context_parallel(batch)

        self.assertIs(output, batch)

    def test_disabled_cp_keeps_batch_unchanged(self):
        trainer = self._trainer(cp_size=1)
        batch = {"input_ids": torch.arange(4).view(1, 4)}

        self.assertIs(trainer._split_batch_for_context_parallel(batch), batch)

    def test_packed_tnd_runtime_is_found_below_the_parallelism_wrapper(self):
        trainer = self._trainer()
        runtime = SimpleNamespace(split_batch=mock.Mock(return_value={"split": True}))
        trainer.model.model = SimpleNamespace(tnd_context_parallel=runtime)
        batch = {"input_ids": torch.arange(4).view(1, 4), "sfa_layout": "TND"}

        output = trainer._split_batch_for_context_parallel(batch)

        self.assertEqual(output, {"split": True})
        runtime.split_batch.assert_called_once_with(batch)

    @mock.patch("transformers.AutoProcessor.from_pretrained")
    def test_multimodal_trainer_builds_processor_and_passes_tokenizer(self, from_pretrained):
        processor = SimpleNamespace(tokenizer=object())
        from_pretrained.return_value = processor
        config = FSDPTurboConfig()
        config.data.is_multimodal = True
        config.model.tokenizer_name_or_path = "processor-source"

        trainer = BaseTrainer(config)
        tokenizer = trainer.build_tokenizer()

        self.assertIs(trainer.processor, processor)
        self.assertIs(tokenizer, processor.tokenizer)
        from_pretrained.assert_called_once_with("processor-source")


class TrainerResourceReleaseTests(unittest.TestCase):
    """The training loop has to give the model a chance to free device memory.

    A backend holding memory outside the caching allocator must release it while
    the process group is still up. MoonEP is the case that exists today: without
    this call ``npu_patch`` reports its symmetric buffer as leaked.
    """

    def test_a_model_that_owns_resources_is_released(self):
        trainer = BaseTrainer()
        released = []
        trainer.model = SimpleNamespace(close=lambda: released.append("close"))

        trainer._release_model_resources()

        self.assertEqual(released, ["close"])

    def test_a_model_without_resources_is_left_alone(self):
        trainer = BaseTrainer()
        trainer.model = SimpleNamespace()

        trainer._release_model_resources()

    def test_a_close_attribute_that_is_not_callable_is_ignored(self):
        """``close`` could be a submodule or a buffer on somebody's model."""
        trainer = BaseTrainer()
        trainer.model = SimpleNamespace(close=torch.zeros(1))

        trainer._release_model_resources()


if __name__ == "__main__":
    unittest.main()

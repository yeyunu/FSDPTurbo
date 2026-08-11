# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import sys
import types
import unittest
from unittest import mock

import torch
from torch.utils.data import Dataset

from fsdp_turbo.fsdp_turbo_config import DataConfig
from fsdp_turbo.training.dataloader import _MultimodalCollator, _process_multimodal_dataset, build_dataloader


class _Dataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index


class _ParallelState:
    @staticmethod
    def get_data_group_size():
        return 4

    @staticmethod
    def get_data_rank():
        return 2


class _MappedDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    def map(self, function, remove_columns=None, **_kwargs):
        return _MappedDataset([function(row) for row in self.rows])


class _Processor:
    image_token_id = 99

    def __init__(self):
        self.tokenizer = types.SimpleNamespace(pad_token_id=0, image_token_id=99)
        self.calls = []
        self.messages = []

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "<image>answer"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": torch.tensor([[99, 4, 0]])}


class DataloaderTests(unittest.TestCase):
    def test_multimodal_config_has_public_schema(self):
        config = DataConfig(is_multimodal=True, image_column="pictures", conversation_column="dialogue")

        self.assertTrue(config.is_multimodal)
        self.assertEqual(config.image_column, "pictures")
        self.assertEqual(config.conversation_column, "dialogue")

    def test_multimodal_dataset_uses_configured_columns_and_chat_template(self):
        processor = _Processor()
        dataset = _MappedDataset(
            [
                {
                    "pictures": ["image.jpg"],
                    "dialogue": '[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]',
                    "unused": 1,
                }
            ]
        )

        output = _process_multimodal_dataset(
            dataset,
            DataConfig(is_multimodal=True, image_column="pictures", conversation_column="dialogue"),
            processor,
        )

        self.assertEqual(output.rows[0]["text"], "<image>answer")
        self.assertEqual(output.rows[0]["pictures"], ["image.jpg"])
        self.assertEqual(processor.messages[0]["content"][0], {"type": "image"})

    def test_multimodal_collator_loads_images_and_masks_special_tokens(self):
        processor = _Processor()
        collator = _MultimodalCollator(processor, "pictures")
        image_module = types.ModuleType("PIL.Image")

        class _Image:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def convert(self, _mode):
                return self

        image_module.open = lambda _path: _Image()
        pil_module = types.ModuleType("PIL")
        pil_module.Image = image_module
        with mock.patch.dict(sys.modules, {"PIL": pil_module, "PIL.Image": image_module}):
            batch = collator([{"text": "prompt", "pictures": ["image.jpg"]}])

        self.assertEqual(len(processor.calls[0]["images"]), 1)
        torch.testing.assert_close(batch["labels"], torch.tensor([[-100, 4, -100]]))

    def test_multimodal_dataloader_requires_processor(self):
        datasets_module = types.ModuleType("datasets")
        datasets_module.load_dataset = lambda *_args, **_kwargs: _MappedDataset([])
        with mock.patch.dict(sys.modules, {"datasets": datasets_module}):
            with self.assertRaisesRegex(ValueError, "processor"):
                build_dataloader(DataConfig(dataset_path="data.json", is_multimodal=True))

    @mock.patch("fsdp_turbo.training.dataloader.print_rank")
    @mock.patch("fsdp_turbo.distributed.parallel_state.get_parallel_state", return_value=_ParallelState())
    @mock.patch("torch.distributed.is_initialized", return_value=True)
    def test_sampler_uses_semantic_data_parallel_api(self, _initialized, _parallel_state, _print_rank):
        datasets_module = types.ModuleType("datasets")
        datasets_module.load_dataset = lambda *_args, **_kwargs: _Dataset()
        with mock.patch.dict(sys.modules, {"datasets": datasets_module}):
            dataloader = build_dataloader(DataConfig(dataset_path="synthetic", batch_size=1, shuffle=False))

        self.assertEqual(dataloader.sampler.num_replicas, 4)
        self.assertEqual(dataloader.sampler.rank, 2)


if __name__ == "__main__":
    unittest.main()

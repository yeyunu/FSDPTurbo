# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import sys
import types
import unittest
from dataclasses import dataclass
from unittest import mock

import torch

from fsdp_turbo.fsdp_turbo_config import DataConfig
from fsdp_turbo.models.deepseek.deepseek_v41.adapter import DeepseekV41ForCausalLMAdapter
from fsdp_turbo.models.deepseek.deepseek_v41.model import (
    Compressor,
    Indexer,
    ModelArgs,
    shared_attn,
)
from fsdp_turbo.training.dataloader import _MultimodalCollator
from fsdp_turbo.ops.npu.sparse_attention import _cmp_mask_mode


@dataclass
class _ImageInput:
    start: int


class _PackedProcessor:
    def __init__(self):
        self.model_args = types.SimpleNamespace(compress_ratios=(0, 2, 1))
        self.tokenizer = types.SimpleNamespace(pad_token_id=0)

    def __call__(self, *, text, images, **_kwargs):
        length = int(text[0])
        image_inputs = [[_ImageInput(start=1)]] if images else [None]
        return {
            "input_ids": torch.arange(1, length + 1).unsqueeze(0),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
            "token_types": torch.full((1, length), -1, dtype=torch.long),
            "images": image_inputs,
        }


class _Image:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def convert(self, _mode):
        return self


def _image_modules():
    image_module = types.ModuleType("PIL.Image")
    image_module.open = lambda _path: _Image()
    pil_module = types.ModuleType("PIL")
    pil_module.Image = image_module
    return pil_module, image_module


class DeepseekV41TndPackingTests(unittest.TestCase):
    def test_adapter_passes_tnd_boundaries_to_shared_backbone(self):
        class Head(torch.nn.Module):
            def forward(self, hidden, full_logits=False):
                self.full_logits = full_logits
                return hidden

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = Head()

            def forward_backbone(self, input_ids, **kwargs):
                self.call_kwargs = kwargs
                hidden = torch.nn.functional.one_hot(input_ids, 8).float()
                return hidden, None, None

        backbone = Backbone()
        model_args = types.SimpleNamespace(
            vision_enabled=False,
            use_sparse_flash_attn=True,
            compress_ratios=(2,),
            n_layers=1,
            use_fused_compressor=False,
            use_fused_lightning_indexer=False,
        )
        adapter = DeepseekV41ForCausalLMAdapter(backbone, model_args)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        output = adapter(
            input_ids,
            labels=input_ids,
            sfa_layout="TND",
            actual_seq_len=torch.tensor([2, 4], dtype=torch.int32),
            position_ids=torch.tensor([[0, 1, 0, 1]]),
        )

        self.assertEqual(backbone.call_kwargs["sfa_layout"], "TND")
        self.assertEqual(backbone.call_kwargs["cu_seqlens"].tolist(), [0, 2, 4])
        self.assertTrue(backbone.head.full_logits)
        expected_targets = torch.tensor([[2, -100, 4]])
        expected_loss = torch.nn.functional.cross_entropy(
            output.logits[:, :-1].reshape(-1, 8),
            expected_targets.reshape(-1),
            ignore_index=-100,
        )
        torch.testing.assert_close(output.loss, expected_loss)

    def test_cmp_mask_mode_matches_a5_and_compressed_kv_contracts(self):
        with mock.patch.dict("os.environ", {"FSDP_TURBO_CMP_MASK_MODE_WITHOUT_KV": "0"}):
            self.assertEqual(_cmp_mask_mode(False), 0)
            self.assertEqual(_cmp_mask_mode(True), 3)
        with mock.patch.dict("os.environ", {"FSDP_TURBO_CMP_MASK_MODE_WITHOUT_KV": "3"}):
            self.assertEqual(_cmp_mask_mode(False), 3)

    def _collate(self, lengths, max_seq_length=1024):
        processor = _PackedProcessor()
        collator = _MultimodalCollator(
            processor,
            "images",
            data_cfg=DataConfig(max_seq_length=max_seq_length),
        )
        documents = [{"text": str(length), "images": [f"{index}.jpg"]} for index, length in enumerate(lengths)]
        pil_module, image_module = _image_modules()
        with mock.patch.dict(sys.modules, {"PIL": pil_module, "PIL.Image": image_module}):
            return collator._collate_tnd(documents, image_module)

    def test_single_document_remains_supported(self):
        batch = self._collate([5])
        self.assertEqual(batch["actual_seq_len"].tolist(), [6])
        self.assertEqual(batch["input_ids"].shape, (1, 6))

    def test_two_documents_keep_boundaries_positions_and_images(self):
        batch = self._collate([3, 5])
        self.assertEqual(batch["actual_seq_len"].tolist(), [4, 10])
        self.assertEqual(batch["position_ids"].tolist(), [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5]])
        self.assertEqual([image.start for image in batch["images"][0]], [1, 5])  # pylint: disable=not-an-iterable
        self.assertEqual(batch["labels"][0, 3].item(), -100)
        self.assertEqual(batch["labels"][0, 9].item(), -100)

    def test_three_different_document_lengths(self):
        batch = self._collate([2, 3, 4])
        self.assertEqual(batch["actual_seq_len"].tolist(), [2, 6, 10])
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertEqual(batch["input_ids"].shape, batch["position_ids"].shape)

    def test_bsnd_and_tnd_use_same_documents_and_loss_targets(self):
        processor = _PackedProcessor()
        collator = _MultimodalCollator(processor, "images", data_cfg=DataConfig(max_seq_length=1024))
        documents = [{"text": str(length), "images": [f"{index}.jpg"]} for index, length in enumerate((3, 5, 7))]
        pil_module, image_module = _image_modules()
        with mock.patch.dict(sys.modules, {"PIL": pil_module, "PIL.Image": image_module}):
            tnd = collator._collate_tnd(documents, image_module)
            bsnd = collator._collate_bsnd_documents(documents, image_module)

        lengths = torch.diff(torch.nn.functional.pad(tnd["actual_seq_len"], (1, 0))).tolist()
        self.assertEqual(lengths, [4, 6, 8])
        self.assertEqual(bsnd["input_ids"].shape, (3, 8))
        for index, length in enumerate(lengths):
            start = sum(lengths[:index])
            torch.testing.assert_close(
                bsnd["input_ids"][index, :length],
                tnd["input_ids"][0, start : start + length],
            )
            torch.testing.assert_close(
                bsnd["attention_mask"][index, :length],
                tnd["attention_mask"][0, start : start + length],
            )

        # The adapter masks each next-document start for TND; BSND excludes
        # the first token of every row by shifting along the sequence axis.
        tnd_labels = tnd["labels"].clone()
        tnd_labels[0, tnd["actual_seq_len"][:-1].long()] = -100
        tnd_targets = tnd_labels[:, 1:]
        bsnd_targets = bsnd["labels"][:, 1:]
        self.assertEqual(
            tnd_targets[tnd_targets != -100].tolist(),
            bsnd_targets[bsnd_targets != -100].tolist(),
        )
        self.assertEqual(len(bsnd["images"]), 3)
        self.assertEqual([images[0].start for images in bsnd["images"]], [1, 1, 1])
        self.assertEqual(bsnd["sfa_layout"], "BSND")

    def test_pack_over_limit_is_rejected_after_alignment(self):
        with self.assertRaisesRegex(ValueError, "exceeds data.max_seq_length"):
            self._collate([5, 5], max_seq_length=10)

    def test_compressor_never_pools_across_document_boundary(self):
        args = ModelArgs(
            dim=4,
            head_dim=2,
            compress_ratios=(2,),
            n_layers=1,
            n_mtp_layers=0,
            use_fused_compressor=False,
        )
        compressor = Compressor(args, 0)
        compressor.train()
        with torch.no_grad():
            for parameter in compressor.parameters():
                parameter.fill_(0.1)
        packed = torch.randn(1, 10, 4, requires_grad=True)
        cu_seqlens = torch.tensor([0, 4, 10], dtype=torch.int32)

        expected = torch.cat((compressor(packed[:, :4], 0), compressor(packed[:, 4:], 0)), dim=1)
        actual = compressor(packed, 0, cu_seqlens=cu_seqlens)

        torch.testing.assert_close(actual, expected)
        actual.float().sum().backward()
        self.assertTrue(torch.isfinite(packed.grad).all())

    def test_indexer_uses_document_local_compressed_indices(self):
        args = ModelArgs(
            dim=4,
            head_dim=4,
            rope_head_dim=2,
            q_lora_rank=4,
            n_heads=2,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=3,
            compress_ratios=(2,),
            kv_source_layers=(0,),
            index_source_layers=(0,),
            n_layers=1,
            n_mtp_layers=0,
        )
        indexer = Indexer(args, 0)
        with torch.no_grad():
            for parameter in indexer.parameters():
                parameter.fill_(0.1)
        indexer.freqs_cis = torch.polar(
            torch.ones(args.max_seq_len, args.rope_head_dim // 2),
            torch.zeros(args.max_seq_len, args.rope_head_dim // 2),
        )
        x = torch.randn(1, 10, args.dim, dtype=torch.bfloat16)
        qr = torch.randn(1, 10, args.q_lora_rank, dtype=torch.bfloat16)
        latent = torch.randn(1, 5, args.head_dim, dtype=torch.bfloat16)

        expected_parts = []
        for token_start, token_end, compressed_start, compressed_end in (
            (0, 4, 0, 2),
            (4, 10, 2, 5),
        ):
            indices = indexer(
                x[:, token_start:token_end],
                qr[:, token_start:token_end],
                latent[:, compressed_start:compressed_end],
                0,
                0,
            )
            expected_parts.append(torch.nn.functional.pad(indices, (0, args.index_topk - indices.shape[-1]), value=-1))
        expected = torch.cat(expected_parts, dim=1)

        actual = indexer(x, qr, latent, 0, 0, cu_seqlens=torch.tensor([0, 4, 10], dtype=torch.int32))

        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.all(actual[:, :4] < 2))
        self.assertTrue(torch.all(actual[:, 4:][actual[:, 4:] >= 0] < 3))

    def test_candidate_masks_are_kept_per_document(self):
        args = ModelArgs(
            dim=4,
            head_dim=4,
            rope_head_dim=2,
            q_lora_rank=4,
            n_heads=2,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=3,
            compress_ratios=(1, 1),
            kv_source_layers=(0,),
            index_source_layers=(0, 1),
            candidate_source_layer=0,
            candidate_topk_blocks=2,
            candidate_block_size=1,
            n_layers=2,
            n_mtp_layers=0,
        )
        source = Indexer(args, 0)
        consumer = Indexer(args, 1)
        freqs = torch.polar(
            torch.ones(args.max_seq_len, args.rope_head_dim // 2),
            torch.zeros(args.max_seq_len, args.rope_head_dim // 2),
        )
        source.freqs_cis = freqs
        consumer.freqs_cis = freqs
        with torch.no_grad():
            for module in (source, consumer):
                for parameter in module.parameters():
                    parameter.fill_(0.1)
        x = torch.randn(1, 7, args.dim, dtype=torch.bfloat16)
        qr = torch.randn(1, 7, args.q_lora_rank, dtype=torch.bfloat16)
        latent = torch.randn(1, 7, args.head_dim, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

        shared_attn.reset()
        source(x, qr, latent, 0, 0, cu_seqlens=cu_seqlens)
        self.assertEqual(len(shared_attn.packed_candidates), 2)
        indices = consumer(x, qr, None, 0, 0, cu_seqlens=cu_seqlens)

        self.assertEqual(indices.shape, (1, 7, args.index_topk))
        self.assertTrue(torch.all(indices[:, :3][indices[:, :3] >= 0] < 3))
        self.assertTrue(torch.all(indices[:, 3:][indices[:, 3:] >= 0] < 4))


if __name__ == "__main__":
    unittest.main()

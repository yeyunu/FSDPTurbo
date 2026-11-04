# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Pin the MTP causal-mask branch (no S^2 mask for GDN layers).

``MultiTokenPredictionBlock.forward`` used to call ``create_causal_mask``
unconditionally at full sequence length -- before the CP split -- materializing
a ``[B, 1, S, S]`` bool mask.  For ``linear_attention`` (GDN) decoder layers,
the only layer type qwen3.8's MTP uses, that mask is never read (the decoder
layer forwards only ``conv_mask`` to GDN, and MTP passes none), so it was pure
S^2 device-memory waste.  The branches locked here:

* ``linear_attention``: no mask materialized, the layer receives ``None``
  (bit-identical to the old code, which computed a mask nobody read).
* ``full_attention`` + CP: the layer receives the full padded 2-D validity
  mask (the CP-patched QSA forward contract); ``None`` becomes valid tokens followed by a masked alignment tail.
* ``full_attention`` without CP: stock ``create_causal_mask`` result is passed
  through unchanged.
"""

import unittest
from unittest import mock

import torch
from torch import nn

from fsdp_turbo.models.qwen import modeling_qwen4_exp as mtp_mod
from fsdp_turbo.models.qwen.modeling_qwen4_exp import (
    Qwen4ExpMultiTokenPredictionBlock as MultiTokenPredictionBlock,
)

HIDDEN = 8
VOCAB = 16


class _StubConfig:
    mtp_num_layers = 1
    hidden_size = HIDDEN
    rms_norm_eps = 1e-6
    hc_count = None
    indexer_compress_ratio = 1
    vocab_size = VOCAB

    def __init__(self, layer_type):
        self.stub_layer_type = layer_type


class _StubNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

    def forward(self, x):
        return x


class _StubDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=0, is_mtp=False):
        super().__init__()
        self.layer_type = config.stub_layer_type
        self.received_attention_mask = "unset"

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        self.received_attention_mask = attention_mask
        return hidden_states


class _StubRotary(nn.Module):
    def forward(self, hidden_states, position_ids):
        table = torch.ones(1, 1, 1)
        return table, table


class _StubHead(nn.Module):
    def forward(self, x):
        return x


class _StubParallelState:
    def __init__(self, cp_size):
        self._cp_size = cp_size

    def is_group_enable(self, name):
        return name == "cp" and self._cp_size > 1

    def get_cp_group_size(self):
        return self._cp_size

    def get_cp_group(self):
        return object()


def _run_mtp(cp_size, layer_type, attention_mask, seq_len=10):
    """Run one MTP forward with stubbed CP primitives.

    Returns ``(decoder_layer, losses, causal_mask_calls, full_len, sentinel)``.
    ``hidden_states`` is sized to the padded full length so the stubbed
    (no-op) CP split keeps the inputs_embeds/hidden cat consistent.
    """
    config = _StubConfig(layer_type)
    block = MultiTokenPredictionBlock(config, _StubDecoderLayer, _StubNorm)
    pad_multiple = cp_size * int(config.indexer_compress_ratio)
    full_len = seq_len + (-seq_len) % pad_multiple
    torch.manual_seed(0)
    hidden = torch.randn(1, full_len, HIDDEN)
    input_ids = torch.randint(0, VOCAB, (1, seq_len))
    labels = torch.ones(1, seq_len, dtype=torch.long)
    embed = nn.Embedding(VOCAB, HIDDEN)
    causal_calls = []
    sentinel = torch.full((1, 1, 1, 1), 7.0)

    def fake_create_causal_mask(**kwargs):
        causal_calls.append(kwargs)
        return sentinel

    state = _StubParallelState(cp_size)
    with (
        mock.patch.object(mtp_mod, "get_parallel_state", return_value=state),
        mock.patch.object(mtp_mod, "_split_forward_gather_backward", side_effect=lambda *a, **k: a[0]),
        mock.patch.object(mtp_mod, "_gather_forward_split_backward", side_effect=lambda *a, **k: a[0]),
        mock.patch.object(mtp_mod, "split_tensor_to_group", side_effect=lambda *a, **k: a[0]),
        mock.patch.object(mtp_mod, "create_causal_mask", side_effect=fake_create_causal_mask),
    ):
        losses = block.forward(
            hidden,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            embed_tokens=embed,
            rotary_emb=_StubRotary(),
            output_layer=_StubHead(),
            loss_function=lambda logits, labels, vocab_size: logits.sum(),
        )
    return block.layers[0], losses, causal_calls, full_len, sentinel


class MTPCausalMaskBranchTest(unittest.TestCase):
    def test_gdn_layer_never_materializes_mask_under_cp(self):
        mask = torch.ones(1, 10, dtype=torch.bool)
        mask[0, 7:] = False
        layer, losses, causal_calls, _, _ = _run_mtp(4, "linear_attention", mask)
        self.assertEqual(causal_calls, [])
        self.assertIsNone(layer.received_attention_mask)
        self.assertEqual(len(losses), 1)
        self.assertTrue(bool(torch.isfinite(losses[0])))

    def test_gdn_layer_skips_mask_without_cp(self):
        mask = torch.ones(1, 10, dtype=torch.bool)
        layer, _, causal_calls, _, _ = _run_mtp(1, "linear_attention", mask)
        self.assertEqual(causal_calls, [])
        self.assertIsNone(layer.received_attention_mask)

    def test_full_attention_cp_gets_padded_validity(self):
        mask = torch.ones(1, 10, dtype=torch.bool)
        mask[0, 7:] = False
        layer, _, causal_calls, full_len, _ = _run_mtp(4, "full_attention", mask)
        # 10 padded to a multiple of cp * compress_ratio (4) -> 12.
        self.assertEqual(full_len, 12)
        self.assertEqual(causal_calls, [])
        received = layer.received_attention_mask
        self.assertEqual(tuple(received.shape), (1, 12))
        expected = torch.ones(1, 12, dtype=torch.bool)
        expected[0, 7:] = False
        self.assertTrue(torch.equal(received, expected))

    def test_full_attention_cp_masks_padding_when_mask_is_none(self):
        layer, _, causal_calls, _, _ = _run_mtp(4, "full_attention", None)
        self.assertEqual(causal_calls, [])
        received = layer.received_attention_mask
        self.assertEqual(tuple(received.shape), (1, 12))
        self.assertEqual(received.dtype, torch.bool)
        self.assertTrue(bool(received[:, :10].all()))
        self.assertFalse(bool(received[:, 10:].any()))

    def test_full_attention_without_cp_keeps_stock_causal_mask(self):
        mask = torch.ones(1, 10, dtype=torch.bool)
        layer, _, causal_calls, _, sentinel = _run_mtp(1, "full_attention", mask)
        self.assertEqual(len(causal_calls), 1)
        self.assertIs(layer.received_attention_mask, sentinel)
        self.assertEqual(tuple(causal_calls[0]["inputs_embeds"].shape), (1, 10, HIDDEN))
        self.assertTrue(torch.equal(causal_calls[0]["attention_mask"], mask))


if __name__ == "__main__":
    unittest.main()

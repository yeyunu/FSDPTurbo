# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Regression checks for the consolidated Qwen long-sequence adapter."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fsdp_turbo.memory.chunk_loss.chunk_loss import calculate_lm_loss, chunk_loss
from fsdp_turbo.models.qwen import modeling_qwen4_exp as model_mod
from fsdp_turbo.models.qwen.modeling_qwen4_exp import (
    Qwen4ExpChunkedLMHead,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
    Qwen4ExpMultiTokenPredictionBlock,
    _shift_and_align_labels,
)


@pytest.mark.parametrize('ignored', [False, True])
@pytest.mark.parametrize('chunk_size', [1, 4, 32])
def test_chunk_head_matches_dense_loss_and_gradients(ignored, chunk_size):
    torch.manual_seed(17)
    head = Qwen4ExpChunkedLMHead(8, 13, chunk_loss, calculate_lm_loss)
    weight = head.weight.detach().clone().requires_grad_()
    hidden = torch.randn(2, 9, 8, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    labels = torch.randint(0, 13, (2, 9))
    labels[:, -2:] = -100
    if ignored:
        labels.fill_(-100)
    shifted = _shift_and_align_labels(labels, hidden, None)
    count = shifted.ne(-100).sum().clamp_min(1)
    expected = (
        F.cross_entropy((reference_hidden @ weight.T).float().flatten(0, 1), shifted.flatten(), reduction='sum') / count
    )
    actual = head(hidden, chunk_loss_labels=shifted, chunk_loss_size=chunk_size) / count
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(head.weight.grad, weight.grad)
    assert list(head.state_dict()) == ['weight']
    torch.testing.assert_close(head(hidden.detach()), (hidden.detach() @ head.weight.T))


def test_labels_shift_before_cp_split():
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    hidden = torch.zeros(1, 4, 2)
    for rank, expected in enumerate(([2, 3, 4, 5], [-100, -100, -100, -100])):
        with (
            mock.patch.object(model_mod.dist, 'get_world_size', return_value=2),
            mock.patch.object(model_mod.dist, 'get_rank', return_value=rank),
        ):
            actual = _shift_and_align_labels(labels, hidden, object())
        assert actual.tolist() == [expected]
    with pytest.raises(ValueError, match='exceeds global hidden-state'):
        _shift_and_align_labels(labels, hidden, None)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(13, 8)
        self.calls = 0
        self.received = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        self.calls += 1
        self.received = dict(inputs_embeds=inputs_embeds, **kwargs)
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        return model_mod.Qwen4ExpModelOutputWithPast(last_hidden_state=hidden, hidden_states=(hidden,))


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()

    def forward(self, **kwargs):
        return self.language_model(**kwargs)


class _MinimalGenerationModel(Qwen4ExpForConditionalGeneration):
    """Keep the real forward while avoiding the full pretrained model allocation."""

    __init__ = nn.Module.__init__


class _MinimalQwenModel(Qwen4ExpModel):
    """Initialize only module state for the backbone alignment test."""

    __init__ = nn.Module.__init__


class _MinimalMTPBlock(Qwen4ExpMultiTokenPredictionBlock):
    """Exercise the inference early return without allocating training layers."""

    __init__ = nn.Module.__init__


def _generation_model():
    # Exercise the real generation forward with a tiny backbone and the real chunk kernel.
    model = _MinimalGenerationModel()
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(chunk_loss_size=3), use_return_dict=True, return_dict=True
    )
    model.model = _Backbone()
    model.lm_head = Qwen4ExpChunkedLMHead(8, 13, chunk_loss, calculate_lm_loss)
    model.enable_chunk_loss = True
    model.enable_mtp = True
    return model


def test_chunk_and_mtp_share_one_backbone_pass_and_accept_positional_ids():
    model = _generation_model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with mock.patch.object(model, '_compute_mtp_loss', return_value=[torch.tensor(0.5)]) as mtp:
        output = model(ids, labels=ids)
    assert model.model.language_model.calls == 1
    assert output.logits is None
    assert output.loss.isfinite()
    assert output.mtp_loss is model._last_mtp_loss
    torch.testing.assert_close(mtp.call_args.kwargs['input_ids'], ids)
    output.loss.backward()
    assert model.lm_head.weight.grad is not None
    assert model.model.language_model.embed_tokens.weight.grad is not None
    assert not model.model.language_model._forward_hooks


def test_chunk_forward_respects_tuple_output_and_global_denominator():
    model = _generation_model()
    model.enable_mtp = False
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    base = model(ids, labels=ids)
    explicit = model(ids, labels=ids, num_items_in_batch=8, return_dict=False)
    assert isinstance(explicit, tuple)
    torch.testing.assert_close(explicit[0], base.loss / 2)
    with pytest.raises(ValueError, match='logits_to_keep=0'):
        model(ids, labels=ids, logits_to_keep=1)
    assert not model.model.language_model._forward_hooks


def test_post_fusion_cp_alignment_and_masks():
    model = _MinimalQwenModel()
    model.language_model = _LanguageModel()
    model.language_model.config = SimpleNamespace(indexer_compress_ratio=4)
    model.rope_deltas = None
    state = SimpleNamespace(
        is_group_enable=lambda name: True,
        get_cp_group_size=lambda: 2,
        get_cp_group=object,
    )
    ids = torch.arange(10).view(1, -1)
    positions = torch.arange(10).view(1, 1, -1).expand(3, 1, -1)
    validity = torch.ones(1, 10, dtype=torch.bool)
    validity[:, -1] = False
    # Marker represents the framework-installed specialized QSA forward.
    with (
        mock.patch.object(model_mod, 'get_parallel_state', return_value=state),
        mock.patch.object(_LanguageModel.forward, '_qwen4_qsa_ulysses', True, create=True),
        mock.patch.object(model_mod.dist, 'get_world_size', return_value=2),
        mock.patch.object(model_mod.dist, 'get_rank', return_value=1),
    ):
        output = model(input_ids=ids, position_ids=positions, attention_mask=validity)
    received = model.language_model.received
    assert output.last_hidden_state.shape == (1, 8, 8)
    assert received['ple_input_ids'].tolist() == [[8, 9, 0, 0, 0, 0, 0, 0]]
    assert received['attention_mask']['full_attention'].shape == (1, 16)
    assert received['attention_mask']['linear_attention'].tolist() == [
        [True, False, False, False, False, False, False, False]
    ]
    assert received['position_ids'].shape == (3, 1, 8)


def test_mtp_inference_does_not_require_training_labels():
    block = _MinimalMTPBlock()
    assert block(torch.zeros(1, 2, 4)) is None


def _tiny_config():
    return model_mod.Qwen4ExpConfig.from_dict(
        dict(
            text_config=dict(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                layer_types=['full_attention'],
                moe_intermediate_size=16,
                shared_expert_intermediate_size=16,
                num_experts=2,
                num_experts_per_tok=1,
                hc_count=2,
                hc_lowrank=4,
                ple_layer_ids=[],
                use_cache=False,
                indexer_n_heads=2,
                indexer_kv_heads=1,
                indexer_head_dim=8,
                indexer_budget=4,
                indexer_compress_ratio=2,
                rope_parameters=dict(rope_type='default', rope_theta=10000.0, mrope_section=[1, 1, 2]),
            ),
            vision_config=dict(
                depth=1,
                hidden_size=16,
                intermediate_size=16,
                num_heads=2,
                out_hidden_size=16,
                num_position_embeddings=16,
            ),
        )
    )


def test_complete_model_dense_chunk_equivalence_and_checkpoint_roundtrip(monkeypatch, tmp_path):
    state = SimpleNamespace(is_group_enable=lambda name: False, get_cp_group_size=lambda: 1)
    monkeypatch.setattr(model_mod, 'get_parallel_state', lambda: state)
    config = _tiny_config()
    config.text_config.mtp_num_layers = 2
    config.text_config.enable_chunk_loss = False
    dense = Qwen4ExpForConditionalGeneration(config)
    config = _tiny_config()
    config.text_config.mtp_num_layers = 2
    config.text_config.enable_chunk_loss = True
    config.text_config.chunk_loss_size = 3
    chunked = Qwen4ExpForConditionalGeneration(
        config, chunk_loss_kernel=chunk_loss, chunk_loss_forward=calculate_lm_loss
    )
    chunked.load_state_dict(dense.state_dict(), strict=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    positions = torch.arange(6).view(1, 1, -1).expand(3, 1, -1)
    dense_output = dense(ids, labels=ids, position_ids=positions, use_cache=False)
    chunk_output = chunked(ids, labels=ids, position_ids=positions, use_cache=False)
    torch.testing.assert_close(chunk_output.loss, dense_output.loss)
    for actual, expected in zip(chunked._last_mtp_loss, dense._last_mtp_loss, strict=True):
        torch.testing.assert_close(actual, expected)
    (dense_output.loss + sum(dense._last_mtp_loss)).backward()
    (chunk_output.loss + sum(chunked._last_mtp_loss)).backward()
    for (name, actual), (_, expected) in zip(chunked.named_parameters(), dense.named_parameters(), strict=True):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-4, msg=name)
    chunked.save_pretrained(tmp_path)
    restored = Qwen4ExpForConditionalGeneration.from_pretrained(
        tmp_path, chunk_loss_kernel=chunk_loss, chunk_loss_forward=calculate_lm_loss
    )
    restored.eval()
    chunked.eval()
    with torch.no_grad():
        expected = chunked(ids, position_ids=positions, use_cache=False).logits
        actual = restored(ids, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
    assert restored._last_mtp_loss is None
    restored.config.return_dict = False
    assert isinstance(restored(ids, position_ids=positions, use_cache=False), tuple)


def _cp_chunk_worker(rank, rendezvous):
    torch.distributed.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(31)
        head = Qwen4ExpChunkedLMHead(8, 13, chunk_loss, calculate_lm_loss)
        reference_weight = head.weight.detach().clone().requires_grad_()
        full_hidden = torch.randn(1, 16, 8, requires_grad=True)
        local_hidden = full_hidden.detach()[:, rank * 8 : (rank + 1) * 8].clone().requires_grad_()
        labels = torch.randint(0, 13, (1, 11))
        # Rank 1 has no supervised tokens: clamp only AFTER the global count reduction.
        labels[:, 7:] = -100
        shifted = _shift_and_align_labels(labels, full_hidden, None)
        expected = (
            F.cross_entropy((full_hidden @ reference_weight.T).flatten(0, 1), shifted.flatten(), reduction='sum')
            / shifted.ne(-100).sum()
        )
        state = SimpleNamespace(
            is_group_enable=lambda name: name == 'cp',
            get_cp_group_size=lambda: 2,
            get_cp_group=lambda: torch.distributed.group.WORLD,
        )
        with mock.patch.object(model_mod, 'get_parallel_state', return_value=state):
            actual = head(local_hidden, loss_function=lambda: None, labels=labels)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        # Shared weight gradients match dense after the FSDP mean reduction.
        torch.distributed.all_reduce(head.weight.grad)
        torch.testing.assert_close(head.weight.grad / 2, reference_weight.grad)
        torch.testing.assert_close(local_hidden.grad / 2, full_hidden.grad[:, rank * 8 : (rank + 1) * 8])
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_cp_chunk_loss_and_gradients(tmp_path):
    torch.multiprocessing.spawn(_cp_chunk_worker, args=((tmp_path / 'cp-init').as_uri(),), nprocs=2, join=True)

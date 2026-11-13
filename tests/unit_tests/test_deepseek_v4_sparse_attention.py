from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from fsdp_turbo.models.deepseek.deepseek_v4 import modeling
from fsdp_turbo.models.deepseek.deepseek_v4.modeling import _deepseek_v4_sparse_attention


def _attention_inputs(layer_type: str):
    batch_size, sequence_length, num_heads, head_dim = 1, 8, 4, 8
    compress_ratio = 1 if layer_type == "sliding_attention" else 4
    compressed_length = 0 if compress_ratio == 1 else sequence_length // compress_ratio
    query = torch.randn(batch_size, num_heads, sequence_length, head_dim, requires_grad=True)
    original = torch.randn(batch_size, 1, sequence_length, head_dim, requires_grad=True)
    compressed = torch.randn(batch_size, 1, compressed_length, head_dim, requires_grad=True)
    positions = torch.arange(sequence_length)
    original_allowed = (positions[None, :] <= positions[:, None]) & (positions[None, :] > positions[:, None] - 3)
    mask = torch.zeros(1, 1, sequence_length, sequence_length).masked_fill(~original_allowed[None, None], -torch.inf)
    indices = None
    if layer_type == "compressed_sparse_attention":
        indices = torch.where(positions >= 3, 0, -1).view(1, sequence_length, 1).to(torch.int32)
    module = SimpleNamespace(
        compressor=SimpleNamespace(compress_rate=compress_ratio) if compressed_length else None,
        config=SimpleNamespace(index_topk=1),
        layer_type=layer_type,
        sinks=torch.randn(num_heads, requires_grad=True),
        sliding_window=3,
    )
    return module, query, original, mask, compressed if compressed_length else None, indices


@pytest.mark.parametrize(
    "layer_type", ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
)
def test_attention_forward_allows_empty_cache_but_rejects_cached_tokens(monkeypatch, layer_type):
    hidden_states = SimpleNamespace(device=SimpleNamespace(type="npu"), shape=(1, 3, 8))
    query = torch.randn(1, 2, 3, 4)
    key_value = torch.randn(1, 1, 3, 4)

    compressed = torch.randn(1, 1, 1, 4) if layer_type != "sliding_attention" else None
    indices = torch.zeros(1, 3, 1, dtype=torch.int32) if layer_type == "compressed_sparse_attention" else None
    compressor = mock.Mock(return_value=(compressed, indices)) if compressed is not None else None
    attention_mask = torch.zeros(1, 1, 3, 3)
    bridge = mock.Mock(return_value=(query.transpose(1, 2), None))

    def identity(value):
        return value

    module = SimpleNamespace(
        config=SimpleNamespace(use_sparse_flash_attn=True, o_groups=1),
        head_dim=4,
        rope_layer_type="main",
        q_a_norm=identity,
        q_a_proj=identity,
        q_b_proj=lambda value: query.transpose(1, 2).contiguous().view(1, 3, 8),
        q_b_norm=identity,
        kv_norm=identity,
        kv_proj=lambda value: key_value.transpose(1, 2).contiguous().view(1, 3, 4),
        compressor=compressor,
        layer_idx=0,
        scaling=0.5,
        training=False,
        attention_dropout=0.0,
        o_a_proj=identity,
        o_b_proj=identity,
    )
    monkeypatch.setattr(modeling, "apply_rotary_pos_emb", lambda value, cos, sin: value)
    monkeypatch.setattr(modeling, "_deepseek_v4_sparse_attention", bridge)

    output, weights = modeling.DeepseekV4Attention.forward(
        module,
        hidden_states,
        {"main": (None, 0)},
        torch.arange(3).unsqueeze(0),
        attention_mask,
        SimpleNamespace(get_seq_length=lambda: 0),
    )

    assert output.shape == hidden_states.shape
    assert weights is None
    assert bridge.call_args.args[2].shape == key_value.shape
    assert bridge.call_args.args[3] is attention_mask
    assert bridge.call_args.kwargs["compressed_key_value"] is compressed
    assert bridge.call_args.kwargs["compressed_topk_indices"] is indices
    if compressor is not None:
        compressor.assert_called_once()
        assert compressor.call_args.kwargs == {"return_indices": True}
    with pytest.raises(ValueError, match="non-empty KV cache"):
        modeling.DeepseekV4Attention.forward(
            module,
            hidden_states,
            {"main": (None, 0)},
            torch.arange(3).unsqueeze(0),
            None,
            SimpleNamespace(get_seq_length=lambda: 1),
        )


@pytest.mark.parametrize(
    ("layer_type", "expected_ratio", "expects_indices"),
    [
        ("sliding_attention", 1, False),
        ("compressed_sparse_attention", 4, True),
        ("heavily_compressed_attention", 4, False),
    ],
)
def test_sparse_attention_routes_v4_kv_sources_separately(monkeypatch, layer_type, expected_ratio, expects_indices):
    module, query, key_value, mask, compressed, indices = _attention_inputs(layer_type)
    captured = {}

    def sparse_attention(query, key_value, attention_sink, topk_indices, softmax_scale, **kwargs):
        captured.update(
            query=query,
            key_value=key_value,
            attention_sink=attention_sink,
            topk_indices=topk_indices,
            softmax_scale=softmax_scale,
            **kwargs,
        )
        output = query
        if kwargs["compressed_key_value"] is not None:
            output = output + kwargs["compressed_key_value"].mean(dim=1)[:, None, None, :]
        return output

    monkeypatch.setattr(
        "fsdp_turbo.models.deepseek.deepseek_v4.modeling.indexed_sparse_attention",
        sparse_attention,
    )
    output, weights = _deepseek_v4_sparse_attention(
        module,
        query,
        key_value,
        mask,
        query.shape[-1] ** -0.5,
        compressed_key_value=compressed,
        compressed_topk_indices=indices,
    )

    assert output.shape == query.transpose(1, 2).shape
    assert weights is None
    assert captured["key_value"].shape == (1, 8, 8)
    assert captured["compress_ratio"] == expected_ratio
    compressed = captured["compressed_key_value"]
    compressed_indices = captured["compressed_topk_indices"]
    assert (compressed is not None) == (layer_type != "sliding_attention")
    assert (compressed_indices is not None) is expects_indices
    if compressed_indices is not None:
        assert compressed_indices.dtype == torch.int32
        assert compressed_indices is indices
    output.square().sum().backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


def test_sparse_attention_casts_fp32_kv_to_query_dtype(monkeypatch):
    module, query, key_value, mask, compressed, indices = _attention_inputs("compressed_sparse_attention")
    query = query.detach().to(torch.bfloat16).requires_grad_()
    key_value = key_value.detach().requires_grad_()
    captured = {}

    def sparse_attention(query, key_value, attention_sink, topk_indices, softmax_scale, **kwargs):
        captured.update(key_value=key_value, **kwargs)
        return query + (key_value.mean(dim=1) + kwargs["compressed_key_value"].mean(dim=1))[:, None, None, :]

    monkeypatch.setattr(modeling, "indexed_sparse_attention", sparse_attention)
    output, _ = _deepseek_v4_sparse_attention(
        module,
        query,
        key_value,
        mask,
        query.shape[-1] ** -0.5,
        compressed_key_value=compressed,
        compressed_topk_indices=indices,
    )

    assert captured["key_value"].dtype == torch.bfloat16
    assert captured["compressed_key_value"].dtype == torch.bfloat16
    output.sum().backward()
    assert key_value.grad is not None
    assert torch.isfinite(key_value.grad).all()


def test_a2_or_a3_detection_uses_npu_device_name(monkeypatch):
    from fsdp_turbo.ops.npu import sparse_attention

    monkeypatch.setattr(torch.npu, "get_device_name", lambda index: "Ascend910_9392", raising=False)
    sparse_attention._is_a2_or_a3.cache_clear()
    try:
        assert sparse_attention._is_a2_or_a3(0)
    finally:
        sparse_attention._is_a2_or_a3.cache_clear()


def test_sparse_flash_mla_accepts_hca_compressed_kv_without_indices():
    from fsdp_turbo.ops.npu import sparse_attention

    query = torch.randn(1, 128, 2, 512, dtype=torch.bfloat16)
    original_key_value = torch.randn(1, 128, 512, dtype=torch.bfloat16)
    compressed_key_value = torch.randn(1, 1, 512, dtype=torch.bfloat16)
    attention_sink = torch.randn(2)
    sentinel = object()

    with mock.patch.object(sparse_attention._SparseFlashMlaFunction, "apply", return_value=sentinel) as apply:
        output = sparse_attention.npu_sparse_flash_mla(
            query,
            original_key_value,
            compressed_key_value,
            None,
            attention_sink,
            512**-0.5,
            128,
            128,
        )

    assert output is sentinel
    arguments = apply.call_args.args
    assert arguments[2].shape == (1, 1, 1, 512)
    assert arguments[3] is None


def test_sparse_flash_mla_metadata_uses_zero_topk_for_hca():
    from fsdp_turbo.ops.npu import sparse_attention

    query = torch.randn(1, 128, 2, 512, dtype=torch.bfloat16)
    original_key_value = torch.randn(1, 128, 1, 512, dtype=torch.bfloat16)
    compressed_key_value = torch.randn(1, 1, 1, 512, dtype=torch.bfloat16)
    attention_sink = torch.randn(2)
    result = torch.empty_like(query)
    softmax_lse = torch.empty(1)
    operations = SimpleNamespace(sparse_flash_mla=lambda *args, **kwargs: (result, softmax_lse))

    class Context:
        metadata_kwargs: dict
        has_compressed_kv: bool
        has_compressed_indices: bool

        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors

        def mark_non_differentiable(self, tensor):
            self.non_differentiable = tensor

    context = Context()
    with (
        mock.patch.object(sparse_attention, "_forward_metadata", return_value=object()),
        mock.patch.object(sparse_attention, "_get_cann_ops", return_value=operations),
    ):
        output = sparse_attention._SparseFlashMlaFunction.forward(
            context,
            query,
            original_key_value,
            compressed_key_value,
            None,
            attention_sink,
            512**-0.5,
            128,
            128,
            "BSND",
            None,
        )

    assert output is result
    assert context.metadata_kwargs["topk"] == 0
    assert context.has_compressed_kv
    assert not context.has_compressed_indices
    assert context.saved_tensors[-1] is compressed_key_value


@pytest.mark.parametrize("blocked_value", [-torch.inf, torch.finfo(torch.float32).min])
def test_sparse_attention_preserves_sliding_mask_validation(blocked_value, monkeypatch):
    monkeypatch.setenv("FSDP_TURBO_VALIDATE_SPARSE_FLASH_MLA", "1")
    module, query, key_value, mask, compressed, indices = _attention_inputs("compressed_sparse_attention")
    mask = mask.masked_fill(torch.isneginf(mask), blocked_value)
    with mock.patch.object(modeling, "indexed_sparse_attention", return_value=query.transpose(1, 2)) as kernel:
        _deepseek_v4_sparse_attention(
            module,
            query,
            key_value,
            mask,
            0.5,
            compressed_key_value=compressed,
            compressed_topk_indices=indices,
        )
        kernel.assert_called_once()
    mask[..., -1, -1] = 0.01
    with pytest.raises(NotImplementedError, match="additive sliding-window bias"):
        _deepseek_v4_sparse_attention(
            module,
            query,
            key_value,
            mask,
            0.5,
            compressed_key_value=compressed,
            compressed_topk_indices=indices,
        )


def test_hca_rejects_explicit_compressed_indices():
    module, query, key_value, mask, compressed, _ = _attention_inputs("heavily_compressed_attention")
    with pytest.raises(ValueError, match="compressed_topk_indices=None"):
        _deepseek_v4_sparse_attention(
            module,
            query,
            key_value,
            mask,
            0.5,
            compressed_key_value=compressed,
            compressed_topk_indices=torch.zeros(1, 8, 1, dtype=torch.int32),
        )


@pytest.mark.parametrize("layout", ["BSND", "TND"])
@pytest.mark.parametrize(
    ("original_length", "compressed_length", "compress_ratio", "error"),
    [
        (256, 2, 128, None),
        (257, 2, 128, None),
        (256, 1, 128, "ori_len == cmp_len|compressed KV length"),
        (256, 3, 128, "ori_len == cmp_len|compressed KV length"),
        (257, 1, 128, "ori_len == cmp_len|compressed KV length"),
        (256, 256, 1, "HCA requires cmp_ratio=128"),
        (256, 64, 4, "HCA requires cmp_ratio=128"),
        (256, 16, 16, "HCA requires cmp_ratio=128"),
        (256, 4, 64, "HCA requires cmp_ratio=128"),
    ],
)
def test_sparse_flash_mla_hca_layout_contract(layout, original_length, compressed_length, compress_ratio, error):
    """Reject incomplete HCA prefixes and unsupported ratios before kernel launch."""
    from fsdp_turbo.ops.npu import sparse_attention

    query = torch.empty(1, original_length, 2, 512, dtype=torch.bfloat16)
    original = torch.empty(1, original_length, 512, dtype=torch.bfloat16)
    compressed = torch.empty(1, compressed_length, 512, dtype=torch.bfloat16)
    cu_seqlens = None
    if layout == "TND":
        query, original, compressed = query[0], original[0], compressed[0]
        cu_seqlens = torch.tensor([0, original_length], dtype=torch.int32)
    sentinel = object()
    with mock.patch.object(sparse_attention._SparseFlashMlaFunction, "apply", return_value=sentinel) as kernel:
        arguments = (query, original, compressed, None, torch.zeros(2), 512**-0.5, compress_ratio, 128)
        if error is not None:
            with pytest.raises(ValueError, match=error):
                sparse_attention.npu_sparse_flash_mla(*arguments, layout=layout, cu_seqlens=cu_seqlens)
            kernel.assert_not_called()
        else:
            assert sparse_attention.npu_sparse_flash_mla(*arguments, layout=layout, cu_seqlens=cu_seqlens) is sentinel
            kernel.assert_called_once()


@pytest.mark.parametrize("compressed_length", [63, 64])
def test_sparse_flash_mla_csa_preserves_compressed_length_validation(compressed_length):
    """Moving the shared length check keeps explicit-index CSA validation intact."""
    from fsdp_turbo.ops.npu import sparse_attention

    query = torch.empty(1, 256, 2, 512, dtype=torch.bfloat16)
    original = torch.empty(1, 256, 512, dtype=torch.bfloat16)
    compressed = torch.empty(1, compressed_length, 512, dtype=torch.bfloat16)
    indices = torch.zeros(1, 256, 1, dtype=torch.int32)
    arguments = (query, original, compressed, indices, torch.zeros(2), 4, 128)
    if compressed_length == 64:
        sparse_attention._validate_sparse_flash_mla_inputs(*arguments)
    else:
        with pytest.raises(ValueError, match="ori_len == cmp_len"):
            sparse_attention._validate_sparse_flash_mla_inputs(*arguments)


@pytest.mark.parametrize("validation", [None, "0", "1"])
def test_sparse_attention_mask_content_checks_are_opt_in(monkeypatch, validation):
    if validation is None:
        monkeypatch.delenv("FSDP_TURBO_VALIDATE_SPARSE_FLASH_MLA", raising=False)
    else:
        monkeypatch.setenv("FSDP_TURBO_VALIDATE_SPARSE_FLASH_MLA", validation)
    module, query, key_value, mask, compressed, indices = _attention_inputs("compressed_sparse_attention")
    with (
        mock.patch.object(modeling, "indexed_sparse_attention", return_value=query.transpose(1, 2)) as kernel,
        mock.patch.object(torch, "isneginf", wraps=torch.isneginf) as isneginf,
        mock.patch.object(torch, "equal", wraps=torch.equal) as equal,
    ):
        _deepseek_v4_sparse_attention(
            module,
            query,
            key_value,
            mask,
            0.5,
            compressed_key_value=compressed,
            compressed_topk_indices=indices,
        )
        kernel.assert_called_once()
        assert isneginf.call_count == (1 if validation == "1" else 0)
        assert equal.call_count == (1 if validation == "1" else 0)


def test_sparse_attention_debug_rejects_non_sliding_mask(monkeypatch):
    monkeypatch.setenv("FSDP_TURBO_VALIDATE_SPARSE_FLASH_MLA", "1")
    module, query, key_value, mask, compressed, indices = _attention_inputs("compressed_sparse_attention")
    mask[..., -1, -1] = -torch.inf
    with (
        mock.patch.object(modeling, "indexed_sparse_attention") as kernel,
        pytest.raises(NotImplementedError, match="unpadded, unpacked"),
    ):
        _deepseek_v4_sparse_attention(
            module,
            query,
            key_value,
            mask,
            0.5,
            compressed_key_value=compressed,
            compressed_topk_indices=indices,
        )
    kernel.assert_not_called()

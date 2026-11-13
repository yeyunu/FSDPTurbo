import pytest
import torch

from fsdp_turbo.ops import registry
from fsdp_turbo.ops.registry import get_op
from fsdp_turbo.ops.sparse_attention import indexed_sparse_attention, sparse_flash_attention


def _reference_sparse_flash_attention(query, key, value, topk_indices, softmax_scale, query_rope=None, key_rope=None):
    """Loop-based reference over topk_indices, skipping -1 (unused) slots."""
    batch_size, sequence_length, num_heads, head_dim = query.shape
    output = torch.zeros(batch_size, sequence_length, num_heads, head_dim, dtype=torch.float32)
    for batch in range(batch_size):
        for position in range(sequence_length):
            valid = topk_indices[batch, position] >= 0
            indices = topk_indices[batch, position][valid].long()
            if indices.numel() == 0:
                continue
            selected_key = key[batch, indices, 0, :].float()
            scores = torch.einsum('hd,kd->hk', query[batch, position].float(), selected_key) * softmax_scale
            if query_rope is not None:
                selected_key_rope = key_rope[batch, indices, 0, :].float()
                rope_scores = torch.einsum('hr,kr->hk', query_rope[batch, position].float(), selected_key_rope)
                scores = scores + rope_scores * softmax_scale
            probabilities = torch.softmax(scores, dim=-1)
            selected_value = value[batch, indices, 0, :].float()
            output[batch, position] = torch.einsum('hk,kd->hd', probabilities, selected_value)
    return output.to(query.dtype)


def _reference_indexed_sparse_attention(
    query, key_value, attention_sink, topk_indices, softmax_scale, attention_mask=None
):
    """Loop-based reference: topk gather plus appended sink logits, -1 slots skipped."""
    batch_size, sequence_length, num_heads, head_dim = query.shape
    output = torch.zeros(batch_size, sequence_length, num_heads, head_dim, dtype=torch.float32)
    for batch in range(batch_size):
        for position in range(sequence_length):
            valid = topk_indices[batch, position] >= 0
            indices = topk_indices[batch, position][valid].long()
            selected_key_value = key_value[batch, indices, :].float()
            scores = torch.einsum('hd,kd->hk', query[batch, position].float(), selected_key_value) * softmax_scale
            if attention_mask is not None:
                token_mask = attention_mask[batch, indices].bool()
                scores = scores.masked_fill(~token_mask.view(1, -1), float('-inf'))
            sink_scores = attention_sink.float().view(1, -1).expand(num_heads, -1)
            probabilities = torch.softmax(torch.cat((scores, sink_scores), dim=-1), dim=-1)[
                :, : -attention_sink.numel()
            ]
            output[batch, position] = torch.einsum('hk,kd->hd', probabilities, selected_key_value)
    return output.to(query.dtype)


def test_sparse_flash_attention_has_cpu_fallback():
    assert get_op('sparse_flash_attention', 'cpu').__name__ == 'sparse_flash_attention_torch'


def test_indexed_sparse_attention_has_cpu_fallback():
    assert get_op('indexed_sparse_attention', 'cpu').__name__ == 'indexed_sparse_attention_torch'


def test_sparse_flash_attention_cpu_matches_reference():
    torch.manual_seed(17)
    batch_size, sequence_length, num_heads, head_dim = 2, 3, 2, 4
    kv_length, topk_size = 6, 4
    query = torch.randn(batch_size, sequence_length, num_heads, head_dim)
    key = torch.randn(batch_size, kv_length, 1, head_dim)
    value = torch.randn(batch_size, kv_length, 1, head_dim)
    topk_indices = torch.randint(0, kv_length, (batch_size, sequence_length, topk_size))

    output = get_op('sparse_flash_attention', 'cpu')(query, key, value, topk_indices, 0.2)

    expected = _reference_sparse_flash_attention(query, key, value, topk_indices, 0.2)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_sparse_flash_attention_cpu_masks_negative_indices():
    torch.manual_seed(19)
    query = torch.randn(1, 2, 1, 4)
    key = torch.randn(1, 5, 1, 4)
    value = torch.randn(1, 5, 1, 4)
    topk_indices = torch.tensor([[[2, -1, 4, -1]]])

    output = get_op('sparse_flash_attention', 'cpu')(query, key, value, topk_indices, 0.2)

    expected = _reference_sparse_flash_attention(query, key, value, topk_indices, 0.2)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    clamped_indices = topk_indices.clamp_min(0)
    clamped_output = get_op('sparse_flash_attention', 'cpu')(query, key, value, clamped_indices, 0.2)
    assert not torch.allclose(output, clamped_output)


def test_sparse_flash_attention_cpu_applies_rope():
    torch.manual_seed(23)
    rope_head_dim = 2
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 5, 1, 4)
    value = torch.randn(1, 5, 1, 4)
    query_rope = torch.randn(1, 2, 2, rope_head_dim)
    key_rope = torch.randn(1, 5, 1, rope_head_dim)
    topk_indices = torch.randint(0, 5, (1, 2, 3))

    output = get_op('sparse_flash_attention', 'cpu')(
        query, key, value, topk_indices, 0.2, rope_head_dim, query_rope, key_rope
    )

    expected = _reference_sparse_flash_attention(query, key, value, topk_indices, 0.2, query_rope, key_rope)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_sparse_flash_attention_wrapper_forwards_semantic_arguments(monkeypatch):
    calls = []

    def fake_sparse_flash_attention(*args, **kwargs):
        calls.append((args, kwargs))
        return 'output'

    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cpu'))
    monkeypatch.setitem(registry._OPERATOR_REGISTRY['sparse_flash_attention'], 'cpu', fake_sparse_flash_attention)

    sparse_flash_attention('q', 'k', 'v', 'idx', 0.2)
    result = sparse_flash_attention('q', 'k', 'v', 'idx', 0.2, rope_head_dim=64, query_rope='qr', key_rope='kr')

    assert result == 'output'
    assert calls == [
        (('q', 'k', 'v', 'idx', 0.2), {'rope_head_dim': 0, 'query_rope': None, 'key_rope': None, 'sparse_mode': 3}),
        (('q', 'k', 'v', 'idx', 0.2), {'rope_head_dim': 64, 'query_rope': 'qr', 'key_rope': 'kr', 'sparse_mode': 3}),
    ]


def test_indexed_sparse_attention_cpu_matches_reference():
    torch.manual_seed(29)
    query = torch.randn(2, 3, 2, 4)
    key_value = torch.randn(2, 6, 4)
    attention_sink = torch.randn(2)
    topk_indices = torch.randint(0, 6, (2, 3, 4))

    output = get_op('indexed_sparse_attention', 'cpu')(query, key_value, attention_sink, topk_indices, 0.2)

    expected = _reference_indexed_sparse_attention(query, key_value, attention_sink, topk_indices, 0.2)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_indexed_sparse_attention_cpu_masks_negative_and_invalid_tokens():
    torch.manual_seed(31)
    query = torch.randn(1, 2, 1, 4)
    key_value = torch.randn(1, 5, 4)
    attention_sink = torch.randn(1)
    topk_indices = torch.tensor([[[3, -1, 1, 2]]])
    attention_mask = torch.tensor([[True, False, True, True, True]])

    output = get_op('indexed_sparse_attention', 'cpu')(
        query, key_value, attention_sink, topk_indices, 0.2, attention_mask=attention_mask
    )

    expected = _reference_indexed_sparse_attention(query, key_value, attention_sink, topk_indices, 0.2, attention_mask)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_indexed_sparse_attention_cpu_merges_compressed_kv():
    torch.manual_seed(37)
    query = torch.randn(1, 2, 1, 4)
    key_value = torch.randn(1, 5, 4)
    compressed_key_value = torch.randn(1, 3, 4)
    attention_sink = torch.randn(1)
    topk_indices = torch.tensor([[[1, 3, -1]]])
    compressed_topk_indices = torch.tensor([[[0, -1, 2]]])

    output = get_op('indexed_sparse_attention', 'cpu')(
        query,
        key_value,
        attention_sink,
        topk_indices,
        0.2,
        compressed_key_value=compressed_key_value,
        compressed_topk_indices=compressed_topk_indices,
    )

    merged_key_value = torch.cat((key_value, compressed_key_value), dim=1)
    kv_length = key_value.size(1)
    merged_indices = torch.cat(
        (
            topk_indices,
            torch.where(compressed_topk_indices >= 0, compressed_topk_indices + kv_length, compressed_topk_indices),
        ),
        dim=-1,
    )
    expected = _reference_indexed_sparse_attention(query, merged_key_value, attention_sink, merged_indices, 0.2)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_indexed_sparse_attention_cpu_builds_hca_causal_indices():
    torch.manual_seed(39)
    query = torch.randn(1, 8, 1, 4)
    key_value = torch.randn(1, 8, 4)
    compressed_key_value = torch.randn(1, 2, 4)
    attention_sink = torch.randn(1)
    positions = torch.arange(8)
    original_indices = torch.arange(8).view(1, 1, -1).expand(1, 8, -1)
    original_indices = original_indices.masked_fill(original_indices > positions.view(1, -1, 1), -1)

    output = get_op('indexed_sparse_attention', 'cpu')(
        query,
        key_value,
        attention_sink,
        original_indices,
        0.2,
        compressed_key_value=compressed_key_value,
        compress_ratio=4,
    )

    compressed_indices = torch.arange(2).view(1, 1, -1).expand(1, 8, -1)
    compressed_indices = compressed_indices.masked_fill(
        compressed_indices >= ((positions + 1) // 4).view(1, -1, 1),
        -1,
    )
    merged_indices = torch.cat(
        (original_indices, torch.where(compressed_indices >= 0, compressed_indices + 8, compressed_indices)),
        dim=-1,
    )
    expected = _reference_indexed_sparse_attention(
        query,
        torch.cat((key_value, compressed_key_value), dim=1),
        attention_sink,
        merged_indices,
        0.2,
    )
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_indexed_sparse_attention_cpu_rejects_non_bsnd_layout():
    query = torch.randn(1, 2, 1, 4)
    key_value = torch.randn(1, 3, 4)

    with pytest.raises(NotImplementedError):
        get_op('indexed_sparse_attention', 'cpu')(
            query, key_value, torch.randn(1), torch.zeros(1, 2, 1, dtype=torch.long), 0.2, layout='BNSD'
        )


def test_indexed_sparse_attention_wrapper_forwards_positional_contract(monkeypatch):
    calls = []

    def fake_indexed_sparse_attention(*args, **kwargs):
        calls.append((args, kwargs))
        return 'output'

    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cpu'))
    monkeypatch.setitem(registry._OPERATOR_REGISTRY['indexed_sparse_attention'], 'cpu', fake_indexed_sparse_attention)

    result = indexed_sparse_attention('q', 'kv', 'sink', 'idx', 0.2, use_eager=True)

    assert result == 'output'
    assert calls == [(('q', 'kv', 'sink', 'idx', 0.2, None, False, None, None, 1, 128, 'BSND', None), {})]


def test_indexed_sparse_attention_use_eager_runs_cpu_implementation():
    torch.manual_seed(41)
    query = torch.randn(1, 2, 1, 4)
    key_value = torch.randn(1, 5, 4)
    attention_sink = torch.randn(1)
    topk_indices = torch.randint(0, 5, (1, 2, 3))

    output = indexed_sparse_attention(query, key_value, attention_sink, topk_indices, 0.2, use_eager=True)
    expected = get_op('indexed_sparse_attention', 'cpu')(query, key_value, attention_sink, topk_indices, 0.2)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)

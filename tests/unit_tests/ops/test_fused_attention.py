import torch

from fsdp_turbo.ops import registry
from fsdp_turbo.ops.fused_attention import npu_fusion_attention
from fsdp_turbo.ops.registry import get_op


def _reference_attention(query, key, value, atten_mask, scale):
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    if atten_mask is not None:
        scores = scores.masked_fill(atten_mask, torch.finfo(torch.float32).min)
    return torch.matmul(torch.softmax(scores, dim=-1), value.float())


def test_npu_fusion_attention_dispatches_positionally(monkeypatch):
    calls = []

    def fake_npu_fusion_attention(*args, **kwargs):
        calls.append((args, kwargs))
        return 'output'

    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cpu'))
    monkeypatch.setitem(registry._OPERATOR_REGISTRY['npu_fusion_attention'], 'cpu', fake_npu_fusion_attention)

    result = npu_fusion_attention('q', 'k', 'v', head_num=8, scale=0.125)

    assert result == 'output'
    assert calls == [(('q', 'k', 'v', 8, None, 0.125, 1.0), {})]


def test_npu_fusion_attention_has_cpu_fallback():
    assert get_op('npu_fusion_attention', 'cpu').__name__ == 'npu_fusion_attention_cpu'


def test_npu_fusion_attention_cpu_matches_reference():
    torch.manual_seed(7)
    query = torch.randn(2, 3, 4, 5)
    key = torch.randn(2, 3, 6, 5)
    value = torch.randn(2, 3, 6, 5)

    output = get_op('npu_fusion_attention', 'cpu')(query, key, value, 3, None, 0.25, 1.0)

    expected = _reference_attention(query, key, value, None, 0.25)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_npu_fusion_attention_cpu_masks_true_positions():
    torch.manual_seed(11)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)
    atten_mask = torch.zeros(1, 1, 3, 5, dtype=torch.bool)
    atten_mask[..., 1::2] = True

    output = get_op('npu_fusion_attention', 'cpu')(query, key, value, 2, atten_mask, 0.2, 1.0)

    expected = _reference_attention(query, key, value, atten_mask, 0.2)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    unmasked = get_op('npu_fusion_attention', 'cpu')(query, key, value, 2, None, 0.2, 1.0)
    assert not torch.allclose(output, unmasked)

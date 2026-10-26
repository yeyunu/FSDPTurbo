import torch
import torch.nn.functional as F

from fsdp_turbo.ops import registry
from fsdp_turbo.ops.lightning_indexer import lightning_indexer_score
from fsdp_turbo.ops.registry import get_op


def _reference_indexer_scores(query, pool_keys, weights, scale):
    scores = torch.einsum('bshd,bpd->bshp', query.float(), pool_keys.float())
    return torch.einsum('bsh,bshp->bsp', weights.float(), F.relu(scores * scale))


def test_lightning_indexer_score_dispatches_positionally(monkeypatch):
    calls = []

    def fake_lightning_indexer_score(*args, **kwargs):
        calls.append((args, kwargs))
        return 'scores'

    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cpu'))
    monkeypatch.setitem(registry._OPERATOR_REGISTRY['lightning_indexer_score'], 'cpu', fake_lightning_indexer_score)

    result = lightning_indexer_score('q', 'pool_keys', 'weights')

    assert result == 'scores'
    assert calls == [(('q', 'pool_keys', 'weights', None, None), {})]


def test_lightning_indexer_score_has_cpu_fallback():
    assert get_op('lightning_indexer_score', 'cpu').__name__ == 'lightning_indexer_score_torch'


def test_lightning_indexer_score_cpu_matches_formula():
    torch.manual_seed(3)
    query = torch.randn(2, 3, 4, 6)
    pool_keys = torch.randn(2, 5, 6)
    weights = torch.randn(2, 3, 4)

    output = get_op('lightning_indexer_score', 'cpu')(query, pool_keys, weights, None, 0.3)

    expected = _reference_indexer_scores(query, pool_keys, weights, 0.3)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    assert output.dtype == torch.float32


def test_lightning_indexer_score_cpu_default_scale():
    torch.manual_seed(5)
    query = torch.randn(1, 2, 3, 8)
    pool_keys = torch.randn(1, 4, 8)
    weights = torch.randn(1, 2, 3)

    output = get_op('lightning_indexer_score', 'cpu')(query, pool_keys, weights)
    expected = _reference_indexer_scores(query, pool_keys, weights, query.shape[-1] ** -0.5)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_lightning_indexer_score_cpu_masks_invalid_candidates():
    torch.manual_seed(13)
    query = torch.randn(1, 2, 2, 4)
    pool_keys = torch.randn(1, 5, 4)
    weights = torch.randn(1, 2, 2)
    valid_candidates = torch.tensor([[[True, True, False, True, False], [False, True, True, True, True]]])

    output = get_op('lightning_indexer_score', 'cpu')(query, pool_keys, weights, valid_candidates, 0.25)

    unmasked = _reference_indexer_scores(query, pool_keys, weights, 0.25)
    float32_min = torch.finfo(torch.float32).min
    torch.testing.assert_close(output[valid_candidates], unmasked[valid_candidates], rtol=0, atol=0)
    assert torch.equal(output[~valid_candidates], torch.full(output[~valid_candidates].shape, float32_min))

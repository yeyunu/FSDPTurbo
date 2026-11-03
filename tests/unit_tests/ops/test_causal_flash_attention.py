# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Registry-dispatched causal flash attention: SDPA reference and NPU kernel args."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from fsdp_turbo.ops import registry
from fsdp_turbo.ops.flash_attn import causal_flash_attention


def _reference_attention(q, k, v, mask=None):
    scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


def test_causal_matches_eager_math():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 7, 5, dtype=torch.float64) for _ in range(3))

    out = causal_flash_attention(q, k, v)

    causal = torch.ones(7, 7, dtype=torch.bool).tril()
    torch.testing.assert_close(out, _reference_attention(q, k, v, causal), rtol=1e-4, atol=1e-5)


def test_query_block_keeps_bottom_right_causality():
    # Q is a block of a longer K/V prefix: row i attends K[0, i + (kv_len - q_len)).
    torch.manual_seed(1)
    q = torch.randn(1, 2, 6, 4, dtype=torch.float64)
    k = torch.randn(1, 2, 14, 4, dtype=torch.float64)
    v = torch.randn(1, 2, 14, 4, dtype=torch.float64)

    out = causal_flash_attention(q, k, v)

    mask = torch.ones(6, 14, dtype=torch.bool).tril(diagonal=14 - 6)
    torch.testing.assert_close(out, _reference_attention(q, k, v, mask), rtol=1e-4, atol=1e-5)


def _load_npu_module(monkeypatch):
    """Exec the NPU ops module against a fake ``torch_npu`` (see test_flash_attn_npu_arguments)."""
    calls = []

    def fake_fusion_attention(query, key, value, head_num, layout, **kwargs):
        calls.append((query.shape, head_num, layout, kwargs))
        return (query,)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fusion_attention=fake_fusion_attention))
    monkeypatch.setattr(registry, "register_op", lambda *args: lambda operator: operator)
    path = Path(__file__).resolve().parents[3] / "fsdp_turbo" / "ops" / "npu" / "flash_attn.py"
    spec = importlib.util.spec_from_file_location("test_causal_flash_attention_npu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def test_npu_uses_native_causal_mode(monkeypatch):
    module, calls = _load_npu_module(monkeypatch)
    q = torch.randn(2, 3, 6, 4)

    out = module.causal_flash_attention_npu(q, q, q)

    assert out.shape == q.shape  # BNSD layout is kept
    shape, head_num, layout, kwargs = calls[0]
    assert layout == "BNSD"
    assert head_num == 3
    assert kwargs["sparse_mode"] == 3
    assert kwargs["keep_prob"] == 1.0
    assert kwargs["scale"] == 4**-0.5
    atten_mask = kwargs["atten_mask"]
    assert atten_mask is not None and atten_mask.dtype == torch.bool

    # Non-causal call: no mask and sparse_mode=0; dropout/scale are forwarded.
    module.causal_flash_attention_npu(q, q, q, scaling=0.1, dropout=0.25, is_causal=False)

    _, head_num, layout, kwargs = calls[1]
    assert layout == "BNSD"
    assert head_num == 3
    assert kwargs["sparse_mode"] == 0
    assert kwargs["atten_mask"] is None
    assert kwargs["keep_prob"] == 0.75
    assert kwargs["scale"] == 0.1

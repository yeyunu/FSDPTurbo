# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Check the packed sequence contract passed to NPU flash attention."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fsdp_turbo.ops import registry


@pytest.mark.parametrize(
    ('boundaries', 'expected_layout', 'expected_lengths'),
    [([0, 7], 'BSND', None), ([0, 3, 7], 'TND', [0, 3, 7])],
)
def test_packed_attention_preserves_cumulative_boundaries(monkeypatch, boundaries, expected_layout, expected_lengths):
    calls = []

    def fake_fusion_attention(query, key, value, head_num, layout, **kwargs):
        calls.append((query.shape, head_num, layout, kwargs))
        return (query,)

    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(npu_fusion_attention=fake_fusion_attention))
    monkeypatch.setattr(registry, 'register_op', lambda *args: lambda operator: operator)
    path = Path(__file__).resolve().parents[3] / 'fsdp_turbo/ops/npu/flash_attn.py'
    spec = importlib.util.spec_from_file_location('test_flash_attn_npu_impl', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    query = torch.randn(1, 7, 2, 4)

    output, _ = module.flash_attention_npu(
        None,
        query,
        query,
        query,
        is_causal=False,
        input_layout='1TND',
        cu_seq_lens_q=torch.tensor(boundaries),
        cu_seq_lens_k=torch.tensor(boundaries),
    )

    assert output.shape == query.shape
    assert len(calls) == 1
    shape, head_num, layout, kwargs = calls[0]
    assert shape == (query.shape if expected_layout == 'BSND' else query.shape[1:])
    assert head_num == 2
    assert layout == expected_layout
    assert kwargs['actual_seq_qlen'] == expected_lengths
    assert kwargs['actual_seq_kvlen'] == expected_lengths

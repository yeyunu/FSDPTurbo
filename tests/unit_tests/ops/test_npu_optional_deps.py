"""A missing optional linear-attention dependency must not unregister the core NPU ops."""

import importlib
import sys

import pytest


def test_core_npu_ops_register_without_optional_linear_attention_deps(monkeypatch):
    pytest.importorskip('torch_npu')
    for name in list(sys.modules):
        if name == 'fsdp_turbo.ops.npu' or name.startswith('fsdp_turbo.ops.npu.'):
            monkeypatch.delitem(sys.modules, name)
    # A None entry makes ``import fla`` raise ImportError, as on an environment without flash-linear-attention.
    monkeypatch.setitem(sys.modules, 'fla', None)

    npu = importlib.import_module('fsdp_turbo.ops.npu')

    from fsdp_turbo.ops.registry import _OPERATOR_REGISTRY

    for op in ('permute', 'unpermute', 'grouped_matmul'):
        assert 'npu' in _OPERATOR_REGISTRY[op], f'{op} lost its NPU implementation'
    # GDN selects the AscendC or FLA backend lazily, so importing the package
    # does not require the optional FLA backend to be installed.
    assert callable(npu.chunk_gdn_npu)
    assert _OPERATOR_REGISTRY['chunk_gated_delta_rule']['npu'] is npu.chunk_gdn_npu

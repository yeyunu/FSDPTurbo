# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Packed-TND entry points: installation, runtime lookup and batch splitting."""

import types

import pytest

from fsdp_turbo.distributed.context_parallel.kv_allgather.tnd import (
    TNDRuntime,
    install_tnd_context_parallel,
    is_tnd_batch,
    split_tnd_batch,
)


class _Attention:
    compress_ratio = 2
    window_size = 128


class _Model:
    def __init__(self):
        self.wired = []
        self.tnd_context_parallel = None

    def set_tnd_context_parallel(self, runtime):
        self.runtime = runtime
        self.wired = [_Attention(), _Attention()]
        return self.wired


def test_install_attaches_runtime_and_returns_the_same_model():
    model = _Model()

    returned = install_tnd_context_parallel(model, None)

    assert returned is model
    assert isinstance(model.tnd_context_parallel, TNDRuntime)
    assert model.tnd_context_parallel.lcm == 2
    assert model.tnd_context_parallel.window == 128
    assert model.wired


def test_install_requires_the_model_hook():
    with pytest.raises(RuntimeError, match="set_tnd_context_parallel"):
        install_tnd_context_parallel(object(), None)


def test_install_rejects_function_patches():
    plan = types.SimpleNamespace(kv_allgather_function_patches=[{"type": "full_attention"}])

    with pytest.raises(ValueError, match="native"):
        install_tnd_context_parallel(_Model(), None, plan)


def test_split_ignores_non_tnd_batches():
    model = _Model()
    install_tnd_context_parallel(model, None)

    assert not is_tnd_batch({"sfa_layout": "BSND"})
    assert not is_tnd_batch({"sfa_layout": ["BSND"]})
    assert split_tnd_batch(model, {"sfa_layout": "BSND"}) is None


def test_split_finds_the_runtime_below_a_wrapper():
    split = types.SimpleNamespace(split_batch=lambda batch: {"split": True})
    inner = types.SimpleNamespace(tnd_context_parallel=split)
    wrapper = types.SimpleNamespace(model=inner)

    assert split_tnd_batch(wrapper, {"sfa_layout": "TND"}) == {"split": True}


def test_split_rejects_a_tnd_batch_without_a_runtime():
    with pytest.raises(RuntimeError, match="no installed runtime"):
        split_tnd_batch(types.SimpleNamespace(), {"sfa_layout": "TND"})

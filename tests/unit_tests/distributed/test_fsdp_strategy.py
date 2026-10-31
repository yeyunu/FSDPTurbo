# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps
import inspect
import logging

import pytest
import torch.distributed.fsdp as torch_fsdp
from torch import nn

from fsdp_turbo.distributed.fine_grained_fully_shard.strategy import NativeFSDPStrategy


@pytest.mark.parametrize("hook_module", [None, nn.Linear(2, 2)])
def test_native_strategy_passes_hook_to_supporting_implementation(monkeypatch, hook_module):
    calls = []

    def patched_fully_shard(module, *, hook_module=None, mesh=None):
        calls.append((module, hook_module, mesh))
        return module

    monkeypatch.setattr(torch_fsdp, "fully_shard", patched_fully_shard)
    strategy = NativeFSDPStrategy()
    module = nn.Linear(2, 2)
    mesh = object()

    assert strategy.supports_hook_module
    assert strategy.get_unified_fully_shard_fn()(module, hook_module=hook_module, mesh=mesh) is module
    assert calls == [(module, hook_module, mesh)]


def test_native_strategy_omits_hook_for_original_interface(monkeypatch):
    calls = []

    def native_fully_shard(module, *, mesh=None):
        calls.append((module, mesh))
        return module

    monkeypatch.setattr(torch_fsdp, "fully_shard", native_fully_shard)
    strategy = NativeFSDPStrategy()
    module = nn.Linear(2, 2)
    hook_module = nn.Linear(2, 2)
    mesh = object()

    assert not strategy.supports_hook_module
    assert strategy.get_unified_fully_shard_fn()(module, hook_module=hook_module, mesh=mesh) is module
    assert calls == [(module, mesh)]


def test_native_strategy_detects_hook_on_outer_decorator(monkeypatch):
    calls = []

    def original(module, *, mesh=None):
        return module

    @wraps(original)
    def patched_fully_shard(module, *, hook_module=None, mesh=None):
        calls.append((module, hook_module, mesh))
        return module

    monkeypatch.setattr(torch_fsdp, "fully_shard", patched_fully_shard)
    module, hook_module, mesh = nn.Linear(2, 2), nn.Linear(2, 2), object()
    strategy = NativeFSDPStrategy()
    assert strategy.supports_hook_module
    assert strategy.get_unified_fully_shard_fn()(module, hook_module=hook_module, mesh=mesh) is module
    assert calls == [(module, hook_module, mesh)]


def test_native_strategy_detects_hook_through_forwarding_decorator(monkeypatch):
    calls = []

    def supporting_fully_shard(module, *, hook_module=None, mesh=None):
        calls.append((module, hook_module, mesh))
        return module

    @wraps(supporting_fully_shard)
    def forwarding_fully_shard(module, **kwargs):
        return supporting_fully_shard(module, **kwargs)

    monkeypatch.setattr(torch_fsdp, "fully_shard", forwarding_fully_shard)
    module, hook_module, mesh = nn.Linear(2, 2), nn.Linear(2, 2), object()
    strategy = NativeFSDPStrategy()
    assert strategy.supports_hook_module
    assert strategy.get_unified_fully_shard_fn()(module, hook_module=hook_module, mesh=mesh) is module
    assert calls == [(module, hook_module, mesh)]


def test_native_strategy_does_not_infer_support_from_kwargs(monkeypatch):
    calls = []

    def original(module, *, mesh=None):
        calls.append((module, mesh))
        return module

    @wraps(original)
    def forwarding_fully_shard(module, **kwargs):
        return original(module, **kwargs)

    monkeypatch.setattr(torch_fsdp, "fully_shard", forwarding_fully_shard)
    module, hook_module, mesh = nn.Linear(2, 2), nn.Linear(2, 2), object()
    strategy = NativeFSDPStrategy()
    assert not strategy.supports_hook_module
    assert strategy.get_unified_fully_shard_fn()(module, hook_module=hook_module, mesh=mesh) is module
    assert calls == [(module, mesh)]


def test_native_strategy_does_not_trust_unreachable_inner_hook(monkeypatch):
    calls = []

    def supporting_fully_shard(module, *, hook_module=None, mesh=None):
        return module

    @wraps(supporting_fully_shard)
    def restrictive_fully_shard(module, *, mesh=None):
        calls.append((module, mesh))
        return module

    monkeypatch.setattr(torch_fsdp, "fully_shard", restrictive_fully_shard)
    module, hook_module, mesh = nn.Linear(2, 2), nn.Linear(2, 2), object()
    strategy = NativeFSDPStrategy()
    assert not strategy.supports_hook_module
    assert strategy.get_unified_fully_shard_fn()(module, hook_module=hook_module, mesh=mesh) is module
    assert calls == [(module, mesh)]


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_native_strategy_warns_and_omits_hook_when_signature_unavailable(monkeypatch, caplog, error_type):
    from fsdp_turbo.utils import log as log_utils

    calls = []

    class UninspectableFullyShard:
        @property
        def __signature__(self):
            raise error_type("signature unavailable")

        def __call__(self, module, *, mesh=None):
            calls.append((module, mesh))
            return module

    monkeypatch.setattr(log_utils, "_warned_messages", set())
    monkeypatch.setattr(torch_fsdp, "fully_shard", UninspectableFullyShard())
    module, hook_module, mesh = nn.Linear(2, 2), nn.Linear(2, 2), object()
    strategy = NativeFSDPStrategy()
    with caplog.at_level(logging.WARNING):
        assert not strategy.supports_hook_module
        assert strategy.get_unified_fully_shard_fn()(module, hook_module=hook_module, mesh=mesh) is module
    assert calls == [(module, mesh)]
    assert len(caplog.records) == 1
    assert "Cannot inspect the active fully_shard signature" in caplog.text


def test_native_strategy_detects_explicit_signature_on_kwargs_wrapper(monkeypatch):
    calls = []

    def supporting_fully_shard(module, *, hook_module=None, mesh=None):
        calls.append((module, hook_module, mesh))
        return module

    def forwarding_fully_shard(module, **kwargs):
        return supporting_fully_shard(module, **kwargs)

    forwarding_fully_shard.__signature__ = inspect.signature(supporting_fully_shard)
    monkeypatch.setattr(torch_fsdp, "fully_shard", forwarding_fully_shard)
    module, hook_module, mesh = nn.Linear(2, 2), nn.Linear(2, 2), object()
    strategy = NativeFSDPStrategy()
    assert strategy.supports_hook_module
    assert strategy.get_unified_fully_shard_fn()(module, hook_module=hook_module, mesh=mesh) is module
    assert calls == [(module, hook_module, mesh)]


def test_native_strategy_does_not_pass_positional_only_hook_as_keyword(monkeypatch):
    def positional_only_hook(module, hook_module=None, /, *, mesh=None):
        return module

    monkeypatch.setattr(torch_fsdp, "fully_shard", positional_only_hook)
    module = nn.Linear(2, 2)
    strategy = NativeFSDPStrategy()
    assert not strategy.supports_hook_module
    assert strategy.get_unified_fully_shard_fn()(module, hook_module=nn.Linear(2, 2)) is module

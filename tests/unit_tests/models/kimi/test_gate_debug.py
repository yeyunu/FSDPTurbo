# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for the MoonEP gate-debug handle attachment (``gate_debug.py``).

CPU only. The attachment must find every Kimi MoE gate through the module
tree, skip non-Kimi gates without raising, and hand each gate the EP group,
the layer index, and the plan's router-debug config -- the exact three
attributes ``router_debug.py`` reads.
"""

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.distributed.expert_parallel.moonep.gate_debug import (  # noqa: E402
    attach_gate_debug,
)
from fsdp_turbo.fsdp_turbo_config import KimiRouterDebugConfig  # noqa: E402


class _FakeGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(4, 8))


class _FakeExperts(torch.nn.Module):
    """Stands in for the experts container; ep_modules entries are these."""

    def __init__(self):
        super().__init__()
        self.inner = torch.nn.Linear(8, 8)


class _FakeMoE(torch.nn.Module):
    def __init__(self, with_gate=True):
        super().__init__()
        self.experts = _FakeExperts()
        if with_gate:
            self.gate = _FakeGate()


class _FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _FakeMoE()


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.ModuleList(_FakeLayer() for _ in range(3))


def _plan(router_debug):
    return type("Plan", (), {"router_debug": router_debug})()


def test_every_gate_gets_the_group_layer_and_debug_handle():
    model = _FakeModel()
    ep_modules = [layer.mlp.experts for layer in model.model.children()]
    group = object()
    debug = KimiRouterDebugConfig(record_load_balance=True)

    # The dispatcher passes ``ep_mesh.get_group()``; the group itself is the
    # third argument, not the mesh.
    attach_gate_debug(model, ep_modules, group, _plan(debug))

    for index in range(len(ep_modules)):
        gate = model.get_submodule(f"model.{index}.mlp.gate")
        assert gate._expert_parallel_group is group
        assert gate._router_debug_layer == index
        assert gate._router_debug is debug


def test_a_model_without_gates_is_skipped_quietly():
    model = _FakeModel()
    # Strip the gates to model a non-Kimi MoE.
    for layer in model.model.children():
        del layer.mlp.gate
    ep_modules = [layer.mlp.experts for layer in model.model.children()]

    attach_gate_debug(model, ep_modules, object(), _plan(None))

    for module in ep_modules:
        # Nothing was attached; router_debug stays absent and default-off.
        assert not hasattr(module, "_router_debug")


def test_router_debug_none_is_passed_through():
    """No router-debug config in the plan still attaches the other handles."""
    model = _FakeModel()
    ep_modules = [layer.mlp.experts for layer in model.model.children()]
    group = object()

    attach_gate_debug(model, ep_modules, group, _plan(None))

    gate = model.get_submodule("model.0.mlp.gate")
    assert gate._expert_parallel_group is group
    assert gate._router_debug_layer == 0
    assert gate._router_debug is None

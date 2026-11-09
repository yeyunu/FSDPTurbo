# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Regression tests pinning the ``1602f9e`` assembly boundary.

``expert_parallelize_modules`` must keep two dispatcher families apart:
``moonep`` gets its own assembly (runtime + gate-debug handles) and the
other dispatchers must neither gain the router-debug attributes nor import
``moonep`` at all. Someone moving ``attach_gate_debug`` back into the common
loop -- or making the non-moonep path import MoonEP eagerly -- should fail
here rather than on a machine.
"""

import os
import pathlib
import sys
import tempfile
import time

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.distributed.expert_parallel.expert_parallel import (  # noqa: E402
    expert_parallelize_modules,
)


class _FakeGate(torch.nn.Module):
    def __init__(self, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = torch.nn.Parameter(torch.empty(num_experts, 8))


class _FakeExperts(torch.nn.ModuleList):
    """The ``ep_modules`` entries: ModuleList carries the expert count."""

    def __init__(self, num_experts=8):
        super().__init__(torch.nn.Linear(8, 8) for _ in range(num_experts))
        self.num_experts = num_experts


class _FakeMoE(torch.nn.Module):
    """Qwen3/Kimi layout: ``gate`` is a sibling of ``experts`` under ``mlp``."""

    def __init__(self, num_experts=8, with_gate=True):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.experts = _FakeExperts(num_experts)
        if with_gate:
            self.mlp.gate = _FakeGate(num_experts)


class _FakeModel(torch.nn.Module):
    """Module paths mirror Qwen3's ``model.{i}.mlp.experts`` layout."""

    def __init__(self, layers=3, with_gate=True):
        super().__init__()
        self.model = torch.nn.ModuleList(_FakeMoE(with_gate=with_gate) for _ in range(layers))


def _ep_mesh(world_size=1):
    """A 1-D CPU mesh over one rank, as the ST harness builds on a card."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    rendezvous = pathlib.Path(tempfile.gettempdir()) / f"moonep_asm_rdzv_{os.getpid()}_{time.monotonic_ns()}"
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=world_size,
    )
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep",))
    rendezvous.unlink(missing_ok=True)
    return mesh


def _plan(dispatcher):
    from fsdp_turbo.fsdp_turbo_config import EPPlanConfig

    return EPPlanConfig(
        apply_modules=["model.{*}.mlp.experts"],
        dispatcher=dispatcher,
        use_fused_ops=False,
    )


def teardown_module():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def test_all2all_does_not_attach_router_debug_or_import_moonep(monkeypatch):
    """The non-moonep path must stay clean of MoonEP.

    After ``expert_parallelize_modules`` with ``dispatcher="all2all"``:
    no gate carries the router-debug attributes, and the moonep package
    (runtime in particular) was never imported into the process. The
    DTensor redistribution is stubbed out: it needs real sharded weights,
    and the boundary under test is the dispatch itself.
    """

    import fsdp_turbo.distributed.expert_parallel.expert_parallel as ep_module

    monkeypatch.setattr(ep_module, "distribute_experts_module", lambda module, mesh: None, raising=True)

    sys.modules.pop("fsdp_turbo.distributed.expert_parallel.moonep", None)
    sys.modules.pop("fsdp_turbo.distributed.expert_parallel.moonep.runtime", None)

    model = _FakeModel()
    mesh = _ep_mesh()
    plan = _plan("all2all")

    expert_parallelize_modules(model, mesh, plan)

    # 1) No router-debug handles anywhere.
    for name, module in model.named_modules():
        for attribute in ("_router_debug", "_expert_parallel_group", "_router_debug_layer"):
            assert not hasattr(module, attribute), (name, attribute)

    # 2) The moonep package was not imported as a side effect.
    assert "fsdp_turbo.distributed.expert_parallel.moonep" not in sys.modules
    assert "fsdp_turbo.distributed.expert_parallel.moonep.runtime" not in sys.modules

    # 3) The all2all forward was installed on the experts containers:
    #    a bound method over partial(experts_forward, dispatch_fn=all2all...).
    from fsdp_turbo.distributed.expert_parallel.dispatcher import all2all_experts_forward_fn

    for layer in model.model:
        installed = layer.mlp.experts.forward
        # Bound-method __func__ is the partial; its dispatch keyword must be
        # the all2all entry point, proving the common (non-moonep) path ran.
        assert installed.__func__.keywords["dispatch_fn"] is all2all_experts_forward_fn


def test_moonep_routes_through_its_own_assembly(monkeypatch):
    """``dispatcher == "moonep"`` routes through MoonEP's entry point.

    ``parallelize_moonep_model`` needs the accelerator backend, so it is
    stubbed here: what is pinned is the handoff itself -- the model's own
    modules, the matched ep_modules, the ep mesh and plan reach MoonEP, and
    nothing else in ``expert_parallelize_modules`` touches the model.
    """

    captured = {}

    import fsdp_turbo.distributed.expert_parallel.moonep as moonep_pkg

    def fake_parallelize_moonep_model(modules, ep_modules, ep_mesh, plan, edp_mesh=None):
        captured["modules"] = modules
        captured["ep_modules"] = ep_modules
        captured["ep_mesh"] = ep_mesh
        captured["plan"] = plan
        captured["edp_mesh"] = edp_mesh
        return modules

    monkeypatch.setattr(moonep_pkg, "parallelize_moonep_model", fake_parallelize_moonep_model, raising=False)

    model = _FakeModel()
    mesh = _ep_mesh()
    plan = _plan("moonep")

    expert_parallelize_modules(model, mesh, plan)

    assert captured["modules"] is model
    assert captured["ep_mesh"] is mesh
    assert captured["plan"] is plan
    assert captured["edp_mesh"] is None
    # Exactly the three matched experts containers, in model order.
    assert captured["ep_modules"] == [layer.mlp.experts for layer in model.model]


def test_moonep_assembly_attaches_the_gate_debug_handles(monkeypatch):
    """Inside MoonEP's assembly, gates receive the debug handles (T2 contract).

    Replays ``attach_gate_debug`` the way the real ``parallelize_moonep_model``
    does, then asserts the three attributes the Kimi gate reads: the EP group,
    the layer index, and the plan's router-debug config.
    """

    import fsdp_turbo.distributed.expert_parallel.moonep as moonep_pkg
    from fsdp_turbo.distributed.expert_parallel.moonep.gate_debug import attach_gate_debug

    def fake_parallelize_moonep_model(modules, ep_modules, ep_mesh, plan, edp_mesh=None):
        attach_gate_debug(modules, ep_modules, ep_mesh.get_group(), plan)
        return modules

    monkeypatch.setattr(moonep_pkg, "parallelize_moonep_model", fake_parallelize_moonep_model, raising=False)

    model = _FakeModel()
    mesh = _ep_mesh()
    plan = _plan("moonep")

    expert_parallelize_modules(model, mesh, plan)

    group = mesh.get_group()
    for index, layer in enumerate(model.model):
        gate = layer.mlp.gate
        assert gate._expert_parallel_group is group
        assert gate._router_debug_layer == index
        assert gate._router_debug is None  # EPPlanConfig default: diagnostics off

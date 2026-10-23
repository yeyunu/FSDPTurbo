import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import get_efsdp_modules
from fsdp_turbo.distributed.expert_parallel.expert_parallel import get_experts_forward_fn
from fsdp_turbo.fsdp_turbo_config import DistributedConfig, EPPlanConfig, FSDPTurboConfig


@pytest.mark.parametrize("efsdp_options", [{}, {"apply_efsdp_modules": []}, {"apply_efsdp_modules": None}])
def test_default_efsdp_modules_are_resolved_before_model_initialization(efsdp_options):
    """Kimi-style omitted, empty and legacy-null plans select the expert modules."""
    plan = EPPlanConfig(
        apply_modules=["language_model.model.layers.{*}.block_sparse_moe.experts"],
        dispatcher="custom_ep_forward_eager",
        **efsdp_options,
    )
    config = FSDPTurboConfig(distributed=DistributedConfig(ep_plan=plan))
    experts = torch.nn.Linear(2, 2)
    model = SimpleNamespace(named_modules=lambda: [("language_model.model.layers.0.block_sparse_moe.experts", experts)])

    assert config.distributed.ep_plan.apply_efsdp_modules == plan.apply_modules
    assert config.distributed.ep_plan.apply_efsdp_modules is not plan.apply_modules
    assert get_efsdp_modules(model, config.distributed.ep_plan) == [experts]


def test_explicit_efsdp_modules_are_preserved():
    """Explicit E-FSDP selections may differ from the EP module selection."""
    selected_modules = ["model.layers.0.mlp.experts"]
    plan = EPPlanConfig(
        apply_modules=["model.layers.{*}.mlp.experts"],
        apply_efsdp_modules=selected_modules.copy(),
    )

    config = FSDPTurboConfig(distributed=DistributedConfig(ep_plan=plan))

    assert config.distributed.ep_plan.apply_efsdp_modules == selected_modules


def test_automatic_efsdp_selection_only_includes_experts():
    """Retain the existing expert suffix filter when deriving the default."""
    plan = EPPlanConfig(apply_modules=["model.layers.{*}.mlp.experts", "model.layers.{*}.router"])

    config = FSDPTurboConfig(distributed=DistributedConfig(ep_plan=plan))

    assert config.distributed.ep_plan.apply_efsdp_modules == ["model.layers.{*}.mlp.experts"]


@pytest.mark.parametrize("dispatcher", ["eager", "fused", "alltoall", "unknown", None, 42])
def test_invalid_dispatcher_is_rejected_during_config_initialization(dispatcher):
    with pytest.raises(ValueError, match="Unsupported distributed.ep_plan.dispatcher"):
        FSDPTurboConfig(distributed=DistributedConfig(ep_plan=EPPlanConfig(dispatcher=dispatcher)))


@pytest.mark.parametrize(
    "dispatcher", ["all2all", "allgather", "chunkmoe", "mc2", "domino", "custom_ep_forward_eager", Mock()]
)
def test_supported_dispatcher_is_preserved(dispatcher):
    config = FSDPTurboConfig(distributed=DistributedConfig(ep_plan=EPPlanConfig(dispatcher=dispatcher)))

    assert config.distributed.ep_plan.dispatcher is dispatcher


@pytest.mark.parametrize("distributed", [DistributedConfig(), DistributedConfig(ep_plan=EPPlanConfig())])
def test_default_dispatcher_selects_a_valid_implementation(distributed):
    config = FSDPTurboConfig(distributed=distributed)
    plan = config.distributed.ep_plan

    assert plan.dispatcher == "all2all"
    assert callable(get_experts_forward_fn(Mock(), plan, dispatcher=plan.dispatcher))


def test_system_test_ep_plans_select_supported_dispatchers():
    """Check actual ST plan literals without importing model or accelerator code."""
    root = Path(__file__).resolve().parents[4]
    for test_path in (root / "tests/system_tests/model").rglob("test_*.py"):
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "EPPlanConfig":
                continue
            options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
            config = FSDPTurboConfig(distributed=DistributedConfig(ep_plan=EPPlanConfig(**options)))
            plan = config.distributed.ep_plan
            assert callable(get_experts_forward_fn(Mock(), plan, dispatcher=plan.dispatcher)), str(test_path)


def test_yaml_ep_plans_select_supported_dispatchers():
    """Guard ST and example YAML plans against stale dispatcher names."""
    root = Path(__file__).resolve().parents[4]
    for config_root in (root / "tests/system_tests/model", root / "examples"):
        for config_path in config_root.rglob("*.yaml"):
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            options = raw.get("distributed", {}).get("ep_plan")
            if options is None:
                continue
            config = FSDPTurboConfig(distributed=DistributedConfig(ep_plan=EPPlanConfig(**options)))
            plan = config.distributed.ep_plan
            assert callable(get_experts_forward_fn(Mock(), plan, dispatcher=plan.dispatcher)), str(config_path)

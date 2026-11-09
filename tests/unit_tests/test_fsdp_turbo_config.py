# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MoonEP must reject the configurations it cannot honour, at validation time.

Every case here was checked by building the configuration and seeing what
validation did, not by reading it. Two were accepted when they should not have
been, and both would have produced wrong training rather than an error: expert
FSDP wrapping the weights MoonEP owns, and a MoE quantization converter
replacing the expert GEMM path MoonEP provides itself.

The registry is consulted for the quantization cases rather than a fixed list, so
a converter added later is covered without this file being touched.
"""

import textwrap

import pytest

torch = pytest.importorskip("torch")

from fsdp_turbo.fsdp_turbo_config import (  # noqa: E402
    DistributedConfig,
    EPPlanConfig,
    FSDPTurboConfig,
    MoonEPConfig,
    QuantizationConfig,
    QuantizeConfig,
    load_config_from_yaml,
)


def build(ep_kwargs=None, quantization_plan=None, dispatcher="moonep", memory=None, **distributed_kwargs):
    ep = {"apply_modules": ["model.layers.{*}.mlp.experts"], "dispatcher": dispatcher}
    ep.update(ep_kwargs or {})
    values = {
        "expert_parallel_size": 2,
        "expert_fully_shard_parallel_size": 1,
        "ep_plan": EPPlanConfig(**ep),
    }
    values.update(distributed_kwargs)
    kwargs = {"distributed": DistributedConfig(**values)}
    if quantization_plan is not None:
        kwargs["quantization"] = QuantizationConfig(quantization_plan=quantization_plan)
    if memory is not None:
        kwargs["memory"] = memory
    return FSDPTurboConfig(**kwargs)


def moe_converter_names():
    """Every registered converter that replaces the expert path.

    Called from inside a test rather than from a ``parametrize`` decorator: the
    decorator would run at collection time, and importing the quantization
    package needs an accelerator build that a CPU-only machine does not have.
    """
    import fsdp_turbo.quantization  # noqa: F401
    import fsdp_turbo.quantization.qat.w8a16.w8a16_converter  # noqa: F401
    from fsdp_turbo.quantization.converter.model_converter import (
        _registry_model_converter_cls,
    )

    return sorted(name for name in _registry_model_converter_cls if name.startswith("quantize.moe."))


def test_the_baseline_configuration_is_accepted():
    """So the rejections below mean something."""
    config = build()
    assert config.distributed.ep_plan.dispatcher == "moonep"
    assert config.distributed.ep_plan.apply_efsdp_modules == []


def test_expert_fsdp_size_is_rejected():
    with pytest.raises(ValueError, match="expert FSDP"):
        build(expert_fully_shard_parallel_size=2)


def test_explicit_expert_fsdp_modules_are_rejected():
    """Was accepted: the list is only defaulted when it was not given."""
    with pytest.raises(ValueError, match="expert FSDP"):
        build(ep_kwargs={"apply_efsdp_modules": ["model.layers.{*}.mlp.experts"]})


def test_an_explicitly_empty_expert_fsdp_list_is_fine():
    assert build(ep_kwargs={"apply_efsdp_modules": []}) is not None


def test_expert_parallel_must_be_enabled():
    with pytest.raises(ValueError, match="expert_parallel_size"):
        build(expert_parallel_size=1)


def test_every_moe_quantization_converter_is_rejected():
    """Was accepted for quantize.moe.w8a16: the guard named only the mx one.

    The registry is consulted rather than a fixed list, so a converter added
    later is covered without this test being touched.
    """
    converters = moe_converter_names()
    assert converters, "no quantize.moe.* converter is registered; the check below is vacuous"
    for converter in converters:
        with pytest.raises(ValueError, match="MoE"):
            build(quantization_plan=QuantizeConfig(quant_recipe="mxfp8", converters=[converter]))


def test_quantized_grouped_matmul_is_rejected():
    plan = QuantizeConfig(quant_recipe="mxfp8")
    plan.quant_gmm = True
    with pytest.raises(ValueError, match="MoE"):
        build(quantization_plan=plan)


def test_non_expert_quantization_is_allowed():
    """MoonEP only owns the expert path; quantizing dense linears is unaffected."""
    assert build(quantization_plan=QuantizeConfig(quant_recipe="mxfp8", converters=["quantize.linear.mx"])) is not None


def test_a_recipe_without_converters_is_allowed():
    """The converter container is built from `converters` alone, so a bare recipe
    quantizes nothing.
    """
    assert build(quantization_plan=QuantizeConfig(quant_recipe="mxfp8")) is not None


def test_other_dispatchers_keep_moe_quantization():
    """The guard must be about MoonEP, not about quantization in general."""
    for converter in moe_converter_names():
        assert (
            build(
                dispatcher="fused",
                quantization_plan=QuantizeConfig(quant_recipe="mxfp8", converters=[converter]),
            )
            is not None
        )


def test_ulysses_is_not_rejected_alongside_moonep():
    """Records the resolution of a suspected conflict.

    ``parallel_state`` builds the EP mesh from ``('edp', 'efsdp', 'ep')`` with
    only the expert sizes and puts ulysses in a separate mesh, so an EP group's
    ranks do not depend on it. Nothing to reject; this keeps a guard from
    creeping in that claims otherwise.
    """
    from fsdp_turbo.fsdp_turbo_config import CPPlanConfig

    config = build(
        ulysses_parallel_size=2,
        cp_plan=CPPlanConfig(
            ulysses_function_patches=[
                {"type": "attention", "target_functions": ["torch.nn.functional.scaled_dot_product_attention"]}
            ]
        ),
    )
    assert config.distributed.ulysses_parallel_size == 2
    assert config.distributed.ep_plan.dispatcher == "moonep"


def test_moonep_defaults_are_materialized():
    config = build()
    moon = config.distributed.ep_plan.moonep_config
    assert moon == MoonEPConfig()
    assert config.distributed.ep_plan.apply_efsdp_modules == []


def test_default_divide_factor_is_world_when_edp_is_two(monkeypatch):
    """Candidate C-3 does not rewrite this formula.

    ``WORLD_SIZE=4``, EP=2, efsdp=1 → edp=2 and the factor is still 4
    (``ep * efsdp * edp``), not 2. Only changing the divisor would leave the
    two replicas unsynced. This pins the current value so a silent rewrite
    is visible; it is not an authorization to change it.
    """
    monkeypatch.setenv("WORLD_SIZE", "4")
    config = build(expert_parallel_size=2, expert_fully_shard_parallel_size=1)
    assert config.distributed.ep_plan.gradient_divide_factor == 4


@pytest.mark.parametrize(
    "moon_config, message",
    [
        (MoonEPConfig(num_sms=0), "num_sms"),
        (MoonEPConfig(token_padding=0), "token_padding"),
        (MoonEPConfig(tokens_per_rank=0), "tokens_per_rank"),
        (MoonEPConfig(top_k=0), "top_k"),
    ],
)
def test_moonep_rejects_invalid_tuning(moon_config, message):
    with pytest.raises(ValueError, match=message):
        build(ep_kwargs={"moonep_config": moon_config})


def test_yaml_loads_nested_moonep_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            """
            distributed:
              expert_parallel_size: 2
              expert_fully_shard_parallel_size: 1
              ep_plan:
                apply_modules: ["model.layers.{*}.mlp.experts"]
                dispatcher: moonep
                moonep_config:
                  num_sms: 24
                  token_padding: 64
                  tokens_per_rank: 512
                  top_k: 4
            """
        ),
        encoding="utf-8",
    )
    config = load_config_from_yaml(path)
    moon = config.distributed.ep_plan.moonep_config
    assert isinstance(moon, MoonEPConfig)
    assert (moon.num_sms, moon.token_padding) == (24, 64)
    assert (moon.tokens_per_rank, moon.top_k) == (512, 4)

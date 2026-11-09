# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for the MoonEP settings normalization and per-knob reporting.

A config that predates a knob must degrade to the default rather than raise
``AttributeError`` inside an expert forward. The inert-knob cases are generated
from ``_INERT_BY_DEVICE`` rather than listed, so they follow the code. CPU only.
"""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Optional

import pytest

from fsdp_turbo.distributed.expert_parallel.moonep.settings import MoonEPSettings


@dataclass
class _LegacyConfig:
    """The ``MoonEPConfig`` shape before the memory and Ascend knobs existed."""

    num_sms: int = 32
    token_padding: int = 128
    comm_stream_priority: int = -1
    enable_pdl: bool = True
    async_finish: bool = True
    tokens_per_rank: Optional[int] = None
    top_k: Optional[int] = None


def test_legacy_config_falls_back_to_defaults(caplog):
    settings = MoonEPSettings.from_config(_LegacyConfig(num_sms=48, tokens_per_rank=4096, top_k=8))

    # Known fields are taken from the config ...
    assert settings.num_sms == 48
    assert settings.tokens_per_rank == 4096
    assert settings.top_k == 8
    # ... and the missing ones degrade to the documented defaults.
    assert settings.recompute_dispatch is False
    assert settings.validate_plan_invariants is False
    assert settings.shmem_ip_port is None
    # The fallback must be visible, not silent.
    assert "recompute_dispatch" in caplog.text
    assert "validate_plan_invariants" in caplog.text


def test_current_config_is_passed_through():
    @dataclass
    class _CurrentConfig(_LegacyConfig):
        recompute_dispatch: bool = True
        validate_plan_invariants: bool = True
        shmem_ip_port: Optional[str] = "tcp://10.0.0.1:29501"
        shmem_stream_barrier: bool = False

    settings = MoonEPSettings.from_config(_CurrentConfig())
    assert settings.recompute_dispatch is True
    assert settings.validate_plan_invariants is True
    assert settings.shmem_ip_port == "tcp://10.0.0.1:29501"
    assert settings.shmem_stream_barrier is False


def test_missing_config_uses_every_default():
    settings = MoonEPSettings.from_config(None)
    assert settings.num_sms == 32
    assert settings.token_padding == 128
    assert settings.async_finish is False
    assert settings.tokens_per_rank is None


def test_shipped_config_defines_every_setting():
    """The shipped ``MoonEPConfig`` must cover all settings, so nothing warns.

    This is the check that would have caught a partially applied patch, where
    ``moonep.settings.py`` was copied but ``fsdp_turbo_config.py`` was not.
    """
    # fsdp_turbo_config imports PyYAML at module scope.
    pytest.importorskip("yaml")
    from fsdp_turbo.fsdp_turbo_config import MoonEPConfig

    config = MoonEPConfig()
    missing = [field.name for field in dataclasses.fields(MoonEPSettings) if not hasattr(config, field.name)]
    assert not missing, f"MoonEPConfig is missing {missing}; re-apply the config change"

    # The defaults must agree, or behaviour depends on whether a config was passed.
    defaults = MoonEPSettings()
    divergent = {
        field.name: (getattr(defaults, field.name), getattr(config, field.name))
        for field in dataclasses.fields(MoonEPSettings)
        if getattr(defaults, field.name) != getattr(config, field.name)
    }
    assert not divergent, f"default drift between MoonEPSettings and MoonEPConfig: {divergent}"


def test_every_config_knob_reaches_the_settings():
    """The other direction: a knob with no setting is silently dead.

    ``from_config`` walks the *settings* fields, so a knob added to
    ``MoonEPConfig`` alone is dropped without a warning -- the user sets it, no
    error appears, and nothing changes. Checked over the dataclass rather than a
    list so a knob added later is covered without touching this test.
    """
    pytest.importorskip("yaml")
    from fsdp_turbo.fsdp_turbo_config import MoonEPConfig

    settings_fields = {field.name for field in dataclasses.fields(MoonEPSettings)}
    dead = [field.name for field in dataclasses.fields(MoonEPConfig) if field.name not in settings_fields]
    assert not dead, (
        f"MoonEPConfig exposes {dead} but MoonEPSettings has no such field, so setting "
        "it has no effect and says nothing. Add it to MoonEPSettings."
    )


def non_default_for_field(field):
    """A value distinguishable from the default, whatever the field's type."""
    default = field.default
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 7
    if default is None:
        # tokens_per_rank / top_k are ints when set; shmem_ip_port is a string.
        return "tcp://10.0.0.9:29999" if "port" in field.name else 4096
    if isinstance(default, str):
        # A future str-default field must still get a distinguishable value,
        # or the propagate test below would compare equal values and pass
        # without checking anything.
        return default + "-changed"
    return default


def test_every_knob_propagates_when_it_differs_from_the_default():
    """Guards against a hand-written mapping that drops or crosses a field.

    ``from_config`` currently copies by name in a loop, so this passes trivially;
    the value is in the day that loop is replaced by explicit assignments.
    """
    pytest.importorskip("yaml")
    from fsdp_turbo.fsdp_turbo_config import MoonEPConfig

    config = MoonEPConfig()
    expected = {}
    for field in dataclasses.fields(MoonEPConfig):
        name = field.name
        value = non_default_for_field(field)
        setattr(config, name, value)
        expected[name] = value

    settings = MoonEPSettings.from_config(config)
    wrong = {
        name: (getattr(settings, name), value) for name, value in expected.items() if getattr(settings, name) != value
    }
    assert not wrong, f"knobs did not reach the settings (got, expected): {wrong}"


#: Derived from ``_INERT_BY_DEVICE`` rather than written out, so the cases
#: follow the code when a knob is added or reclassified.
INERT_PAIRS = [(device, knob) for device, knobs in MoonEPSettings._INERT_BY_DEVICE.items() for knob in knobs]
LIVE_PAIRS = [
    (other, knob)
    for device, knob in INERT_PAIRS
    for other, knobs in MoonEPSettings._INERT_BY_DEVICE.items()
    if other != device and knob not in knobs
]


def non_default_for_knob(knob):
    """A value distinguishable from the default, whatever the field's type."""
    default = getattr(MoonEPSettings(), knob)
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 7
    return "tcp://10.0.0.9:29999"


def test_the_knob_inventory_is_not_empty():
    """Guards the parametrizations against silently covering nothing."""
    assert INERT_PAIRS, "the inert table is empty"
    assert LIVE_PAIRS, "no knob is live on one accelerator and inert on the other"


@pytest.mark.parametrize("device_type, knob", INERT_PAIRS)
def test_a_knob_with_no_effect_here_is_reported(device_type, knob, caplog):
    settings = dataclasses.replace(MoonEPSettings(), **{knob: non_default_for_knob(knob)})

    assert knob in settings.inert_knobs(device_type)
    with caplog.at_level(logging.WARNING):
        settings.warn_about_inert_knobs(device_type)
    assert knob in caplog.text
    assert device_type in caplog.text


@pytest.mark.parametrize("device_type, knob", LIVE_PAIRS)
def test_a_knob_that_does_work_here_is_not_reported(device_type, knob, caplog):
    """The mirror of the case above: warning about a live knob is worse than
    saying nothing, because it trains the reader to ignore the warning.
    """
    settings = dataclasses.replace(MoonEPSettings(), **{knob: non_default_for_knob(knob)})

    assert not settings.inert_knobs(device_type)
    with caplog.at_level(logging.WARNING):
        settings.warn_about_inert_knobs(device_type)
    assert knob not in caplog.text


def test_defaults_are_never_reported_as_inert(caplog):
    """A default the user never touched is not worth a warning."""
    settings = MoonEPSettings()
    for device_type in ("cuda", "npu"):
        assert not settings.inert_knobs(device_type)
        with caplog.at_level(logging.WARNING):
            settings.warn_about_inert_knobs(device_type)
    assert caplog.text == ""


def test_an_unknown_device_reports_nothing():
    settings = dataclasses.replace(MoonEPSettings(), shmem_stream_barrier=True)
    assert not settings.inert_knobs("cpu")


@pytest.mark.parametrize("device_type", sorted(MoonEPSettings._INERT_BY_DEVICE))
def test_several_inert_knobs_are_reported_together(device_type, caplog):
    knobs = MoonEPSettings._INERT_BY_DEVICE[device_type]
    settings = dataclasses.replace(MoonEPSettings(), **{knob: non_default_for_knob(knob) for knob in knobs})

    assert len(settings.inert_knobs(device_type)) == len(knobs)
    with caplog.at_level(logging.WARNING):
        settings.warn_about_inert_knobs(device_type)
    for knob in knobs:
        assert knob in caplog.text


@pytest.mark.parametrize("knob", sorted(MoonEPSettings._ASYNC_ONLY))
def test_async_knobs_stop_being_inert_once_the_backend_has_a_side_stream(knob, caplog):
    """``npu_async_comm=True`` gives both knobs a target, so warning about
    them would send the reader looking for a problem that is not there.
    """
    settings = dataclasses.replace(MoonEPSettings(), **{knob: non_default_for_knob(knob)})

    assert knob in settings.inert_knobs("npu")
    assert knob not in settings.inert_knobs("npu", supports_async_comm=True)
    with caplog.at_level(logging.WARNING):
        settings.warn_about_inert_knobs("npu", supports_async_comm=True)
    assert knob not in caplog.text


def test_the_async_only_knobs_are_all_listed_as_inert_somewhere():
    """Subtracting a knob that was never in the table would be a silent no-op."""
    tabled = {knob for knobs in MoonEPSettings._INERT_BY_DEVICE.values() for knob in knobs}
    assert MoonEPSettings._ASYNC_ONLY <= tabled


def test_the_inert_table_only_names_real_knobs():
    """A typo in the table would make the warning unreachable."""
    for device_type, knobs in MoonEPSettings._INERT_BY_DEVICE.items():
        for knob in knobs:
            assert knob in {field.name for field in dataclasses.fields(MoonEPSettings)}, (
                f"{device_type} lists '{knob}', which is not a MoonEPSettings field"
            )

import weakref
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fsdp_turbo.quantization.core.recompute_phase import (
    MXFP8WeightPhase,
    ensure_mxfp8_backward_phase,
    mark_module_for_recompute,
    mxfp8_recompute_context,
    register_recompute_aware_group,
    select_recompute_aware_phase,
)


class _FakeEvent:
    def __init__(self, calls):
        self.calls = calls

    def record(self):
        self.calls.append("record")


class _FakeParamGroup:
    def __init__(self):
        self.calls = []
        self.is_unsharded = True
        self.unshard_async_op = False
        self.device_handle = Mock()
        self.device_handle.Event.side_effect = lambda: _FakeEvent(self.calls)
        self._fsdp_turbo_mxfp8_recompute_aware = True
        self._fsdp_turbo_mxfp8_weight_phase = MXFP8WeightPhase.RECOMPUTE_FORWARD
        self._fsdp_turbo_mxfp8_recompute_depth = 0
        self._reshard_after_forward = True

    def reshard(self):
        self.calls.append("reshard")
        self.is_unsharded = False

    def _wait_all_gather_streams_on_event(self, event):
        self.calls.append(("wait_compute", event))

    def unshard(self, async_op):
        self.calls.append(("unshard", async_op, self._fsdp_turbo_mxfp8_weight_phase))

    def wait_for_unshard(self):
        self.calls.append("wait_unshard")
        self.is_unsharded = True


def test_register_recompute_aware_group_from_marked_descendant():
    module = Mock()
    child = Mock()
    module.modules.return_value = iter((module, child))
    mark_module_for_recompute(child)
    param_group = Mock(modules=(module,))
    state = Mock(_fsdp_param_group=param_group, _modules=(module,))
    config = Mock(_recompute_aware=True)

    assert register_recompute_aware_group(state, config)
    assert param_group._fsdp_turbo_mxfp8_recompute_aware is True


def test_register_recompute_aware_group_from_custom_hook_module():
    managed_module = Mock()
    managed_module.modules.return_value = iter((managed_module,))
    hook_module = Mock()
    hook_module.modules.return_value = iter((hook_module,))
    mark_module_for_recompute(hook_module)
    param_group = Mock(modules=(managed_module,))
    state = Mock(
        _fsdp_param_group=param_group,
        _modules=(managed_module,),
        hook_module=weakref.ref(hook_module),
    )
    config = Mock(_recompute_aware=True)

    assert register_recompute_aware_group(state, config)


def test_non_resharding_group_does_not_enter_recompute_state_machine():
    param_group = _FakeParamGroup()
    param_group._reshard_after_forward = False
    state = Mock(_fsdp_param_group=param_group)
    config = Mock(_recompute_aware=True)
    training_state = SimpleNamespace(name="PRE_BACKWARD")

    assert select_recompute_aware_phase(state, config, training_state) is None

    module = Mock()
    with patch(
        "fsdp_turbo.quantization.core.recompute_phase._get_fsdp_state",
        return_value=state,
    ):
        ensure_mxfp8_backward_phase(module)
    assert not param_group.calls


def test_pre_backward_selects_recompute_forward_before_backward_handoff():
    param_group = Mock()
    param_group._fsdp_turbo_mxfp8_recompute_aware = True
    param_group._fsdp_turbo_mxfp8_weight_phase = MXFP8WeightPhase.FORWARD
    state = Mock(_fsdp_param_group=param_group)
    config = Mock(_recompute_aware=True)
    training_state = SimpleNamespace(name="PRE_BACKWARD")

    phase = select_recompute_aware_phase(state, config, training_state)

    assert phase == MXFP8WeightPhase.RECOMPUTE_FORWARD


def test_recompute_context_hands_forward_weight_off_to_backward():
    module = Mock()
    param_group = _FakeParamGroup()
    state = Mock(_fsdp_param_group=param_group)

    with (
        patch(
            "fsdp_turbo.quantization.core.recompute_phase._get_fsdp_state",
            return_value=state,
        ),
        mxfp8_recompute_context(module),
    ):
        param_group.calls.append("recompute")

    assert param_group._fsdp_turbo_mxfp8_weight_phase == MXFP8WeightPhase.BACKWARD
    assert "reshard" in param_group.calls
    assert ("unshard", False, MXFP8WeightPhase.BACKWARD) in param_group.calls


def test_ensure_backward_phase_is_idempotent():
    module = Mock()
    param_group = _FakeParamGroup()
    state = Mock(_fsdp_param_group=param_group)

    with patch(
        "fsdp_turbo.quantization.core.recompute_phase._get_fsdp_state",
        return_value=state,
    ):
        ensure_mxfp8_backward_phase(module)
        calls_after_first = list(param_group.calls)
        ensure_mxfp8_backward_phase(module)

    assert param_group.calls == calls_after_first


def test_concurrent_backward_nodes_wait_for_one_handoff():
    module = Mock()
    param_group = _FakeParamGroup()
    transition_started = Event()
    release_transition = Event()

    def wait_for_unshard():
        param_group.calls.append("wait_unshard")
        transition_started.set()
        assert release_transition.wait(timeout=5)
        param_group.is_unsharded = True

    param_group.wait_for_unshard = wait_for_unshard
    state = Mock(_fsdp_param_group=param_group)
    errors = []

    def ensure_phase():
        try:
            ensure_mxfp8_backward_phase(module)
        except (AssertionError, RuntimeError) as error:  # pragma: no cover - reported below
            errors.append(error)

    with patch(
        "fsdp_turbo.quantization.core.recompute_phase._get_fsdp_state",
        return_value=state,
    ):
        first = Thread(target=ensure_phase)
        second = Thread(target=ensure_phase)
        first.start()
        assert transition_started.wait(timeout=5)
        second.start()
        release_transition.set()
        first.join(timeout=5)
        second.join(timeout=5)

    assert not errors
    assert sum(call[0] == "unshard" for call in param_group.calls if isinstance(call, tuple)) == 1


def test_failed_handoff_poisons_the_phase():
    module = Mock()
    param_group = _FakeParamGroup()
    param_group.wait_for_unshard = Mock(side_effect=RuntimeError("collective failed"))
    state = Mock(_fsdp_param_group=param_group)

    with patch(
        "fsdp_turbo.quantization.core.recompute_phase._get_fsdp_state",
        return_value=state,
    ):
        try:
            ensure_mxfp8_backward_phase(module)
        except RuntimeError as error:
            assert "collective failed" in str(error)
        else:
            raise AssertionError("the injected collective failure must propagate")

        try:
            ensure_mxfp8_backward_phase(module)
        except RuntimeError as error:
            assert "poisoned" in str(error)
        else:
            raise AssertionError("a poisoned phase must fail closed")


def test_nested_replay_handoff_fails_closed():
    module = Mock()
    param_group = _FakeParamGroup()
    param_group._fsdp_turbo_mxfp8_recompute_depth = 2
    state = Mock(_fsdp_param_group=param_group)

    with patch(
        "fsdp_turbo.quantization.core.recompute_phase._get_fsdp_state",
        return_value=state,
    ):
        try:
            ensure_mxfp8_backward_phase(module)
        except RuntimeError as error:
            assert "Nested checkpoint replay" in str(error)
        else:
            raise AssertionError("nested replay must fail closed")

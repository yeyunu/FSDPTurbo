"""Regression tests: the registered replay context must be re-entrant.

PR 173 review comment 196628216: non-reentrant checkpoint calls ``context_fn()``
once during the original forward and keeps the returned replay context in the
graph, entering it again on every recompute. The old ``@contextmanager`` based
``_replay_context`` was a one-shot object, so a second backward over a retained
graph raised ``'_GeneratorContextManager' object has no attribute 'args'``.
Importing the DeepSeek-V4.1 adapter registers such a context globally, so plain
modules under ``recompute_wrapper`` were affected too.

The factory contract is pinned here as well: registered factories run once,
during the forward (so they capture forward-time state), while the replay context
object is entered once per recompute.
"""

# pytest injects fixtures through same-named test parameters, which pylint reads
# as redefining the fixture function.
# pylint: disable=redefined-outer-name

import pytest
import torch
from torch import nn

from fsdp_turbo.memory.recompute.recompute import (
    _replay_context,
    _replay_context_fns,
    recompute_wrapper,
    register_replay_context,
    unregister_replay_context,
)


class _ReplayRecorder:
    """Counts forward-time factory calls and replay-time entries separately."""

    def __init__(self):
        self.factory_calls = 0
        self.factory_modules = []
        self.enter_count = 0
        self.exit_count = 0


class _CountingReplayContext:
    """Re-entrant registered context: entering it again is legal by contract."""

    def __init__(self, recorder):
        self._recorder = recorder

    def __enter__(self):
        self._recorder.enter_count += 1
        return self

    def __exit__(self, *exc_info):
        self._recorder.exit_count += 1
        return False


class _FailingContext:
    def __enter__(self):
        raise RuntimeError("enter failed")

    def __exit__(self, *exc_info):
        return False


class _ToyModule(nn.Module):
    def forward(self, value):
        return value.sin().square()


@pytest.fixture
def recorder():
    """Register a counting replay context and always unregister it.

    The global registry is swapped for the duration of the test and restored
    afterwards: an adapter imported earlier in the session (DeepSeek-V4.1
    registers its replay barrier at import time) must not take part in these
    cases.
    """
    state = _ReplayRecorder()
    saved_factories = list(_replay_context_fns)
    _replay_context_fns.clear()

    def factory(module):
        state.factory_calls += 1
        state.factory_modules.append(module)
        return _CountingReplayContext(state)

    register_replay_context(factory)
    try:
        yield state
    finally:
        unregister_replay_context(factory)
        _replay_context_fns[:] = saved_factories


def _checkpointed_sin_square():
    def forward(value):
        return value.sin().square()

    return recompute_wrapper(forward), torch.randn(4, requires_grad=True)


def test_second_backward_on_retained_graph_reenters_replay_context(recorder):
    wrapped, value = _checkpointed_sin_square()
    output = wrapped(value)

    # The reviewer's minimal case: the factories already ran during the forward,
    # and nothing was entered until the first recompute.
    assert recorder.factory_calls == 1
    assert recorder.enter_count == 0

    output.sum().backward(retain_graph=True)
    assert recorder.enter_count == 1

    output.sum().backward()
    assert recorder.enter_count == 2
    assert recorder.exit_count == 2
    # Replay only re-enters the contexts captured at forward time.
    assert recorder.factory_calls == 1


def test_repeated_autograd_grad_on_retained_graph(recorder):
    wrapped, value = _checkpointed_sin_square()
    output = wrapped(value)

    (first,) = torch.autograd.grad(output.sum(), value, retain_graph=True)
    (second,) = torch.autograd.grad(output.sum(), value)

    assert torch.allclose(first, second)
    assert recorder.factory_calls == 1
    assert recorder.enter_count == 2
    assert recorder.exit_count == 2


def test_module_recompute_wrapper_passes_module_and_replays(recorder):
    module = _ToyModule()
    wrapped = recompute_wrapper(module.forward, module=module)
    value = torch.randn(4, requires_grad=True)
    output = wrapped(value)

    output.sum().backward(retain_graph=True)
    output.sum().backward()

    assert recorder.factory_modules == [module]
    assert recorder.enter_count == 2


def test_replay_context_is_reentrant_per_replay(recorder):
    context = _replay_context(None)
    assert recorder.factory_calls == 1
    assert recorder.enter_count == 0

    for _ in range(3):
        with context:
            pass

    assert recorder.enter_count == 3
    assert recorder.exit_count == 3
    assert recorder.factory_calls == 1


def test_base_context_is_reentered_on_every_replay(recorder):
    events = []

    class _BaseContext:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *exc_info):
            events.append("exit")
            return False

    context = _replay_context(None, base_context=_BaseContext())
    with context:
        pass
    with context:
        pass

    assert events == ["enter", "exit", "enter", "exit"]


def test_enter_failure_exits_contexts_entered_so_far(recorder):
    register_replay_context(lambda module: _FailingContext())
    context = _replay_context(None)

    with pytest.raises(RuntimeError, match="enter failed"):
        with context:
            pass

    assert recorder.enter_count == 1
    assert recorder.exit_count == 1

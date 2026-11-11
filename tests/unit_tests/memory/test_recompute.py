from contextlib import contextmanager, nullcontext
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from fsdp_turbo.memory.recompute.recompute import (
    RecomputePlanConfig,
    compose_checkpoint_contexts,
    get_recompute_modules,
    recompute_modules,
    recompute_wrapper,
)

CHECKPOINT = "fsdp_turbo.memory.recompute.recompute.checkpoint"
RECOMPUTE_WRAPPER = "fsdp_turbo.memory.recompute.recompute.recompute_wrapper"


class _ToyLayer(nn.Module):
    def forward(self, value):
        return self.attention(value)

    def attention(self, value):
        return value

    def unrelated(self, value):
        return value


class _ToyModel(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer() for _ in range(num_layers)])

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


def _wrapped_module_names(model, plan):
    with patch(RECOMPUTE_WRAPPER, side_effect=lambda function, *args, **kwargs: function) as mock_wrapper:
        recompute_modules(model, plan)
    names = {id(module): name for name, module in model.named_modules()}
    return [names[id(call.args[-1])] for call in mock_wrapper.call_args_list]


def test_list_plan_wraps_matched_modules_without_context_fn():
    model = _ToyModel()
    original_forward = model.layers[0].forward
    value = torch.zeros(2)

    with patch(CHECKPOINT, return_value="output") as mock_checkpoint:
        recompute_modules(model, ["layers.{*}"])
        assert model.layers[0](value) == "output"

    call_args, call_kwargs = mock_checkpoint.call_args
    assert call_args[0] == original_forward
    assert call_args[1] is value
    assert call_kwargs == {"use_reentrant": False}


def test_context_fn_factory_receives_matched_module_at_wiring():
    model = _ToyModel()
    context_fn = Mock()
    factory = Mock(return_value=context_fn)

    with patch(CHECKPOINT, return_value="output") as mock_checkpoint:
        recompute_modules(model, ["layers.{*}"], context_fn_factory=factory)
        model.layers[0](torch.zeros(2))

    assert [call.args for call in factory.call_args_list] == [(model.layers[0],), (model.layers[1],)]
    assert mock_checkpoint.call_args.kwargs == {"use_reentrant": False, "context_fn": context_fn}


def test_method_entry_wraps_named_method_only():
    model = _ToyModel()
    layer = model.layers[0]
    original_unrelated = layer.unrelated
    factory = Mock(return_value=None)
    input_context = Mock(side_effect=nullcontext)

    with patch(CHECKPOINT, return_value="output") as mock_checkpoint:
        recompute_modules(model, ["layers.{*}.attention"], context_fn_factory=factory, input_context=input_context)
        assert layer.attention("value") == "output"

    # Method entries cannot be marked per module, so the factory receives None.
    assert [args for args, _ in factory.call_args_list] == [(None,), (None,)]
    input_context.assert_called_once_with()
    assert mock_checkpoint.call_args.kwargs == {"use_reentrant": False}
    assert layer.unrelated == original_unrelated


def test_method_entry_with_missing_method_raises():
    with pytest.raises(RuntimeError, match="No callable method"):
        recompute_modules(_ToyModel(), ["layers.{*}.missing_method"])


def test_overlapping_patterns_wrap_each_module_once():
    model = _ToyModel()
    matched = get_recompute_modules(model, ["layers.0", "layers.{*}"])
    assert [name for name, _ in matched] == ["layers.0", "layers.1"]

    with patch(RECOMPUTE_WRAPPER, side_effect=lambda function, *args, **kwargs: function) as mock_wrapper:
        recompute_modules(model, ["layers.0", "layers.{*}"])
    assert [call.args[-1] for call in mock_wrapper.call_args_list] == [model.layers[0], model.layers[1]]


def test_overlapping_method_patterns_wrap_each_method_once():
    model = _ToyModel()
    with patch(RECOMPUTE_WRAPPER, side_effect=lambda function, *args, **kwargs: function) as mock_wrapper:
        recompute_modules(model, ["layers.{*}.attention", "layers.0.attention"])

    assert mock_wrapper.call_count == 2


def test_forward_method_does_not_duplicate_module_wrapper():
    model = _ToyModel()
    with patch(RECOMPUTE_WRAPPER, side_effect=lambda function, *args, **kwargs: function) as mock_wrapper:
        recompute_modules(model, ["layers.{*}.forward", "layers.{*}"])

    assert mock_wrapper.call_count == 2


def test_flatten_inputs_flattens_kwargs_and_preserves_aliases():
    received = {}

    def forward(positional, *, keyword):
        received["positional"] = positional
        received["keyword"] = keyword
        return positional

    wrapped = recompute_wrapper(forward, flatten_inputs=True)
    shared = torch.zeros(2)
    with patch(CHECKPOINT) as mock_checkpoint:
        wrapped(shared, keyword={"left": shared, "right": "text"})

    call_args, call_kwargs = mock_checkpoint.call_args
    flattened_function, *leaves = call_args
    assert call_kwargs == {"use_reentrant": False}
    # The aliased tensor occupies one leaf; the string is a leaf of its own.
    assert len(leaves) == 2
    assert leaves[0] is shared
    assert leaves[1] == "text"

    flattened_function(*leaves)
    assert received["positional"] is shared
    assert received["keyword"]["left"] is shared
    assert received["keyword"]["right"] == "text"


def test_flatten_inputs_real_checkpoint_backward():
    calls = []
    scale = torch.tensor(2.0, requires_grad=True)

    def forward(positional, *, keyword):
        calls.append((positional, keyword))
        return positional * scale + keyword["left"] * keyword["right"]

    shared = torch.ones(2, requires_grad=True)
    wrapped = recompute_wrapper(forward, flatten_inputs=True)
    output = wrapped(shared, keyword={"left": shared, "right": 3.0})
    output.sum().backward()

    assert len(calls) == 2
    assert torch.equal(shared.grad, torch.full_like(shared, 5.0))
    assert torch.equal(scale.grad, torch.tensor(2.0))
    for positional, keyword in calls:
        assert positional is shared
        assert keyword["left"] is shared


@pytest.mark.parametrize("ancestor", ["", "layers"])
def test_reject_nested_checkpoints(ancestor):
    with pytest.raises(ValueError, match="nested checkpoints"):
        recompute_modules(_ToyModel(), [ancestor, "layers.{*}"], reject_nested=True)


def test_past_key_values_is_bound_to_none():
    def forward(hidden_states, past_key_values=None, use_cache=True):
        return hidden_states

    value = torch.zeros(2)
    wrapped = recompute_wrapper(forward)
    with patch(CHECKPOINT) as mock_checkpoint:
        wrapped(value, ("cache",), use_cache=True)

    call_args, call_kwargs = mock_checkpoint.call_args
    assert call_args == (forward, value, None, True)
    assert call_kwargs == {"use_reentrant": False}

    def forward_without_cache(hidden_states):
        return hidden_states

    wrapped = recompute_wrapper(forward_without_cache)
    with patch(CHECKPOINT) as mock_checkpoint:
        wrapped(value, marker="keep")
    assert mock_checkpoint.call_args.kwargs == {"use_reentrant": False, "marker": "keep"}


def test_past_key_values_in_var_kwargs_is_none():
    def forward(hidden_states, **kwargs):
        return hidden_states

    wrapped = recompute_wrapper(forward)
    with patch(CHECKPOINT) as mock_checkpoint:
        wrapped(torch.zeros(2), past_key_values="cache", marker="keep")

    assert mock_checkpoint.call_args.kwargs == {
        "use_reentrant": False,
        "past_key_values": None,
        "marker": "keep",
    }


def test_compose_empty_checkpoint_contexts():
    forward_context, recompute_context = compose_checkpoint_contexts([])()
    with forward_context, recompute_context:
        pass


def test_compose_checkpoint_contexts_enters_and_exits_in_order():
    events = []

    @contextmanager
    def tracker(name):
        events.append(f"enter-{name}")
        try:
            yield name
        finally:
            events.append(f"exit-{name}")

    composed = compose_checkpoint_contexts(
        [
            lambda: (tracker("forward-a"), tracker("recompute-a")),
            lambda: (tracker("forward-b"), tracker("recompute-b")),
        ]
    )
    forward_context, recompute_context = composed()
    with forward_context:
        pass
    with recompute_context:
        pass

    assert events == [
        "enter-forward-a",
        "enter-forward-b",
        "exit-forward-b",
        "exit-forward-a",
        "enter-recompute-a",
        "enter-recompute-b",
        "exit-recompute-b",
        "exit-recompute-a",
    ]


def test_compose_checkpoint_contexts_exits_entered_contexts_on_enter_failure():
    events = []

    @contextmanager
    def entered_context():
        events.append("enter-first")
        try:
            yield
        finally:
            events.append("exit-first")

    class FailingContext:
        def __enter__(self):
            raise RuntimeError("enter failed")

        def __exit__(self, *exc_info):
            return False

    composed = compose_checkpoint_contexts(
        [
            lambda: (entered_context(), entered_context()),
            lambda: (FailingContext(), FailingContext()),
        ]
    )
    forward_context, _ = composed()
    with pytest.raises(RuntimeError, match="enter failed"):
        with forward_context:
            pass

    assert events == ["enter-first", "exit-first"]


def test_plan_config_equals_list_input():
    assert _wrapped_module_names(_ToyModel(), ["layers.{*}"]) == ["layers.0", "layers.1"]
    assert _wrapped_module_names(_ToyModel(), RecomputePlanConfig(apply_modules=["layers.{*}"])) == [
        "layers.0",
        "layers.1",
    ]


def test_unmatched_pattern_raises_runtime_error():
    with pytest.raises(RuntimeError, match="No module named"):
        recompute_modules(_ToyModel(), ["missing.block.{*}"])
    with pytest.raises(RuntimeError, match="No module named"):
        get_recompute_modules(_ToyModel(), ["missing.{*}"])

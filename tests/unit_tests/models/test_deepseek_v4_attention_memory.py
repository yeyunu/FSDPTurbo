import ast
import sys
import weakref
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional

_MODULE_PATH = Path(__file__).parents[3] / "examples" / "deepseekv4" / "modeling_deepseek_v4.py"


def _repeat_kv(hidden_states, repeats):
    batch_size, kv_heads, sequence_length, head_dim = hidden_states.shape
    if repeats == 1:
        return hidden_states
    return (
        hidden_states[:, :, None]
        .expand(batch_size, kv_heads, repeats, sequence_length, head_dim)
        .reshape(batch_size, kv_heads * repeats, sequence_length, head_dim)
    )


def _load_attention(ulysses_size=1, retain_temporaries=False):
    """Load the actual attention function without importing fused backend modules."""
    module = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    function = deepcopy(
        next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "eager_attention_forward"
        )
    )
    if retain_temporaries:
        function.body = [node for node in function.body if not isinstance(node, ast.Delete)]
    text = ast.unparse(function)
    filename = "deepseek_v4_attention_keep_buffers" if retain_temporaries else "deepseek_v4_attention_release_buffers"
    parallel_state = SimpleNamespace(
        get_ulysses_rank=lambda: ulysses_size - 1,
        get_ulysses_group_size=lambda: ulysses_size,
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": functional,
        "repeat_kv": _repeat_kv,
        "get_parallel_state": lambda: parallel_state,
    }
    compiled = compile(text, filename, "exec")
    function_code = next(
        constant
        for constant in compiled.co_consts
        if isinstance(constant, CodeType) and constant.co_name == "eager_attention_forward"
    )
    default_values = tuple(ast.literal_eval(value) for value in function.args.defaults)
    attention = partial(FunctionType(function_code, namespace, argdefs=default_values))
    return attention, text, filename


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("shared_kv", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("ulysses_size", [1, 2])
def test_attention_releasing_buffers_preserves_output_and_gradients(
    dtype, kv_heads, shared_kv, use_mask, dropout, ulysses_size
):
    torch.manual_seed(123)
    query_data = torch.randn(1, 4, 6, 4).to(dtype)
    key_data = torch.randn(1, kv_heads, 8, 4).to(dtype)
    value_data = torch.randn_like(key_data)
    sink_data = torch.randn(4 * ulysses_size).to(dtype)
    upstream = torch.randn(1, 6, 4, 4).to(dtype)
    attention_mask = None
    if use_mask:
        attention_mask = torch.zeros(1, 1, 6, 8)
        attention_mask.masked_fill_(torch.arange(8)[None, :] > torch.arange(6)[:, None], float("-inf"))

    results = []
    for retain_temporaries in (True, False):
        attention, _, _ = _load_attention(ulysses_size, retain_temporaries)
        query = query_data.clone().requires_grad_()
        key = key_data.clone().requires_grad_()
        value = key if shared_kv else value_data.clone().requires_grad_()
        sink = sink_data.clone().requires_grad_()
        module = SimpleNamespace(num_key_value_groups=4 // kv_heads, sinks=sink, training=True)
        torch.manual_seed(456)
        output, weights = attention(module, query, key, value, attention_mask, 0.5, dropout=dropout)
        output.backward(upstream)
        results.append((output.detach(), weights.detach(), query.grad, key.grad, value.grad, sink.grad))

    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [1, 4])
@pytest.mark.parametrize("ulysses_size", [1, 2])
def test_causal_attention_blocks_future_tokens_and_gradients(dtype, kv_heads, ulysses_size):
    torch.manual_seed(123)
    attention, _, _ = _load_attention(ulysses_size)
    sequence_length = 6
    prefix_length = 2
    query = torch.ones(1, 4, sequence_length, 4, dtype=dtype)
    key = torch.randn(1, kv_heads, sequence_length, 4, dtype=dtype, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    module = SimpleNamespace(
        num_key_value_groups=4 // kv_heads,
        sinks=torch.zeros(4 * ulysses_size, dtype=dtype),
        training=False,
    )
    future_positions = torch.arange(sequence_length)[None, :] > torch.arange(sequence_length)[:, None]
    attention_mask = torch.zeros(1, 1, sequence_length, sequence_length)
    attention_mask.masked_fill_(future_positions, float("-inf"))

    output, weights = attention(module, query, key, value, attention_mask, 0.5)
    changed_key = key.detach().clone()
    changed_value = value.detach().clone()
    changed_key[:, :, prefix_length:] += 32
    changed_value[:, :, prefix_length:] += 64
    changed_output, changed_weights = attention(module, query, changed_key, changed_value, attention_mask, 0.5)

    torch.testing.assert_close(output[:, :prefix_length], changed_output[:, :prefix_length], rtol=0, atol=0)
    torch.testing.assert_close(weights[:, :, :prefix_length], changed_weights[:, :, :prefix_length], rtol=0, atol=0)
    assert torch.count_nonzero(weights.masked_select(future_positions)) == 0

    unmasked_output, _ = attention(module, query, key, value, None, 0.5)
    changed_unmasked_output, _ = attention(module, query, changed_key, changed_value, None, 0.5)
    assert not torch.allclose(unmasked_output[:, :prefix_length], changed_unmasked_output[:, :prefix_length])

    output[:, :prefix_length].sum().backward()
    assert torch.count_nonzero(key.grad[:, :, prefix_length:]) == 0
    assert torch.count_nonzero(value.grad[:, :, prefix_length:]) == 0
    assert torch.count_nonzero(value.grad[:, :, :prefix_length]) > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_releases_old_scores_before_dtype_conversion(dtype):
    attention, source, filename = _load_attention()
    lines = source.splitlines()
    references = {}
    observed = {}

    def trace(frame, event, arg):
        if event != "line" or frame.f_code.co_filename != filename:
            return trace
        line = lines[frame.f_lineno - 1].strip()
        if line.startswith("combined_logits = torch.cat"):
            references["masked_scores"] = weakref.ref(frame.f_locals["attn_weights"])
        if line.startswith("probs = F.softmax"):
            references["normalized_logits"] = weakref.ref(frame.f_locals["combined_logits"])
        if line.startswith("attn_weights = nn.functional.dropout"):
            tuple(frame.f_locals)
            observed.update({name: reference() is not None for name, reference in references.items()})
        return trace

    query = torch.randn(1, 4, 6, 4, dtype=dtype, requires_grad=True)
    key = torch.randn(1, 1, 8, 4, dtype=dtype, requires_grad=True)
    sinks = torch.randn(4, dtype=dtype, requires_grad=True)
    module = SimpleNamespace(num_key_value_groups=4, sinks=sinks, training=True)
    previous_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        output, _ = attention(module, query, key, key, torch.zeros(1, 1, 6, 8), 0.5)
    finally:
        sys.settrace(previous_trace)
    output.sum().backward()
    assert observed == {"masked_scores": False, "normalized_logits": False}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_nonreentrant_checkpoint_preserves_gradients(dtype):
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(123)
    query_data = torch.randn(1, 4, 6, 4).to(dtype)
    key_data = torch.randn(1, 1, 8, 4).to(dtype)
    sink_data = torch.randn(4).to(dtype)
    upstream = torch.randn(1, 6, 4, 4).to(dtype)
    results = []
    for retain_temporaries in (True, False):
        attention, _, _ = _load_attention(retain_temporaries=retain_temporaries)
        query = query_data.clone().requires_grad_()
        key = key_data.clone().requires_grad_()
        sinks = sink_data.clone().requires_grad_()

        def operation(query_states, kv_states, sink_weights, attention_function=attention):
            module = SimpleNamespace(num_key_value_groups=4, sinks=sink_weights, training=True)
            return attention_function(module, query_states, kv_states, kv_states, torch.zeros(1, 1, 6, 8), 0.5)[0]

        output = checkpoint(operation, query, key, sinks, use_reentrant=False)
        output.backward(upstream)
        results.append((output.detach(), query.grad, key.grad, sinks.grad))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

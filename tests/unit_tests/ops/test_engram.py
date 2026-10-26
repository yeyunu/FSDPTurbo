import torch

from fsdp_turbo.ops import registry
from fsdp_turbo.ops.engram import engram_backward, engram_forward
from fsdp_turbo.ops.registry import get_op

CLAMP_VALUE = 1e-6
EPS = 1e-6


def test_engram_forward_dispatches_to_registered_cpu_op(monkeypatch):
    calls = []

    def fake_engram_forward(*args, **kwargs):
        calls.append((args, kwargs))
        return ("value_out", "scores", "keys", "value", "gates")

    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cpu"))
    monkeypatch.setitem(registry._OPERATOR_REGISTRY["engram_forward"], "cpu", fake_engram_forward)

    result = engram_forward("h", "e", "wk", "wv", "kg", "qg", clamp_value=2.0, eps=3.0)

    assert result == ("value_out", "scores", "keys", "value", "gates")
    assert calls == [(("h", "e", "wk", "wv", "kg", "qg"), {"clamp_value": 2.0, "eps": 3.0})]


def test_engram_has_cpu_fallback():
    assert get_op("engram_forward", "cpu").__name__ == "engram_forward_cpu"
    assert get_op("engram_backward", "cpu").__name__ == "engram_backward_cpu"


def _reference_engram(
    hidden_states,
    embeddings,
    key_proj_weights,
    value_proj_weights,
    key_gamma,
    query_gamma,
    grad_output,
):
    """Autograd-differentiated FP32 reference of the fused engram math."""
    dim = hidden_states.shape[-1]

    head_count, _, head_dim = key_proj_weights.shape
    keys = torch.nn.functional.linear(
        embeddings, key_proj_weights.transpose(1, 2).reshape(head_count * head_dim, -1)
    ).unflatten(-1, (head_count, head_dim))
    value = embeddings @ value_proj_weights

    def rms_norm(x, gamma):
        normed = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPS)
        return normed * gamma

    normed_keys = rms_norm(keys, key_gamma)
    normed_queries = rms_norm(hidden_states, query_gamma)
    scores = (normed_keys * normed_queries).sum(dim=-1) * dim**-0.5
    logits = torch.sign(scores) * scores.abs().clamp_min(CLAMP_VALUE).sqrt()
    gates = torch.sigmoid(logits)
    value_out = gates.unsqueeze(-1) * value.float().unsqueeze(2)

    gradients = torch.autograd.grad(
        value_out,
        (
            hidden_states,
            embeddings,
            key_proj_weights,
            value_proj_weights,
            key_gamma,
            query_gamma,
        ),
        grad_output,
    )
    return value_out, scores, keys, value, gates, gradients


def _make_inputs():
    torch.manual_seed(0)
    batch_size, sequence_length, head_count, dim, embedding_dim = 2, 5, 3, 32, 64
    hidden_states = torch.randn(batch_size, sequence_length, head_count, dim, requires_grad=True)
    embeddings = torch.randn(batch_size, sequence_length, embedding_dim, requires_grad=True)
    key_proj_weights = torch.randn(head_count, embedding_dim, dim, requires_grad=True) * (embedding_dim**-0.5)
    value_proj_weights = torch.randn(embedding_dim, dim, requires_grad=True) * (embedding_dim**-0.5)
    key_gamma = torch.ones(head_count, dim, requires_grad=True)
    query_gamma = torch.ones(head_count, dim, requires_grad=True)
    grad_output = torch.randn(batch_size, sequence_length, head_count, dim) * 0.1
    return (
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
        grad_output,
    )


def test_engram_cpu_matches_autograd_reference():
    (
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
        grad_output,
    ) = _make_inputs()

    value_out, scores, keys, value, gates = engram_forward(
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
        clamp_value=CLAMP_VALUE,
        eps=EPS,
    )
    assert value_out.shape == hidden_states.shape
    assert scores.shape == gates.shape == (*hidden_states.shape[:3],)
    assert keys.shape == hidden_states.shape
    assert value.shape == (*hidden_states.shape[:2], hidden_states.shape[-1])

    (
        ref_value_out,
        ref_scores,
        ref_keys,
        ref_value,
        ref_gates,
        ref_gradients,
    ) = _reference_engram(
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
        grad_output,
    )

    torch.testing.assert_close(value_out, ref_value_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(scores, ref_scores, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gates, ref_gates, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(keys, ref_keys, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(value, ref_value, rtol=1e-5, atol=1e-6)

    gradients = engram_backward(
        grad_output,
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
        scores,
        gates,
        keys,
        value,
        clamp_value=CLAMP_VALUE,
        eps=EPS,
    )

    names = (
        "grad_hidden",
        "grad_embeddings",
        "grad_key_proj",
        "grad_value_proj",
        "grad_key_gamma",
        "grad_query_gamma",
    )
    for name, gradient, ref_gradient in zip(names, gradients, ref_gradients):
        assert gradient.shape == ref_gradient.shape, name
        torch.testing.assert_close(gradient, ref_gradient, rtol=2e-4, atol=1e-5)


def test_engram_backward_pretransposed_packed_views_matches_standard_layout():
    (
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
        grad_output,
    ) = _make_inputs()
    head_count, embedding_dim, dim = key_proj_weights.shape

    _, scores, keys, value, gates = engram_forward(
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
        clamp_value=CLAMP_VALUE,
        eps=EPS,
    )

    standard_gradients = engram_backward(
        grad_output,
        hidden_states,
        embeddings,
        key_proj_weights,
        value_proj_weights,
        key_gamma,
        query_gamma,
        scores,
        gates,
        keys,
        value,
        clamp_value=CLAMP_VALUE,
        eps=EPS,
    )

    # Packed wkv parameter layout, sliced into the kernel-consumed views
    # exactly as EngramWkvFunction.backward does: [M,H,De] key and [H,De] value.
    packed_key_weights = key_proj_weights.transpose(1, 2).reshape(head_count * dim, embedding_dim)
    packed_value_weights = value_proj_weights.t().reshape(dim, embedding_dim)
    packed_weights = torch.cat([packed_key_weights, packed_value_weights])
    key_weight_view = packed_weights[: head_count * dim].view(head_count, dim, embedding_dim)
    value_weight_view = packed_weights[head_count * dim :]

    view_gradients = engram_backward(
        grad_output,
        hidden_states,
        embeddings,
        key_weight_view,
        value_weight_view,
        key_gamma,
        query_gamma,
        scores,
        gates,
        keys,
        value,
        clamp_value=CLAMP_VALUE,
        eps=EPS,
        pretransposed_weights=True,
    )

    names = (
        "grad_hidden",
        "grad_embeddings",
        "grad_key_proj",
        "grad_value_proj",
        "grad_key_gamma",
        "grad_query_gamma",
    )
    for name, standard, view in zip(names, standard_gradients, view_gradients):
        assert view.shape == standard.shape, name
        torch.testing.assert_close(view, standard, rtol=2e-4, atol=1e-5)

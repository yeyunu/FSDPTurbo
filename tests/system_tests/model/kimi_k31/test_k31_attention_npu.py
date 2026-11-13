# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Hardware regressions for K31 packed vision, CANN SFA/SFAG and TorchNPU indexer training."""

from copy import deepcopy

import pytest
import torch

from fsdp_turbo.utils.device import set_accelerator_compatible


@pytest.fixture(scope="module", name="accelerator")
def _accelerator():
    """Initialize the accelerator only for explicitly selected hardware tests."""
    set_accelerator_compatible()
    torch.accelerator.set_device_index(0)
    torch.accelerator.matmul.allow_hf32 = False
    torch.use_deterministic_algorithms(True)
    return torch.accelerator.current_accelerator()


def relative_error(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("grids", [((1, 1, 2), (1, 1, 3)), ((2, 1, 2), (1, 2, 3))])
def test_packed_vision_output_and_gradient_isolation(accelerator, grids):
    from fsdp_turbo.models.kimi.kimi_k31.modeling_kimi_k31 import MoonViT3dEncoder, Rope2DPosEmbRepeated

    torch.manual_seed(61)
    encoder = MoonViT3dEncoder(128, 1, dict(hidden_dim=128, num_heads=1, mlp_dim=256))
    encoder.rope_2d = Rope2DPosEmbRepeated(128, 8, 8)
    encoder = encoder.bfloat16().to(accelerator)
    grid = torch.tensor(grids, device=accelerator)
    lengths = grid.prod(-1).tolist()
    hidden = torch.randn(sum(lengths), 128, device=accelerator, dtype=torch.bfloat16, requires_grad=True)
    combined = encoder(hidden, grid)
    separate = torch.cat([encoder(part, row.unsqueeze(0)) for part, row in zip(hidden.split(lengths), grid)])
    assert relative_error(combined, separate) < 0.01
    upstream = torch.randn_like(combined)
    combined_gradient = torch.autograd.grad(combined, hidden, upstream, retain_graph=True)[0]
    separate_gradient = torch.autograd.grad(separate, hidden, upstream)[0]
    assert relative_error(combined_gradient, separate_gradient) < 0.02
    isolated = torch.autograd.grad(combined[: lengths[0]].float().square().sum(), hidden)[0]
    assert torch.count_nonzero(isolated[lengths[0] :]).item() == 0


def _sparse_inputs(accelerator, query_heads, sequence):
    from fsdp_turbo.ops.lightning_indexer import lightning_indexer_topk

    torch.manual_seed(62)
    shapes = [
        (1, sequence, query_heads, 512),
        (1, sequence, 1, 512),
        (1, sequence, query_heads, 64),
        (1, sequence, 1, 64),
    ]
    inputs = [(torch.randn(shape, device=accelerator, dtype=torch.bfloat16) * 0.1).requires_grad_() for shape in shapes]
    index_shapes = [(1, sequence, 32, 128), (1, sequence, 128), (1, sequence, 32)]
    index_inputs = [
        (torch.randn(shape, device=accelerator, dtype=torch.bfloat16) * 0.1).requires_grad_() for shape in index_shapes
    ]
    # Keep the LI output width equal to the requested top-k. Extra -1 columns
    # must not masquerade as a larger sparse_count in the native kernels.
    indices = lightning_indexer_topk(*index_inputs, 2048)
    return inputs, index_inputs, indices


@pytest.mark.parametrize("query_heads, sequence", [(32, 128), (96, 2112)], ids=["dense-32-heads", "sparse-96-heads"])
def test_sparse_mla_against_reference(accelerator, query_heads, sequence):
    from fsdp_turbo.ops.sparse_attention import sparse_flash_attention

    inputs, _, indices = _sparse_inputs(accelerator, query_heads, sequence)
    keys = indices.squeeze(2)
    assert ((keys < 0) | (keys <= torch.arange(sequence, device=accelerator)[None, :, None])).all()
    if sequence > 2048:
        assert (keys[:, -1] >= 0).sum().item() == 2048 < sequence
    query, key, query_rope, key_rope = inputs
    actual = sparse_flash_attention(
        query,
        key,
        key,
        indices.squeeze(-2),
        192**-0.5,
        rope_head_dim=query_rope.shape[-1],
        query_rope=query_rope,
        key_rope=key_rope,
        return_softmax_stats=True,
    )
    reference_inputs = [item.detach().cpu().requires_grad_() for item in inputs]
    query, key, query_rope, key_rope = reference_inputs
    expected = sparse_flash_attention(
        query,
        key,
        key,
        indices.cpu().squeeze(-2),
        192**-0.5,
        rope_head_dim=query_rope.shape[-1],
        query_rope=query_rope,
        key_rope=key_rope,
        return_softmax_stats=True,
    )
    for result, reference in zip(actual, expected):
        assert relative_error(result, reference) < 0.03
    if sequence > 2048:
        assert relative_error(actual[0][:, -64:], expected[0][:, -64:]) < 0.03
    assert all(not item.requires_grad and item.dtype == torch.float32 for item in actual[1:])
    upstream = torch.randn_like(actual[0])
    if sequence > 2048:
        upstream[:, :-64] = 0
    gradients = torch.autograd.grad(actual[0], inputs, upstream)
    expected_gradients = torch.autograd.grad(expected[0], reference_inputs, upstream.cpu())
    for result, reference in zip(gradients, expected_gradients):
        assert relative_error(result, reference) < 0.05


def test_sparse_mla_96_head_nondeterministic_pipeline(accelerator):
    """Exercise the N>64 non-deterministic SFAG pipeline from PR #14872."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        test_sparse_mla_against_reference(accelerator, 96, 2112)
    finally:
        torch.use_deterministic_algorithms(deterministic)


def test_indexer_kl_against_reference(accelerator):
    """Check the original TorchNPU KL path on its supported 32-head contract."""
    from fsdp_turbo.ops.cpu.lightning_indexer import lightning_indexer_kl_loss_cpu
    from fsdp_turbo.ops.lightning_indexer import lightning_indexer_kl_loss
    from fsdp_turbo.ops.sparse_attention import sparse_flash_attention

    inputs, index_inputs, indices = _sparse_inputs(accelerator, 32, 128)
    query, key, query_rope, key_rope = inputs
    actual = sparse_flash_attention(
        query,
        key,
        key,
        indices.squeeze(-2),
        192**-0.5,
        rope_head_dim=query_rope.shape[-1],
        query_rope=query_rope,
        key_rope=key_rope,
        return_softmax_stats=True,
    )
    expected = [None, actual[1].cpu(), actual[2].cpu()]
    loss = lightning_indexer_kl_loss(
        *index_inputs, *[item.detach() for item in inputs], indices, actual[1], actual[2], 192**-0.5
    )
    reference_index = [item.detach().cpu().requires_grad_() for item in index_inputs]
    expected_loss = lightning_indexer_kl_loss_cpu(
        *reference_index, *[item.detach().cpu() for item in inputs], indices.cpu(), expected[1], expected[2], 192**-0.5
    )
    torch.testing.assert_close(loss.cpu(), expected_loss, rtol=0.05, atol=1e-5)
    gradients = torch.autograd.grad(loss, index_inputs)
    expected_gradients = torch.autograd.grad(expected_loss, reference_index)
    for result, reference in zip(gradients, expected_gradients):
        assert relative_error(result, reference) < 0.08


def test_fused_dsa_model_forward_backward_and_optimizer_step(accelerator):
    """Verify native KL training at 32 heads; 96-head fused KL remains deferred."""
    from fsdp_turbo.models.kimi.kimi_k31.configuration_kimi_k31 import KimiLinearConfig
    from fsdp_turbo.models.kimi.kimi_k31.dsa import KimiDSAAttention

    torch.manual_seed(63)
    config = KimiLinearConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_experts=None,
        num_attention_heads=32,
        num_key_value_heads=32,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        mla_use_output_gate=True,
        attn_res_block_size=None,
        linear_attn_config={
            "kda_layers": [],
            "dsa_config": {
                "index_num_heads": 32,
                "index_head_dim": 128,
                "index_topk": 2048,
                "implementation": "ascendc",
            },
        },
    )
    config._attn_implementation = "eager"
    model = KimiDSAAttention(config, 0).bfloat16().to(accelerator)
    reference = deepcopy(model)
    reference.indexer.implementation = "eager"
    hidden = torch.randn(1, 128, 256, device=accelerator, dtype=torch.bfloat16, requires_grad=True)
    expected_hidden = hidden.detach().clone().requires_grad_()
    output = model(hidden)
    expected = reference(expected_hidden)
    assert relative_error(output, expected) < 0.04
    loss = output.float().square().mean() + model.indexer_loss
    expected_loss = expected.float().square().mean() + reference.indexer_loss
    torch.testing.assert_close(loss, expected_loss, rtol=0.05, atol=1e-4)
    loss.backward()
    expected_loss.backward()
    assert relative_error(hidden.grad, expected_hidden.grad) < 0.06
    norms = {}
    for name, parameter in model.named_parameters():
        expected_gradient = dict(reference.named_parameters())[name].grad
        assert parameter.grad is not None and expected_gradient is not None, name
        assert relative_error(parameter.grad, expected_gradient) < 0.12, name
        norms[name] = parameter.grad.float().norm().item()
    assert all(value > 0 for value in norms.values())
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    # BF16 scale parameters near 1 need an update larger than their ULP.
    torch.optim.AdamW(model.parameters(), lr=1e-2).step()
    unchanged = [name for name, parameter in model.named_parameters() if torch.equal(parameter, before[name])]
    assert not unchanged, unchanged

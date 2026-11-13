"""Direct sparse compressor outputs match stock masks, KV and gradients."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACompressor,
    DeepseekV4HCACompressor,
    eager_attention_forward,
)

from fsdp_turbo.models.deepseek.deepseek_v4 import modeling
from fsdp_turbo.ops.cpu.sparse_attention import indexed_sparse_attention_torch


@pytest.mark.parametrize("layer_type", ["heavily_compressed_attention", "compressed_sparse_attention"])
@pytest.mark.parametrize("nonzero_bias", [False, True], ids=["zero_init", "nonzero_init"])
def test_compressor_indices_match_stock_after_position_bias_update(layer_type: str, nonzero_bias: bool) -> None:
    """Compare direct HCA/CSA indices, KV and gradients before and after SGD."""
    generator = torch.Generator().manual_seed(296)
    config = DeepseekV4Config(
        hidden_size=32,
        num_hidden_layers=1,
        layer_types=[layer_type],
        num_attention_heads=4,
        head_dim=16,
        q_lora_rank=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=2,
        sliding_window=3,
        partial_rotary_factor=0.5,
    )
    compressor_class = (
        DeepseekV4HCACompressor if layer_type == "heavily_compressed_attention" else DeepseekV4CSACompressor
    )
    reference = compressor_class(config).train()
    sparse_class = (
        modeling.DeepseekV4HCACompressor
        if layer_type == "heavily_compressed_attention"
        else modeling.DeepseekV4CSACompressor
    )
    compressor = sparse_class(config).train()
    with torch.no_grad():
        for name, parameter in compressor.named_parameters():
            if name.endswith("position_bias") and not nonzero_bias:
                parameter.zero_()
            elif name.endswith("norm.weight"):
                parameter.fill_(1)
            else:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    # Keep the production compression rates (HCA=128, CSA=4) and multiple blocks.
    batch_size, sequence_length = 2, 3 * compressor.compress_rate
    hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size, generator=generator)
    q_residual = torch.randn(batch_size, sequence_length, config.q_lora_rank, generator=generator)
    position_ids = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1)
    query = torch.randn(batch_size, config.num_attention_heads, sequence_length, config.head_dim, generator=generator)
    query.requires_grad_()
    original = torch.randn(batch_size, 1, sequence_length, config.head_dim, generator=generator, requires_grad=True)
    positions = position_ids[0]
    original_allowed = (positions[None, :] <= positions[:, None]) & (
        positions[None, :] > positions[:, None] - config.sliding_window
    )
    original_mask = original.new_zeros(batch_size, 1, sequence_length, sequence_length).masked_fill(
        ~original_allowed, -torch.inf
    )
    module = SimpleNamespace(
        compressor=compressor,
        config=config,
        layer_type=layer_type,
        sliding_window=config.sliding_window,
        sinks=torch.zeros(config.num_attention_heads, requires_grad=True),
        num_key_value_groups=config.num_attention_heads,
        training=True,
    )
    optimizer = torch.optim.SGD([compressor.position_bias], lr=0.1)
    initial_bias = compressor.position_bias.detach().clone()
    loss_weights = torch.randn(batch_size, 1, 3, config.head_dim, generator=generator)

    for step in range(2):
        reference.load_state_dict(compressor.state_dict(), strict=True)
        reference.zero_grad()
        compressor.zero_grad()
        reference_compressed, block_bias = reference(hidden_states, q_residual, position_ids, None, 0)
        compressed, direct_indices = compressor(hidden_states, q_residual, position_ids, None, 0, return_indices=True)
        torch.testing.assert_close(compressed, reference_compressed, rtol=0, atol=0)
        # The default interface used by eager attention and CP remains stock-compatible.
        default_compressed, default_mask = compressor(hidden_states, q_residual, position_ids, None, 0)
        torch.testing.assert_close(default_compressed, reference_compressed, rtol=0, atol=0)
        torch.testing.assert_close(default_mask, block_bias, rtol=0, atol=0)
        assert torch.isfinite(compressed).all()
        # Stock masks are the visibility oracle; routing is covered in sparse_attention tests.
        allowed = block_bias[:, 0] == 0
        indices = direct_indices
        if layer_type == "heavily_compressed_attention":
            assert indices is None
        else:
            selected = torch.zeros(batch_size, sequence_length, 4, dtype=torch.bool)
            selected.scatter_(-1, indices.long().masked_fill(indices < 0, 3), True)
            assert torch.equal(selected[..., :3], allowed)

        # Exercise the real CPU indexed-attention backend against the stock dense reference.
        with mock.patch.object(modeling, "indexed_sparse_attention", indexed_sparse_attention_torch):
            sparse_output, _ = modeling._deepseek_v4_sparse_attention(
                module,
                query,
                original,
                original_mask,
                config.head_dim**-0.5,
                compressed_key_value=compressed,
                compressed_topk_indices=direct_indices,
            )
        dense_key_value = torch.cat((original, reference_compressed), dim=2)
        dense_mask = torch.cat((original_mask, block_bias), dim=-1)
        dense_output, _ = eager_attention_forward(
            module,
            query,
            dense_key_value,
            dense_key_value,
            dense_mask,
            config.head_dim**-0.5,
        )
        torch.testing.assert_close(sparse_output, dense_output, rtol=1e-5, atol=1e-6)
        sparse_gradients = torch.autograd.grad(
            sparse_output.square().mean(),
            (query, original, module.sinks, compressor.position_bias),
            retain_graph=True,
        )
        dense_gradients = torch.autograd.grad(
            dense_output.square().mean(),
            (query, original, module.sinks, reference.position_bias),
            retain_graph=True,
        )
        for actual_gradient, expected_gradient in zip(sparse_gradients, dense_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-4, atol=1e-8)

        (compressed * loss_weights).sum().backward()
        (reference_compressed * loss_weights).sum().backward()
        for name, parameter in compressor.named_parameters():
            reference_gradient = dict(reference.named_parameters())[name].grad
            if reference_gradient is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, reference_gradient, rtol=0, atol=0)
        if step == 0:
            gradient = compressor.position_bias.grad
            assert gradient is not None and torch.isfinite(gradient).all()
            assert gradient.abs().sum() > 0
            optimizer.step()
            optimizer.zero_grad()
            assert not torch.equal(compressor.position_bias, initial_bias)


@pytest.mark.parametrize("layer_type", ["heavily_compressed_attention", "compressed_sparse_attention"])
@pytest.mark.parametrize("length_kind", ["short", "one_block", "trailing_tokens"])
def test_sparse_compressor_partial_windows(layer_type: str, length_kind: str) -> None:
    """Match stock pooling at empty, single-block and incomplete-window boundaries."""
    config = DeepseekV4Config(
        hidden_size=32,
        num_hidden_layers=1,
        layer_types=[layer_type],
        num_attention_heads=4,
        head_dim=16,
        q_lora_rank=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=2,
        partial_rotary_factor=0.5,
    )
    compressor_class = (
        modeling.DeepseekV4HCACompressor
        if layer_type == "heavily_compressed_attention"
        else modeling.DeepseekV4CSACompressor
    )
    compressor = compressor_class(config)
    generator = torch.Generator().manual_seed(296)
    with torch.no_grad():
        for parameter in compressor.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    ratio = compressor.compress_rate
    sequence_length = {"short": ratio - 1, "one_block": ratio, "trailing_tokens": 3 * ratio + 1}[length_kind]
    hidden = torch.randn(2, sequence_length, config.hidden_size, generator=generator)
    residual = torch.randn(2, sequence_length, config.q_lora_rank, generator=generator)
    positions = torch.arange(sequence_length).unsqueeze(0).expand(2, -1)
    expected, mask = compressor(hidden, residual, positions, None, 0)
    actual, indices = compressor(hidden, residual, positions, None, 0, return_indices=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if layer_type == "heavily_compressed_attention":
        assert indices is None
    else:
        block_count = sequence_length // ratio
        selected = torch.zeros(2, sequence_length, block_count + 1, dtype=torch.bool)
        selected.scatter_(-1, indices.long().masked_fill(indices < 0, block_count), True)
        assert torch.equal(selected[..., :block_count], mask[:, 0] == 0)


@pytest.mark.parametrize("layer_type", ["heavily_compressed_attention", "compressed_sparse_attention"])
def test_attention_constructs_sparse_compressor_with_stock_state_dict(layer_type: str) -> None:
    """Production Attention installs subclasses without changing checkpoint keys."""
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention

    config = DeepseekV4Config(
        hidden_size=32,
        num_hidden_layers=1,
        layer_types=[layer_type],
        num_attention_heads=4,
        head_dim=16,
        q_lora_rank=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=2,
        partial_rotary_factor=0.5,
        o_groups=2,
        o_lora_rank=8,
    )
    reference = DeepseekV4Attention(config, 0)
    attention = modeling.DeepseekV4Attention(config, 0)
    attention.load_state_dict(reference.state_dict(), strict=True)
    expected_class = (
        modeling.DeepseekV4HCACompressor
        if layer_type == "heavily_compressed_attention"
        else modeling.DeepseekV4CSACompressor
    )
    assert isinstance(attention.compressor, expected_class)


@pytest.mark.parametrize("layer_type", ["compressed_sparse_attention", "heavily_compressed_attention"])
@pytest.mark.parametrize("complete_block", [False, True], ids=["before_boundary", "at_boundary"])
def test_sparse_compressor_short_sequence_reaches_kernel(layer_type: str, complete_block: bool) -> None:
    """Exercise real compression, V4 bridging and NPU validation across the first block boundary."""
    from fsdp_turbo.ops.npu import sparse_attention

    config = DeepseekV4Config(
        hidden_size=32,
        num_hidden_layers=1,
        layer_types=[layer_type],
        num_attention_heads=2,
        head_dim=512,
        q_lora_rank=16,
        index_n_heads=2,
        index_head_dim=64,
        index_topk=2,
        sliding_window=128,
        partial_rotary_factor=0.125,
    )
    compressor_class = (
        modeling.DeepseekV4CSACompressor
        if layer_type == "compressed_sparse_attention"
        else modeling.DeepseekV4HCACompressor
    )
    compressor = compressor_class(config)
    generator = torch.Generator().manual_seed(296)
    with torch.no_grad():
        for parameter in compressor.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    ratio = compressor.compress_rate
    sequence_length = ratio if complete_block else ratio - 1
    hidden = torch.randn(1, sequence_length, config.hidden_size, generator=generator)
    residual = torch.randn(1, sequence_length, config.q_lora_rank, generator=generator)
    positions = torch.arange(sequence_length).unsqueeze(0)
    compressed, indices = compressor(hidden, residual, positions, None, 0, return_indices=True)
    assert compressed.shape[2] == int(complete_block)
    query = torch.zeros(1, 2, sequence_length, 512, dtype=torch.bfloat16)
    original = torch.zeros(1, 1, sequence_length, 512, dtype=query.dtype)
    mask = torch.zeros(1, 1, sequence_length, sequence_length, dtype=query.dtype).masked_fill(
        positions[:, None, :] > positions[:, :, None], -torch.inf
    )
    module = SimpleNamespace(
        compressor=compressor,
        layer_type=layer_type,
        sliding_window=config.sliding_window,
        sinks=torch.zeros(2),
    )
    expected = query.transpose(1, 2).contiguous()
    # Select the real NPU wrapper explicitly; only the hardware autograd boundary is stubbed.
    with (
        mock.patch.object(modeling, "indexed_sparse_attention", sparse_attention.indexed_sparse_attention_npu),
        mock.patch.object(sparse_attention._SparseFlashMlaFunction, "apply", return_value=expected) as kernel,
    ):
        output, weights = modeling._deepseek_v4_sparse_attention(
            module,
            query,
            original,
            mask,
            512**-0.5,
            compressed_key_value=compressed,
            compressed_topk_indices=indices,
        )
    assert output is expected and weights is None
    kernel.assert_called_once()
    arguments = kernel.call_args.args
    assert arguments[6] == (ratio if complete_block else 1)
    if complete_block:
        assert arguments[2].shape == (1, 1, 1, 512)
        torch.testing.assert_close(arguments[2][:, :, 0], compressed[:, 0].to(query.dtype))
        if layer_type == "compressed_sparse_attention":
            torch.testing.assert_close(arguments[3][..., 0, : indices.shape[-1]], indices)
        else:
            assert arguments[3] is None
    else:
        assert arguments[2] is None and arguments[3] is None
    assert compressor.compress_rate == ratio
    assert config.compress_rates[layer_type] == ratio

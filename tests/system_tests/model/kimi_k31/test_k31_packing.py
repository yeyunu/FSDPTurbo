# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Hardware packing regressions for short-convolution/KDA and native DSA."""

from copy import deepcopy
from contextlib import ExitStack
import importlib
from unittest.mock import patch

import pytest
import torch

from fsdp_turbo.models.kimi.kimi_k31.configuration_kimi_k31 import KimiLinearConfig
from fsdp_turbo.models.kimi.kimi_k31.dsa import KimiDSAAttention
from fsdp_turbo.models.kimi.kimi_k31.modeling_kimi_linear import KimiDeltaAttention
from fsdp_turbo.utils.device import set_accelerator_compatible


@pytest.fixture(scope="module", name="accelerator")
def _accelerator():
    from fsdp_turbo.modules.kcp import _ensure_triton_driver

    set_accelerator_compatible()
    torch.accelerator.set_device_index(0)
    torch.accelerator.matmul.allow_hf32 = False
    torch.use_deterministic_algorithms(True)
    _ensure_triton_driver()
    return torch.accelerator.current_accelerator()


def assert_relative_close(actual, expected, tolerance):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = (actual - expected).norm() / expected.norm().clamp_min(1e-8)
    assert error.item() < tolerance, error.item()


def test_kda_packing_short_document_chunk_boundary_and_padding(accelerator):
    config = KimiLinearConfig(
        hidden_size=256,
        num_hidden_layers=1,
        num_experts=None,
        linear_attn_config=dict(
            kda_layers=[1],
            short_conv_kernel_size=4,
            head_dim=128,
            num_heads=2,
            use_full_rank_gate=True,
            gate_lower_bound=-5.0,
        ),
    )
    torch.manual_seed(1105)
    model = KimiDeltaAttention(config, 0).bfloat16().to(accelerator)
    with torch.no_grad():
        model.A_log.uniform_(1, 16).log_()
        model.dt_bias.zero_()
    separate = deepcopy(model)
    hidden = (torch.randn(1, 70, 256, device=accelerator, dtype=torch.bfloat16) * 0.1).requires_grad_()
    reference = hidden.detach().clone().requires_grad_()
    mask = torch.tensor([[1] * 67 + [0] * 3], device=accelerator)
    output = model(hidden, attention_mask=mask, cu_seqlens=torch.tensor([0, 2, 70], device=accelerator))
    expected = torch.cat([separate(reference[:, :2]), separate(reference[:, 2:67])], dim=1)
    assert_relative_close(output[:, :67], expected, 0.025)
    assert torch.count_nonzero(output[:, 67:]) == 0
    upstream = torch.randn_like(expected)
    (output[:, :67].float() * upstream.float()).sum().backward()
    (expected.float() * upstream.float()).sum().backward()
    assert_relative_close(hidden.grad[:, :67], reference.grad[:, :67], 0.05)
    for name, parameter in model.named_parameters():
        assert_relative_close(parameter.grad, dict(separate.named_parameters())[name].grad, 0.08)
    isolated = hidden.detach().clone().requires_grad_()
    result = model(isolated, cu_seqlens=torch.tensor([0, 2, 70], device=accelerator))
    gradient = torch.autograd.grad(result[:, 2:67].float().square().sum(), isolated)[0]
    assert torch.count_nonzero(gradient[:, :2]) == 0


@pytest.mark.parametrize("lengths", [(1, 2, 65), (64, 129), (65, 2113)])
@pytest.mark.parametrize("query_heads", [32, 96])
def test_native_dsa_packing_logits_kl_and_gradients(accelerator, lengths, query_heads):
    config = KimiLinearConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_experts=None,
        num_attention_heads=query_heads,
        num_key_value_heads=query_heads,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        mla_use_output_gate=True,
        attn_res_block_size=None,
        linear_attn_config=dict(kda_layers=[], full_attn_layers=[1]),
        dsa_config=dict(
            dsa_layers=[1], index_num_heads=32, index_head_dim=128, index_topk=2048, implementation="ascendc"
        ),
    )
    config._attn_implementation = "eager"
    torch.manual_seed(1106)
    model = KimiDSAAttention(config, 0).bfloat16().to(accelerator).train()
    separate = deepcopy(model)
    valid_length = sum(lengths)
    physical_length = valid_length + 7
    hidden = (torch.randn(1, physical_length, 256, device=accelerator, dtype=torch.bfloat16) * 0.1).requires_grad_()
    reference = hidden.detach().clone().requires_grad_()
    boundaries = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist(), physical_length], device=accelerator, dtype=torch.int32
    )
    mask = torch.tensor([[1] * valid_length + [0] * 7], device=accelerator)
    indexer_backend = importlib.import_module("fsdp_turbo.ops.npu.lightning_indexer")
    sfa_backend = importlib.import_module("cann_ops_transformer.ops.attention.sparse_flash_attention")
    sfag_backend = importlib.import_module("cann_ops_transformer.ops.attention.sparse_flash_attention_grad")
    with ExitStack() as stack:
        calls = [
            stack.enter_context(patch.object(module, name, wraps=getattr(module, name)))
            for module, name in (
                (indexer_backend.torch_npu, "npu_lightning_indexer"),
                (sfa_backend, "sparse_flash_attention"),
                (indexer_backend.torch_npu, "npu_sparse_lightning_indexer_grad_kl_loss"),
                (sfag_backend, "sparse_flash_attention_grad"),
            )
        ]
        output = model(hidden, attention_mask=mask, cu_seqlens=boundaries)
        assert [call.call_count for call in calls] == [1, 1, 1, 0]
        assert calls[0].call_args.kwargs["layout_query"] == "TND"
        assert calls[1].call_args.kwargs["layout_query"] == "TND"
        assert calls[2].call_args.kwargs["layout"] == "TND"
        (output[:, :valid_length].float().square().mean() + model.indexer_loss).backward(retain_graph=True)
        assert calls[3].call_count == 1
        assert calls[3].call_args.kwargs["layout"] == "TND"
    model.zero_grad(set_to_none=True)
    hidden.grad = None
    packed_kl = model.indexer_loss
    parts, losses = [], []
    start = 0
    for length in lengths:
        parts.append(separate(reference[:, start : start + length]))
        losses.append(separate.indexer_loss * length)
        start += length
    expected_kl = sum(losses) / valid_length
    expected = torch.cat(parts, dim=1)
    assert_relative_close(output[:, :valid_length], expected, 0.025)
    torch.testing.assert_close(packed_kl, expected_kl, rtol=0.05, atol=1e-5)
    assert torch.count_nonzero(output[:, valid_length:]) == 0
    upstream = torch.randn_like(expected)
    ((output[:, :valid_length].float() * upstream.float()).sum() + packed_kl).backward()
    ((expected.float() * upstream.float()).sum() + expected_kl).backward()
    assert_relative_close(hidden.grad[:, :valid_length], reference.grad[:, :valid_length], 0.05)
    assert torch.count_nonzero(hidden.grad[:, valid_length:]) == 0
    for name, parameter in model.named_parameters():
        assert_relative_close(parameter.grad, dict(separate.named_parameters())[name].grad, 0.08)
    altered = hidden.detach().clone()
    altered[:, : lengths[0]] *= 10
    changed = model(altered, attention_mask=mask, cu_seqlens=boundaries)
    torch.testing.assert_close(
        output[:, lengths[0] : valid_length], changed[:, lengths[0] : valid_length], rtol=0.01, atol=1e-4
    )
    isolated = hidden.detach().clone().requires_grad_()
    result = model(isolated, attention_mask=mask, cu_seqlens=boundaries)
    gradient = torch.autograd.grad(result[:, lengths[0] : valid_length].float().square().sum(), isolated)[0]
    assert torch.count_nonzero(gradient[:, : lengths[0]]) == 0


@pytest.mark.parametrize("packed", [False, True], ids=["BSND", "TND"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_native_96_head_kl_matches_cpu_reference(accelerator, packed, deterministic):
    """PR14709 KL must match independent FP32 loss/gradients in both layouts."""
    from fsdp_turbo.ops.cpu.lightning_indexer import lightning_indexer_kl_loss_cpu
    from fsdp_turbo.ops.lightning_indexer import lightning_indexer_topk, lightning_indexer_kl_loss
    from fsdp_turbo.ops.sparse_attention import sparse_flash_attention

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        torch.manual_seed(14709)
        shapes = [(196, 32, 128), (196, 128), (196, 32), (196, 96, 512), (196, 1, 512), (196, 96, 64), (196, 1, 64)]
        tensors = [torch.randn(shape, device=accelerator, dtype=torch.bfloat16) * 0.1 for shape in shapes]
        if not packed:
            tensors = [tensor.unsqueeze(0) for tensor in tensors]
        index_inputs = [tensor.requires_grad_() for tensor in tensors[:3]]
        attention_inputs = tensors[3:]
        boundaries = torch.tensor([0, 65, 196], device=accelerator, dtype=torch.int32) if packed else None
        indices = lightning_indexer_topk(*index_inputs, 2048, boundaries)
        query, key, query_rope, key_rope = attention_inputs
        _, maximum, denominator = sparse_flash_attention(
            query,
            key,
            key,
            indices.squeeze(-2),
            192**-0.5,
            rope_head_dim=query_rope.shape[-1],
            query_rope=query_rope,
            key_rope=key_rope,
            cu_seqlens=boundaries,
            return_softmax_stats=True,
        )
        loss = lightning_indexer_kl_loss(
            *index_inputs, *attention_inputs, indices, maximum, denominator, 192**-0.5, boundaries
        )
        reference_index = [tensor.detach().float().cpu().requires_grad_() for tensor in index_inputs]
        reference_loss = lightning_indexer_kl_loss_cpu(
            *reference_index,
            *[tensor.float().cpu() for tensor in attention_inputs],
            indices.cpu(),
            maximum.cpu(),
            denominator.cpu(),
            192**-0.5,
            None if boundaries is None else boundaries.cpu(),
        )
        torch.testing.assert_close(loss.float().cpu(), reference_loss, rtol=0.05, atol=1e-5)
        for actual, expected in zip(
            torch.autograd.grad(loss, index_inputs), torch.autograd.grad(reference_loss, reference_index)
        ):
            assert_relative_close(actual, expected, 0.08)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

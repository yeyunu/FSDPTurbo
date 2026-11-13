# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Kimi-K31 model semantics, CPU shard initialization and NPU integration."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from fsdp_turbo.models.kimi.kimi_k31.adapter import _initialize_embedding
from fsdp_turbo.models.kimi.kimi_k31.configuration_kimi_k31 import KimiLinearConfig
from fsdp_turbo.models.kimi.kimi_k31.dsa import KimiDSAAttention, KimiDSAModel, LightningIndexer
from fsdp_turbo.models.kimi.kimi_k31.modeling_kimi_linear import KimiMLAAttention, _apply_attention_residual
from fsdp_turbo.ops.cpu.attn_res import attn_res_cpu
from fsdp_turbo.ops.cpu.situ_glu import situ_glu_eager
from fsdp_turbo.ops.kda import chunk_kda
from fsdp_turbo.ops.situ_glu import fused_situ_glu as situ_glu
from fsdp_turbo.utils.device import set_accelerator_compatible


def configuration(topk=32):
    config = KimiLinearConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_experts=None,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        mla_use_nope=True,
        mla_use_output_gate=True,
        attn_res_block_size=None,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1, 2]},
        dsa_config={"dsa_layers": [1, 2], "index_num_heads": 2, "index_head_dim": 8, "index_topk": topk},
    )
    config._attn_implementation = "eager"
    return config


def test_dense_attention_parity_and_gradients():
    torch.manual_seed(42)
    config = configuration()
    dense = KimiMLAAttention(config, 0).double()
    sparse = KimiDSAAttention(config, 0).double()
    sparse.load_state_dict(dense.state_dict(), strict=False)
    hidden = torch.randn(1, 7, 32, dtype=torch.float64, requires_grad=True)
    reference = hidden.detach().clone().requires_grad_()
    mask = torch.zeros(1, 1, 7, 7, dtype=hidden.dtype).masked_fill(
        torch.ones(7, 7, dtype=torch.bool).triu(1), -torch.inf
    )
    output = sparse(hidden, mask)
    expected = dense(reference, mask)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    output.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(hidden.grad, reference.grad, rtol=1e-5, atol=1e-6)
    for name, parameter in dense.named_parameters():
        torch.testing.assert_close(dict(sparse.named_parameters())[name].grad, parameter.grad, rtol=1e-5, atol=1e-6)


def test_indexer_scores_and_gradients_against_scalar_reference():
    """Independent per-head/per-key formula checks weighting on the query axis."""
    torch.manual_seed(81)
    indexer = LightningIndexer(configuration(2))
    hidden = torch.randn(2, 4, 32, requires_grad=True)
    latent = torch.randn(2, 4, 16, requires_grad=True)
    scores = indexer(hidden, latent)
    rows = []
    for batch in range(2):
        queries = []
        for query in range(4):
            keys = []
            for key in range(4):
                key_vector = torch.nn.functional.layer_norm(
                    indexer.wk.weight @ hidden[batch, key].detach(),
                    (8,),
                    indexer.k_norm.weight,
                    indexer.k_norm.bias,
                    1e-6,
                )
                score = 0
                for head in range(2):
                    query_vector = indexer.wq_b.weight[head * 8 : (head + 1) * 8] @ latent[batch, query].detach()
                    weight = indexer.weights_proj.weight[head] @ hidden[batch, query].detach()
                    score = score + weight * (query_vector @ key_vector).relu() / (8 * 2) ** 0.5
                keys.append(score)
            queries.append(torch.stack(keys))
        rows.append(torch.stack(queries))
    reference = torch.stack(rows)
    torch.testing.assert_close(scores, reference, rtol=2e-5, atol=2e-6)
    actual = torch.autograd.grad(scores.square().sum(), tuple(indexer.parameters()), retain_graph=True)
    expected = torch.autograd.grad(reference.square().sum(), tuple(indexer.parameters()))
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=5e-5, atol=5e-6)
    assert hidden.grad is None and latent.grad is None


def test_bool_additive_masks_empty_rows_and_gradient_isolation():
    torch.manual_seed(82)
    model = KimiDSAAttention(configuration(2), 0)
    hidden = torch.randn(2, 7, 32, requires_grad=True)
    visible = torch.ones(2, 1, 7, 7, dtype=torch.bool).tril()
    visible[0] = False
    additive = torch.zeros_like(visible, dtype=torch.float32).masked_fill(~visible, -torch.inf)
    actual = model(hidden, visible)
    expected = model(hidden, additive)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual[0]) == 0
    auxiliary = model.indexer_loss
    main_gradients = torch.autograd.grad(
        actual.square().sum(), tuple(model.indexer.parameters()), allow_unused=True, retain_graph=True
    )
    assert all(value is None for value in main_gradients)
    hidden_gradient = torch.autograd.grad(auxiliary, hidden, allow_unused=True, retain_graph=True)[0]
    assert hidden_gradient is None
    empty = model(hidden, torch.zeros(2, 7))
    assert torch.count_nonzero(empty) == 0 and model.indexer_loss.item() == 0
    model.indexer_loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.indexer.parameters())


def test_sparse_causality_padding_and_indexer_gradients():
    torch.manual_seed(43)
    model = KimiDSAAttention(configuration(2), 0)
    hidden = torch.randn(1, 9, 32)
    original = model(hidden)
    changed = hidden.clone()
    changed[:, 6:] = 100 * torch.randn_like(changed[:, 6:])
    torch.testing.assert_close(model(changed)[:, :6], original[:, :6])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0]])
    first = model(hidden, mask)
    second = model(changed, mask)
    torch.testing.assert_close(first[:, :6], second[:, :6])
    (second.square().mean() + model.indexer_loss).backward()
    assert model.indexer.wq_b.weight.grad is not None
    assert model.indexer.wq_b.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.indexer.wq_b.weight.grad).all()


def test_all_full_attention_layers_are_dsa():
    with torch.device("meta"):
        model = KimiDSAModel(configuration())
    assert all(isinstance(layer.self_attn, KimiDSAAttention) for layer in model.layers)


def test_optional_gdn_does_not_disable_other_npu_operators():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("torch_npu is not installed")
    from fsdp_turbo.ops.registry import get_op

    for operator in ("flash_attention", "chunk_kda", "grouped_matmul", "permute", "rms_norm"):
        assert get_op(operator, "npu").__module__.startswith("fsdp_turbo.ops.npu.")


def test_packed_latent_moe_against_per_token_formula():
    """Independent routed/shared SiTU expressions check the packed expert adapter."""
    from fsdp_turbo.models.kimi.kimi_k31.moe import FSDPTurboKimiSparseMoeBlock

    torch.manual_seed(87)
    config = KimiLinearConfig(
        hidden_size=32,
        intermediate_size=64,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=8,
        routed_expert_hidden_size=16,
        latent_moe_use_norm=True,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    actual = FSDPTurboKimiSparseMoeBlock(config)
    actual.gate.e_score_correction_bias.data.zero_()
    expected = copy.deepcopy(actual)
    hidden = torch.randn(2, 5, 32, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    flat = other.reshape(-1, 32)
    scores = (flat @ expected.gate.weight.T).sigmoid()
    selected = scores.topk(2, dim=-1, sorted=False).indices
    weights = scores.gather(1, selected)
    weights = weights / weights.sum(-1, keepdim=True)
    latent = flat @ expected.routed_expert_down_proj.weight.T
    routed = []
    for token in range(10):
        result = 0
        for position in range(2):
            expert = int(selected[token, position])
            projected = latent[token] @ expected.experts.gate_up_proj[expert]
            gate, up = projected.chunk(2)
            activated = (4 * (gate / 4).tanh() * gate.sigmoid()) * (25 * (up / 25).tanh())
            result = result + weights[token, position] * (activated @ expected.experts.down_proj[expert])
        routed.append(result)
    routed = torch.stack(routed)
    routed = routed * torch.rsqrt(routed.square().mean(-1, keepdim=True) + config.rms_norm_eps)
    routed = (routed * expected.routed_expert_norm.weight) @ expected.routed_expert_up_proj.weight.T
    gate = flat @ expected.shared_experts.gate_proj.weight.T
    up = flat @ expected.shared_experts.up_proj.weight.T
    shared = (
        (4 * (gate / 4).tanh() * gate.sigmoid()) * (25 * (up / 25).tanh())
    ) @ expected.shared_experts.down_proj.weight.T
    reference = (routed + shared).reshape(2, 5, 32)
    output = actual(hidden)
    torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-6)
    output.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(hidden.grad, other.grad, rtol=3e-5, atol=3e-6)
    for name, parameter in actual.named_parameters():
        gradient = dict(expected.named_parameters())[name].grad
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=3e-5, atol=3e-6)


class KimiK31RegistrationTests(unittest.TestCase):
    def test_registration_is_lazy_and_preserves_legacy_mappings(self):
        package_name = '_test_kimi_k31'
        source = Path(__file__).resolve().parents[4] / 'fsdp_turbo/models/kimi/kimi_k31/__init__.py'
        specification = importlib.util.spec_from_file_location(package_name, source)
        package = importlib.util.module_from_spec(specification)
        with patch.dict(sys.modules, {'transformers': None, 'torch': None}):
            specification.loader.exec_module(package)

        legacy_config = type('LegacyKimiConfig', (), {})
        legacy_linear_config = type('LegacyLinearConfig', (), {})
        legacy_model = type('LegacyKimiModel', (), {})
        config_registry = {'kimi_k3': legacy_config, 'kimi_linear': legacy_linear_config}
        model_registry = {legacy_config: legacy_model}
        variant_config = type('KimiK31Config', (), {'model_type': 'kimi_k31'})
        variant_model = type('KimiK31ForConditionalGeneration', (), {})
        config_module = ModuleType(f'{package_name}.configuration_kimi_k31')
        config_module.KimiK31Config = variant_config
        model_module = ModuleType(f'{package_name}.dsa')
        model_module.KimiK31ForConditionalGeneration = variant_model

        class AutoConfig:
            @staticmethod
            def register(model_type, config_class, exist_ok=False):
                if model_type != config_class.model_type:
                    raise ValueError('model_type mismatch')
                if model_type in config_registry and not exist_ok:
                    raise ValueError('Duplicate config')
                config_registry[model_type] = config_class

        class AutoModel:
            @staticmethod
            def register(config_class, model_class, exist_ok=False):
                if config_class in model_registry and not exist_ok:
                    raise ValueError('Duplicate model')
                model_registry[config_class] = model_class

        transformers = ModuleType('transformers')
        transformers.AutoConfig = AutoConfig
        transformers.AutoModel = AutoModel
        with patch.dict(
            sys.modules,
            {
                package_name: package,
                f'{package_name}.configuration_kimi_k31': config_module,
                f'{package_name}.dsa': model_module,
                'transformers': transformers,
            },
        ):
            package.register_kimi_k31()
            package.register_kimi_k31()

        self.assertEqual(set(config_registry), {'kimi_k3', 'kimi_linear', 'kimi_k31'})
        self.assertIs(config_registry['kimi_k3'], legacy_config)
        self.assertIs(config_registry['kimi_linear'], legacy_linear_config)
        self.assertIs(config_registry['kimi_k31'], variant_config)
        self.assertIs(model_registry[legacy_config], legacy_model)
        self.assertIs(model_registry[variant_config], variant_model)


def _check_shards(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        for padding_idx in (1, 7):
            with torch.device("meta"):
                embedding = torch.nn.Embedding(8, 4, padding_idx=padding_idx)
            fully_shard(embedding, mesh=mesh)
            assert isinstance(embedding.weight, DTensor)
            assert embedding.weight.is_meta
            embedding.to_empty(device="cpu")
            assert embedding.weight.to_local().shape == (4, 4)
            _initialize_embedding(embedding, 0.02)
            weight = embedding.weight.full_tensor()
            assert torch.count_nonzero(weight[padding_idx]) == 0
            other_rows = torch.arange(8) != padding_idx
            assert torch.all(torch.count_nonzero(weight[other_rows], dim=1) == 4)
    finally:
        dist.destroy_process_group()


def test_sharded_embedding_padding(tmp_path):
    """Only the rank owning the padding row clears it, without a full allocation."""
    mp.spawn(_check_shards, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


def test_unsharded_embedding_padding():
    """Keep normal embedding initialization valid when FSDP is disabled."""
    embedding = torch.nn.Embedding(8, 4, padding_idx=7)
    _initialize_embedding(embedding, 0.02)
    assert torch.count_nonzero(embedding.weight[7]) == 0
    assert torch.isfinite(embedding.weight).all()


@pytest.fixture(scope="module")
def initialize_framework_device():
    set_accelerator_compatible()


@pytest.mark.usefixtures("initialize_framework_device")
class TestKimiK31NPU:
    """Single-process NPU checks; device initialization applies only here."""

    def test_dsa_npu_output_gradients_and_repeatability(self):
        """Exercise the actual 96-head, QK=192/V=128 attention contract with sparse masks."""
        torch.accelerator.set_device_index(0)
        torch.accelerator.matmul.allow_hf32 = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(83)
        config = KimiLinearConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_experts=None,
            num_attention_heads=96,
            num_key_value_heads=96,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            mla_use_nope=True,
            mla_use_output_gate=True,
            attn_res_block_size=None,
            linear_attn_config={"kda_layers": [], "full_attn_layers": [1]},
            dsa_config={"dsa_layers": [1], "index_num_heads": 2, "index_head_dim": 8, "index_topk": 2},
        )
        config._attn_implementation = "eager"
        reference = KimiDSAAttention(config, 0).bfloat16()
        model = copy.deepcopy(reference).to(torch.accelerator.current_accelerator())
        model.config = copy.deepcopy(config)
        model.config._attn_implementation = "flash_attention_2"
        hidden = torch.randn(2, 17, 32).bfloat16().requires_grad_()
        actual_hidden = hidden.detach().to(torch.accelerator.current_accelerator()).requires_grad_()
        mask = torch.ones(2, 17)
        mask[0, 14:] = 0
        expected = reference(hidden, mask)
        expected_loss = expected.float().square().mean() + reference.indexer_loss
        expected_loss.backward()
        outputs, gradients = [], []
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            actual_hidden.grad = None
            output = model(actual_hidden, mask.to(actual_hidden.device))
            loss = output.float().square().mean() + model.indexer_loss
            loss.backward()
            outputs.append(output.detach().cpu())
            gradients.append(actual_hidden.grad.detach().cpu())
        torch.testing.assert_close(outputs[0], expected, rtol=0.03, atol=0.006)
        torch.testing.assert_close(gradients[0], hidden.grad, rtol=0.05, atol=0.003)
        assert torch.equal(outputs[0], outputs[1]) and torch.equal(gradients[0], gradients[1])
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter.grad.cpu(), dict(reference.named_parameters())[name].grad, rtol=0.05, atol=0.003
            )

    @pytest.mark.parametrize(
        "implementation, tokens, expect_fused",
        [("auto", 64, True), (None, 64, True), ("eager", 64, False), ("ascendc", 63, False)],
    )
    def test_attention_residual_dispatch_and_gradients(self, implementation, tokens, expect_fused):
        """Check model backend selection and differentiable fallback across the token boundary."""
        from fsdp_turbo.ops.npu.attn_res import ATTN_RES_BACKEND, _attn_res_impl

        if expect_fused and ATTN_RES_BACKEND != "ascendc":
            pytest.skip("CANN fused attention residuals are unavailable")
        torch.accelerator.set_device_index(0)
        torch.use_deterministic_algorithms(True)
        config_dict = configuration().to_dict()
        config_dict.pop("attn_res_implementation")
        if implementation is not None:
            config_dict["attn_res_implementation"] = implementation
        config = KimiLinearConfig(**(config_dict | {"attn_res_block_size": 2}))
        assert config.attn_res_implementation == (implementation or "ascendc")
        config._attn_implementation = "eager"
        with torch.device("meta"):
            model = KimiDSAModel(config)
        torch.manual_seed(84)
        inputs = [
            torch.randn(tokens, 7168) * 0.1,
            torch.randn(tokens, 2, 7168) * 0.1,
            torch.randn(1, 7168) * 0.01,
            torch.ones(7168),
        ]
        inputs = [value.bfloat16().requires_grad_() for value in inputs]
        expected = attn_res_cpu(*inputs, 1e-5)
        expected.float().square().sum().backward()
        records = []
        for residual_function in (model.layers[0].attn_res_fn, model.attn_res_fn):
            device = torch.accelerator.current_accelerator()
            prefix, blocks, weight, norm = [value.detach().to(device).requires_grad_() for value in inputs]
            projection = torch.nn.Linear(weight.shape[-1], 1, bias=False, dtype=weight.dtype, device=device)
            projection.weight = torch.nn.Parameter(weight)
            with patch("fsdp_turbo.ops.npu.attn_res._attn_res_impl", wraps=_attn_res_impl) as selected_backend:
                actual = _apply_attention_residual(residual_function, prefix, blocks, projection, norm, 1e-5)
                actual.float().square().sum().backward()
            assert selected_backend.call_count == int(expect_fused)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0.025, atol=0.002)
            actual_inputs = (prefix, blocks, projection.weight, norm)
            for observed, reference in zip(actual_inputs, inputs):
                torch.testing.assert_close(observed.grad.cpu(), reference.grad, rtol=0.04, atol=0.003)
            records.append([actual.detach().cpu()] + [value.grad.cpu() for value in actual_inputs])
        assert all(torch.equal(left, right) for left, right in zip(*records))

    def test_kda_forward_and_all_input_gradients(self):
        """Cross a chunk boundary and exercise normalization, safe decay and beta sigmoid."""
        from fsdp_turbo.modules.kcp import _ensure_triton_driver

        set_accelerator_compatible()
        _ensure_triton_driver()
        torch.accelerator.set_device_index(0)
        torch.accelerator.matmul.allow_hf32 = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(85)
        shapes = [(1, 65, 2, 128)] * 4 + [(1, 65, 2), (2,), (256,)]
        inputs = [(torch.randn(shape) * 0.1).bfloat16().requires_grad_() for shape in shapes]
        query, key, value, raw_gate, raw_beta, rate, bias = inputs
        query = (query.float() * torch.rsqrt(query.float().square().sum(-1, keepdim=True) + 1e-6)).bfloat16().float()
        key = (key.float() * torch.rsqrt(key.float().square().sum(-1, keepdim=True) + 1e-6)).bfloat16().float()
        decay = (
            -5 * (rate.float().exp().view(1, 1, 2, 1) * (raw_gate.float() + bias.float().view(2, 128))).sigmoid()
        ).exp()
        beta = raw_beta.float().sigmoid().bfloat16().float()
        state = torch.zeros(1, 2, 128, 128)
        outputs = []
        for token in range(65):
            state = state * decay[:, token, :, :, None]
            residual = value[:, token].float() - torch.einsum("bhk,bhkv->bhv", key[:, token], state)
            state = state + key[:, token, :, :, None] * (beta[:, token, :, None] * residual)[:, :, None, :]
            outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, token], state) / 128**0.5)
        reference = torch.stack(outputs, dim=1).bfloat16()
        upstream = torch.randn_like(reference)
        (reference.float() * upstream.float()).sum().backward()
        actual_inputs = [
            tensor.detach().to(torch.accelerator.current_accelerator()).requires_grad_() for tensor in inputs
        ]
        actual, _ = chunk_kda(
            *actual_inputs[:5],
            A_log=actual_inputs[5],
            dt_bias=actual_inputs[6],
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            state_v_first=True,
            output_final_state=False,
        )
        (actual.float() * upstream.to(actual.device).float()).sum().backward()
        torch.testing.assert_close(actual.cpu(), reference, rtol=0.04, atol=5e-4)
        for observed, expected in zip(actual_inputs, inputs):
            relative = (
                observed.grad.cpu().float() - expected.grad.float()
            ).norm() / expected.grad.float().norm().clamp_min(1e-10)
            assert float(relative) < 0.08, float(relative)

    def test_situ_native_against_fp32_expression(self):
        set_accelerator_compatible()
        torch.accelerator.set_device_index(0)
        torch.use_deterministic_algorithms(True)
        from fsdp_turbo.ops.npu.situ_glu import _HAS_CANN_OPS_NN

        if not _HAS_CANN_OPS_NN:
            pytest.skip("CANN fused SiTU-GLU is unavailable")
        torch.manual_seed(86)
        source = (torch.randn(65, 6144) * 4).bfloat16().requires_grad_()
        expected = situ_glu_eager(source, beta=4.0, linear_beta=25.0)
        upstream = torch.randn_like(expected)
        (expected.float() * upstream.float()).sum().backward()
        records = []
        for _ in range(2):
            tensor = source.detach().to(torch.accelerator.current_accelerator()).requires_grad_()
            observed = situ_glu(tensor, beta=4.0, linear_beta=25.0)
            (observed.float() * upstream.to(tensor.device).float()).sum().backward()
            torch.testing.assert_close(observed.cpu(), expected, rtol=0.025, atol=0.04)
            torch.testing.assert_close(tensor.grad.cpu(), source.grad, rtol=0.03, atol=0.04)
            relative = (tensor.grad.cpu().float() - source.grad.float()).norm() / source.grad.float().norm()
            assert float(relative) < 0.01, float(relative)
            records.append((observed.detach().cpu(), tensor.grad.cpu()))
        assert all(torch.equal(left, right) for left, right in zip(*records))

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import sys
import types
import unittest
from unittest import mock

import torch

from fsdp_turbo.distributed.context_parallel.batch import pad_tensor_to_group, split_sequence_batch
from fsdp_turbo.distributed.context_parallel.loss import cp_compute_lm_loss
from fsdp_turbo.models.deepseek.deepseek_v4.context_parallel import build_raw_attention_mask
from fsdp_turbo.distributed.context_parallel.patch import (
    CP_PATCHES_ATTR,
    apply_cp_loss_patches,
    apply_function_patch_specs,
    install_model_context_parallel,
)
from fsdp_turbo.fsdp_turbo_config import CPPlanConfig, DistributedConfig, FSDPTurboConfig, MemoryConfig


class ContextParallelConfigTests(unittest.TestCase):
    def test_function_patch_specs_are_normalized(self):
        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                ulysses_parallel_size=2,
                cp_plan=CPPlanConfig(
                    ulysses_function_patches={
                        "target_functions": "package.module.attention",
                        "type": "full_attention",
                    },
                    kv_allgather_function_patches={
                        "target_functions": "package.module.kv_attention",
                        "type": "full_attention",
                    },
                    loss_function_patches={
                        "target_functions": "package.module.loss",
                        "type": "causal_lm_loss",
                    },
                ),
            )
        )
        self.assertEqual(
            config.distributed.cp_plan.ulysses_function_patches,
            [
                {
                    "target_functions": ["package.module.attention"],
                    "type": "full_attention",
                }
            ],
        )
        self.assertEqual(
            config.distributed.cp_plan.loss_function_patches,
            [
                {
                    "target_functions": ["package.module.loss"],
                    "type": "causal_lm_loss",
                }
            ],
        )
        self.assertEqual(
            config.distributed.cp_plan.kv_allgather_function_patches,
            [
                {
                    "target_functions": ["package.module.kv_attention"],
                    "type": "full_attention",
                }
            ],
        )

    def test_cp_patch_requirements_match_axis_interface(self):
        with self.assertRaisesRegex(ValueError, "ulysses_function_patches"):
            FSDPTurboConfig(distributed=DistributedConfig(ulysses_parallel_size=2))
        config = FSDPTurboConfig(distributed=DistributedConfig(kv_allgather_parallel_size=2))
        # validate_cp_config normalizes the unset Ulysses plan to an empty list;
        # the None/[] distinction only matters while checking Ulysses sizes.
        self.assertEqual(config.distributed.cp_plan, CPPlanConfig(ulysses_function_patches=[], mla_backend="ulysses"))
        checkpointed = FSDPTurboConfig(
            distributed=DistributedConfig(kv_allgather_parallel_size=2),
            memory=MemoryConfig(recompute=True, recompute_plan=["model.layers.{*}"]),
        )
        self.assertTrue(checkpointed.memory.recompute)
        with self.assertRaisesRegex(ValueError, "chunk_batch"):
            FSDPTurboConfig(
                distributed=DistributedConfig(kv_allgather_parallel_size=2),
                memory=MemoryConfig(chunk_batch=True),
            )

    def test_deepseek_compressor_rejects_dual_axis_cp(self):
        with self.assertRaisesRegex(ValueError, "cannot both be greater than 1"):
            FSDPTurboConfig(
                distributed=DistributedConfig(
                    kv_allgather_parallel_size=2,
                    ulysses_parallel_size=2,
                    cp_plan=CPPlanConfig(
                        kv_allgather_function_patches={
                            "target_functions": "package.module.attention",
                            "type": "compressor_full_attention",
                        },
                        ulysses_function_patches={
                            "target_functions": "package.module.attention",
                            "type": "compressor_full_attention",
                        },
                    ),
                )
            )

    def test_deepseek_compressor_supports_single_axis_cp(self):
        for kv_allgather_size, ulysses_size in ((2, 1), (1, 2)):
            with self.subTest(kv_allgather_size=kv_allgather_size, ulysses_size=ulysses_size):
                config = FSDPTurboConfig(
                    distributed=DistributedConfig(
                        kv_allgather_parallel_size=kv_allgather_size,
                        ulysses_parallel_size=ulysses_size,
                        cp_plan=CPPlanConfig(
                            kv_allgather_function_patches={
                                "target_functions": "package.module.attention",
                                "type": "compressor_full_attention",
                            },
                            ulysses_function_patches={
                                "target_functions": "package.module.attention",
                                "type": "compressor_full_attention",
                            },
                        ),
                    )
                )
                self.assertEqual(config.distributed.kv_allgather_parallel_size, kv_allgather_size)
                self.assertEqual(config.distributed.ulysses_parallel_size, ulysses_size)

    def test_deepseek_compressor_rejects_kcp_axis(self):
        with self.assertRaisesRegex(ValueError, "not supported on the KCP axis"):
            FSDPTurboConfig(
                distributed=DistributedConfig(
                    kcp_parallel_size=2,
                    cp_plan=CPPlanConfig(
                        kcp_function_patches={
                            "target_functions": "package.module.attention",
                            "type": "compressor_full_attention",
                        },
                    ),
                )
            )

    def test_generic_dual_axis_cp_is_not_rejected(self):
        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                kv_allgather_parallel_size=2,
                ulysses_parallel_size=2,
                cp_plan=CPPlanConfig(
                    kv_allgather_function_patches={
                        "target_functions": "package.module.attention",
                        "type": "full_attention",
                    },
                    ulysses_function_patches={
                        "target_functions": "package.module.attention",
                        "type": "full_attention",
                    },
                ),
            )
        )
        self.assertEqual(config.distributed.kv_allgather_parallel_size, 2)
        self.assertEqual(config.distributed.ulysses_parallel_size, 2)

    def test_kv_allgather_accepts_embedded_ulysses_patch_types(self):
        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                kv_allgather_parallel_size=16,
                ulysses_parallel_size=2,
                cp_plan=CPPlanConfig(
                    ulysses_function_patches=[
                        {"target_functions": ["a.b.C.forward"], "type": "qsa"},
                        {"target_functions": ["a.b.D.forward"], "type": "gated_delta_net"},
                    ],
                ),
            )
        )
        self.assertEqual(len(config.distributed.cp_plan.ulysses_function_patches), 2)

    def test_kv_allgather_rejects_unrelated_ulysses_patch_types(self):
        distributed = DistributedConfig(
            kv_allgather_parallel_size=2,
            cp_plan=CPPlanConfig(
                ulysses_function_patches=[
                    {"target_functions": ["a.b.C"], "type": "full_attention"},
                ],
            ),
        )
        with self.assertRaisesRegex(ValueError, "kv_allgather_function_patches"):
            FSDPTurboConfig(distributed=distributed)

    def test_kv_allgather_rejects_generic_kv_patches_mixed_with_embedded_types(self):
        distributed = DistributedConfig(
            kv_allgather_parallel_size=2,
            cp_plan=CPPlanConfig(
                ulysses_function_patches=[
                    {"target_functions": ["a.b.C.forward"], "type": "qsa"},
                ],
                kv_allgather_function_patches=[
                    {"target_functions": ["a.b.D"], "type": "full_attention"},
                ],
            ),
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            FSDPTurboConfig(distributed=distributed)


class DeepSeekV4AttentionMaskTests(unittest.TestCase):
    def test_packed_mask_preserves_batch_and_head_axes(self):
        global_position_ids = torch.tensor([[0, 1, 0, 1], [0, 1, 2, 0]])
        query_indices = torch.tensor([1, 3])

        mask = build_raw_attention_mask(
            None,
            global_position_ids,
            query_indices,
            2,
            torch.float32,
            None,
        )

        self.assertEqual(mask.shape, (2, 1, 2, 4))
        allowed = torch.tensor(
            [
                [[True, True, False, False], [False, False, True, True]],
                [[True, True, False, False], [False, False, False, True]],
            ]
        ).unsqueeze(1)
        expected = torch.zeros_like(mask).masked_fill(~allowed, torch.finfo(torch.float32).min)
        torch.testing.assert_close(mask, expected)

        query = torch.randn(2, 1, 2, 4, requires_grad=True)
        key = torch.randn(2, 1, 4, 4, requires_grad=True)
        value = torch.randn_like(key, requires_grad=True)
        probabilities = torch.softmax(query @ key.transpose(-1, -2) + mask, dim=-1)
        output = probabilities @ value
        output[:, :, :1].square().sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(key.grad).all())
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertEqual(float(key.grad[0, :, 2:].abs().sum()), 0.0)


class ContextParallelBatchTests(unittest.TestCase):
    @mock.patch("torch.distributed.get_rank", return_value=2)
    @mock.patch("torch.distributed.get_world_size", return_value=4)
    def test_sequence_batch_split_builds_position_ids_and_keeps_labels(self, _world_size, _rank):
        batch = {
            "input_ids": torch.arange(7).view(1, 7),
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
            "labels": torch.arange(7).view(1, 7),
        }
        output = split_sequence_batch(
            batch,
            object(),
            sequence_keys=("input_ids", "attention_mask"),
            build_position_ids=True,
        )
        torch.testing.assert_close(output["input_ids"], torch.tensor([[4, 5]]))
        torch.testing.assert_close(output["position_ids"], torch.tensor([[4, 5]]))
        torch.testing.assert_close(output["labels"], batch["labels"])


class ContextParallelLossTests(unittest.TestCase):
    @mock.patch("torch.distributed.get_rank", return_value=1)
    @mock.patch("torch.distributed.get_world_size", return_value=2)
    @mock.patch("fsdp_turbo.distributed.context_parallel.loss.cp_fixed_cross_entropy")
    def test_labels_pad_to_model_global_logits_length(self, cross_entropy, _world_size, _rank):
        cross_entropy.side_effect = lambda _source, target, **_kwargs: target
        logits = torch.zeros(1, 4, 8)
        labels = torch.arange(6).view(1, 6)

        local_targets = cp_compute_lm_loss(logits, labels, object())

        torch.testing.assert_close(
            local_targets,
            torch.tensor([5, -100, -100, -100]),
        )

    @mock.patch("torch.distributed.get_rank", return_value=1)
    @mock.patch("torch.distributed.get_world_size", return_value=2)
    @mock.patch("fsdp_turbo.distributed.context_parallel.loss.cp_fixed_cross_entropy")
    def test_pre_shifted_labels_are_padded_and_sharded(self, cross_entropy, _world_size, _rank):
        cross_entropy.side_effect = lambda _source, target, **_kwargs: target
        logits = torch.zeros(1, 4, 8)
        labels = torch.full((1, 6), 99)
        shift_labels = torch.arange(6).view(1, 6)

        local_targets = cp_compute_lm_loss(logits, labels, object(), shift_labels=shift_labels)

        torch.testing.assert_close(local_targets, torch.tensor([4, 5, -100, -100]))

    @mock.patch("torch.distributed.get_world_size", return_value=2)
    def test_labels_longer_than_global_logits_are_rejected(self, _world_size):
        logits = torch.zeros(1, 3, 8)
        labels = torch.arange(7).view(1, 7)

        with self.assertRaisesRegex(ValueError, "labels longer than the global logits"):
            cp_compute_lm_loss(logits, labels, object())


class ContextParallelPadTests(unittest.TestCase):
    @mock.patch("torch.distributed.get_world_size", return_value=4)
    def test_aligned_tensor_is_returned_unchanged(self, _world_size):
        tensor = torch.randn(2, 8, 3)
        self.assertIs(tensor, pad_tensor_to_group(tensor, object(), dim=1))

    @mock.patch("torch.distributed.get_world_size", return_value=4)
    def test_small_tensor_matches_fpad(self, _world_size):
        tensor = torch.arange(14, dtype=torch.float).view(1, 14)
        padded = pad_tensor_to_group(tensor, object(), dim=1, pad_value=0.0)
        self.assertEqual((1, 16), padded.shape)
        torch.testing.assert_close(padded, torch.nn.functional.pad(tensor, (0, 2), value=0.0))

    @mock.patch("torch.distributed.get_world_size", return_value=4)
    def test_padded_copy_matches_fpad(self, _world_size):
        # The NPU-safe pad assembles the result from a constant tail plus
        # chunked strided copies (ConstantPadNd overflows int32 offsets beyond
        # 2**31 elements); the result must stay bit-identical to F.pad,
        # including a negative pad value and dim=-1.
        tensor = torch.randn(1, 13, 5)
        padded = pad_tensor_to_group(tensor, object(), dim=1, pad_value=-1.0)
        expected = torch.nn.functional.pad(tensor, (0, 0, 0, 3, 0, 0), value=-1.0)
        self.assertEqual((1, 16, 5), padded.shape)
        torch.testing.assert_close(padded, expected)
        self.assertTrue(padded[0, 13:].eq(-1.0).all())

        tail = torch.full((6,), 7.0)
        padded = pad_tensor_to_group(tail, object(), dim=-1, pad_value=7.0)
        self.assertEqual((8,), padded.shape)
        torch.testing.assert_close(padded, torch.full((8,), 7.0))


class ContextParallelPatchTests(unittest.TestCase):
    module_name = "_fsdp_turbo_cp_patch_target"

    def setUp(self):
        self.module = types.ModuleType(self.module_name)
        self.trace = []

        def causal_loss(logits, labels, vocab_size, **kwargs):
            del labels, vocab_size, kwargs
            return logits.sum()

        def attention(value):
            self.trace.append("attention")
            return value + 1

        self.module.causal_loss = causal_loss
        self.module.attention = attention
        sys.modules[self.module_name] = self.module

    def tearDown(self):
        sys.modules.pop(self.module_name, None)

    def test_loss_patch_is_idempotent(self):
        target = f"{self.module_name}.causal_loss"
        specs = [{"target_functions": [target], "type": "causal_lm_loss"}]

        apply_cp_loss_patches(specs, object())
        first = getattr(self.module, "causal_loss")
        apply_cp_loss_patches(specs, object())

        self.assertIs(getattr(self.module, "causal_loss"), first)
        self.assertEqual(getattr(first, CP_PATCHES_ATTR), ("CP:causal_lm_loss",))

    def test_kv_then_ulysses_attention_patches_execute_in_filo_order(self):
        target = f"{self.module_name}.attention"
        specs = [{"target_functions": [target], "type": "full_attention"}]

        def patch_factory(label):
            def factory(original, _group):
                def wrapped(*args, **kwargs):
                    self.trace.append(f"{label}:before")
                    output = original(*args, **kwargs)
                    self.trace.append(f"{label}:after")
                    return output

                return wrapped

            return factory

        apply_function_patch_specs(
            specs,
            object(),
            {"full_attention": patch_factory("kv")},
            label="KVAllGather",
        )
        apply_function_patch_specs(
            specs,
            object(),
            {"full_attention": patch_factory("ulysses")},
            label="Ulysses",
        )

        self.assertEqual(getattr(self.module, "attention")(1), 2)
        self.assertEqual(
            self.trace,
            ["ulysses:before", "kv:before", "attention", "kv:after", "ulysses:after"],
        )

    def test_context_parallel_discovers_capability_inside_adapter(self):
        class Attention(torch.nn.Module):
            def forward(self, hidden_states, *args, **kwargs):
                return hidden_states

        class Backbone(torch.nn.Module):
            context_parallel_attention_class = Attention

            def __init__(self):
                super().__init__()
                self.attention = Attention()

            def context_parallel_attention_forward(self, *args, **kwargs):
                return self.attention(*args, **kwargs)

            def forward(self, input_ids=None, **kwargs):
                return self.attention(input_ids, **kwargs)

        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()

            def forward(self, input_ids=None, **kwargs):
                return self.model(input_ids=input_ids, **kwargs)

        adapter = Adapter()
        group = object()
        install_model_context_parallel(adapter, group, mode="ulysses")

        self.assertEqual(getattr(adapter.model, "_fsdp_turbo_context_parallel"), ("ulysses", group))
        self.assertEqual(getattr(adapter.model.attention, "_fsdp_turbo_context_parallel"), ("ulysses", group))

    def test_compressor_cp_restores_stock_deepseek_capability(self):
        from transformers.models.deepseek_v4 import DeepseekV4Config
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model

        from fsdp_turbo.distributed.context_parallel.kv_allgather.kv_allgather_context_parallel import (
            kv_allgather_context_parallelize_modules,
        )
        from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_context_parallel import (
            ulysses_context_parallelize_modules,
        )

        for mode in ("ulysses", "kv_allgather"):
            with self.subTest(mode=mode):
                with torch.device("meta"):
                    stock_model = DeepseekV4Model(DeepseekV4Config(num_hidden_layers=1))
                attention = stock_model.layers[0].self_attn
                group = object()
                spec = {"target_functions": [], "type": "compressor_full_attention"}

                if mode == "ulysses":
                    plan = types.SimpleNamespace(ulysses_function_patches=[spec])
                    ulysses_context_parallelize_modules(stock_model, group, plan)
                else:
                    plan = types.SimpleNamespace(kv_allgather_function_patches=[spec])
                    kv_allgather_context_parallelize_modules(stock_model, group, plan)

                self.assertEqual(getattr(stock_model, "_fsdp_turbo_context_parallel"), (mode, group))
                self.assertEqual(getattr(attention, "_fsdp_turbo_context_parallel"), (mode, group))

    def test_deepseek_adapter_forward_uses_only_inner_cp_container(self):
        from fsdp_turbo.models.deepseek.deepseek_v4.adapter import DeepseekV4ForCausalLMAdapter

        class Attention(torch.nn.Module):
            def forward(self, hidden_states, *args, **kwargs):
                return hidden_states

        class Backbone(torch.nn.Module):
            context_parallel_attention_class = Attention

            def __init__(self):
                super().__init__()
                self.attention = Attention()
                self.cp_state = None
                self.query_indices = None

            def context_parallel_attention_forward(
                self,
                attention_module,
                hidden_states,
                position_embeddings,
                query_indices,
                cp_state,
                group,
                gather_tensor,
                kwargs,
            ):
                del attention_module, position_embeddings, group, gather_tensor, kwargs
                self.cp_state = cp_state
                self.query_indices = query_indices
                return hidden_states, None

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=None,
                **kwargs,
            ):
                del use_cache
                hidden_states = input_ids if input_ids is not None else inputs_embeds
                hidden_states, _ = self.attention(
                    hidden_states,
                    None,
                    position_ids,
                    attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )
                return types.SimpleNamespace(loss=None, logits=hidden_states, hidden_states=None)

        class CausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                use_cache=False,
                output_hidden_states=False,
                **kwargs,
            ):
                del labels, output_hidden_states
                return self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    **kwargs,
                )

        adapter = DeepseekV4ForCausalLMAdapter(CausalLM(), model_args=object())
        group = object()
        install_model_context_parallel(adapter, group, mode="ulysses")

        self.assertIsNone(getattr(adapter, "_fsdp_turbo_context_parallel", None))
        self.assertEqual(getattr(adapter.model.model, "_fsdp_turbo_context_parallel"), ("ulysses", group))

        def gather_positions(tensor, _group, gather_dim):
            return torch.cat([tensor, tensor + tensor.shape[gather_dim]], dim=gather_dim)

        input_ids = torch.tensor([[1, 2]])
        with (
            mock.patch("fsdp_turbo.distributed.context_parallel.patch.dist.get_world_size", return_value=2),
            mock.patch("fsdp_turbo.distributed.context_parallel.patch.dist.get_rank", return_value=0),
            mock.patch(
                "fsdp_turbo.distributed.context_parallel.patch.kv_allgather",
                side_effect=gather_positions,
            ),
        ):
            output = adapter(input_ids=input_ids)

        torch.testing.assert_close(output.logits, input_ids)
        torch.testing.assert_close(adapter.model.model.cp_state["position_ids"], torch.tensor([[0, 1, 2, 3]]))
        torch.testing.assert_close(adapter.model.model.query_indices, torch.arange(4))

    def test_kda_patch_targets_remote_code_class_by_suffix(self):
        suffix = "modeling_kimi_linear_for_cp_test"
        dynamic_name = f"transformers_modules.checkpoint.{suffix}"
        dynamic_module = types.ModuleType(dynamic_name)

        class KimiDeltaAttention(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                del kwargs
                return hidden_states

        KimiDeltaAttention.__module__ = dynamic_name
        dynamic_module.KimiDeltaAttention = KimiDeltaAttention
        sys.modules[dynamic_name] = dynamic_module

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = KimiDeltaAttention()

        def factory(_original, _group):
            def replacement(_module, hidden_states, **kwargs):
                del kwargs
                return hidden_states + 3

            return replacement

        try:
            model = Model()
            specs = [
                {
                    "target_functions": [f"{suffix}.KimiDeltaAttention.forward"],
                    "type": "kda",
                }
            ]
            apply_function_patch_specs(specs, object(), {"kda": factory}, label="Ulysses", model=model)
            torch.testing.assert_close(model.attention(torch.tensor(2)), torch.tensor(5))
        finally:
            sys.modules.pop(dynamic_name, None)


if __name__ == "__main__":
    unittest.main()

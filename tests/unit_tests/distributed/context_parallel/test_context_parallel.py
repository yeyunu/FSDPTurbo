# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import sys
import types
import unittest
from unittest import mock

import torch

from fsdp_turbo.distributed.context_parallel.batch import split_sequence_batch
from fsdp_turbo.distributed.context_parallel.patch import (
    CP_PATCHES_ATTR,
    apply_cp_loss_patches,
    apply_function_patch_specs,
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
        self.assertEqual(config.distributed.cp_plan, CPPlanConfig())
        checkpointed = FSDPTurboConfig(
            distributed=DistributedConfig(kv_allgather_parallel_size=2),
            memory=MemoryConfig(recompute=True, recompute_plan=["model.layers.{*}"]),
        )
        self.assertTrue(checkpointed.memory.recompute)
        with self.assertRaisesRegex(ValueError, "does not support chunk_batch"):
            FSDPTurboConfig(
                distributed=DistributedConfig(kv_allgather_parallel_size=2),
                memory=MemoryConfig(chunk_batch=True),
            )


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
            with mock.patch("fsdp_turbo.modules.kda.get_ulysses_kda_forward", side_effect=factory):
                apply_function_patch_specs(specs, object(), {}, label="Ulysses", model=model)
            torch.testing.assert_close(model.attention(torch.tensor(2)), torch.tensor(5))
        finally:
            sys.modules.pop(dynamic_name, None)


if __name__ == "__main__":
    unittest.main()

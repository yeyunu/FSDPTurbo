# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for chunked micro-batch (ChunkMBS) execution.

Guards the contract relied on by MindSpeed-LLM's ``enable_chunk_batch``
feature: selected module forwards run on batch slices and the outputs are
concatenated back, so a chunked forward must be numerically identical to the
plain forward while never materialising the full batch inside the module.
All tests are CPU-only.
"""

import copy

import pytest
import torch
from torch import nn

from fsdp_turbo.fsdp_turbo_config import ChunkBatchPlanConfig
from fsdp_turbo.memory.chunk_batch.chunk_batch import (
    chunk_batch_modules,
    chunk_mbs_forward,
    get_chunkmbs_modules,
)


class _CountingModule(nn.Module):
    """Records the batch size of every forward call."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.seen_batch_sizes = []

    def forward(self, x, attention_mask=None):
        self.seen_batch_sizes.append(x.shape[0])
        out = self.linear(x)
        if attention_mask is not None:
            out = out + attention_mask
        return out


class TestChunkMbsForward:
    def test_chunked_output_matches_plain_forward(self):
        """Chunked execution must reproduce the plain forward, for both even
        and uneven splits.
        """
        torch.manual_seed(0)
        module = _CountingModule()
        wrapped = chunk_mbs_forward(chunk_mbs=1, batch_dim=0, chunk_arg_indexes=[0])(module.forward)

        # Even split: 4 rows in chunks of 1 -> four micro calls.
        x = torch.randn(4, 4)
        reference = module.forward(x)
        module.seen_batch_sizes.clear()
        assert torch.allclose(wrapped(x), reference)
        assert module.seen_batch_sizes == [1, 1, 1, 1]

        # Uneven split: 5 rows in chunks of 2 -> sizes 2, 2, 1 with the tail
        # handled by the min() bound.
        wrapped2 = chunk_mbs_forward(chunk_mbs=2, batch_dim=0, chunk_arg_indexes=[0])(module.forward)
        x = torch.randn(5, 4)
        reference = module.forward(x)
        module.seen_batch_sizes.clear()
        chunked = wrapped2(x)
        assert chunked.shape == (5, 4)
        assert torch.allclose(chunked, reference)
        assert module.seen_batch_sizes == [2, 2, 1]

    def test_slices_configured_kwargs_together_with_args(self):
        module = _CountingModule()
        wrapped = chunk_mbs_forward(
            chunk_mbs=1, batch_dim=0, chunk_arg_indexes=[0], chunk_kwarg_names=["attention_mask"]
        )(module.forward)
        x = torch.randn(3, 4)
        mask = torch.randn(3, 4)

        chunked = wrapped(x, attention_mask=mask)

        assert torch.allclose(chunked, module.forward(x, attention_mask=mask))

    def test_concatenates_tuple_outputs(self):
        def forward(x):
            return x * 2, x.sum(dim=-1)

        wrapped = chunk_mbs_forward(chunk_mbs=1, batch_dim=0, chunk_arg_indexes=[0])(forward)
        x = torch.randn(3, 4)

        out_a, out_b = wrapped(x)

        assert isinstance((out_a, out_b), tuple)
        assert torch.allclose(out_a, x * 2)
        assert torch.allclose(out_b, x.sum(dim=-1))


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def forward(self, x):
        hidden = self.embed(x)
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        return hidden


class TestChunkBatchModules:
    def test_no_matching_module_raises(self):
        model = _ToyModel()
        plan = ChunkBatchPlanConfig(chunk_mbs=1, apply_modules=["no.such.{*}"], batch_dim=0, chunk_arg_indexes=[0])

        with pytest.raises(RuntimeError, match=r"\[ChunkMBS\]"):
            get_chunkmbs_modules(model, plan.apply_modules)

    def test_wraps_only_pattern_matched_modules_and_preserves_numerics(self):
        torch.manual_seed(0)
        model = _ToyModel()
        reference = copy.deepcopy(model)
        plan = ChunkBatchPlanConfig(
            chunk_mbs=1,
            apply_modules=["layers.{*}"],
            batch_dim=0,
            chunk_arg_indexes=[0],
        )

        patched = chunk_batch_modules(model, plan)

        assert patched is model  # same instance, per the in-place patch contract
        for layer in model.layers:
            assert hasattr(layer.forward, "__wrapped__")  # decorated by chunk_mbs_forward
        assert not hasattr(model.embed.forward, "__wrapped__")

        x = torch.randn(4, 4)
        assert torch.allclose(model(x), reference(x))

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Numerical-equivalence tests for the combined Ulysses x KV-AllGather QSA forward.

Each spawned rank holds one contiguous CP shard (``cp_rank = kv * ulysses +
ulysses_rank``).  The patched ``Qwen4ExpTextAttention.forward`` is compared
against a single-process full-sequence reference for several topologies
(``kv x ulysses``), in both sparse-gather and dense-gather modes, including
forward and backward.  The Ulysses=4 topology exercises Q-head padding (6 real
heads padded to 8, with dummy heads clamped onto the last GQA group).
"""

import os
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F

from fsdp_turbo.distributed.context_parallel.ulysses import ulysses_attention as qsa_mod
from fsdp_turbo.ops.cpu.qsa import qsa_reference_mask
from fsdp_turbo.ops.qsa import apply_query_rope


def _init_gloo(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)


class _FakeRMSNorm(nn.Module):
    """Qwen4-Exp residual RMSNorm: ``x * rsqrt(mean(x^2) + eps) * (1 + weight)``."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1 + self.weight.float())
        return out.to(dtype)


class _FakeIndexer(nn.Module):
    def __init__(self, hidden, n_heads, head_dim, budget, rate):
        super().__init__()
        self.index_qk_proj = nn.Linear(hidden, (n_heads + 1) * head_dim, bias=False)
        self.q_layernorm = _FakeRMSNorm(head_dim)
        self.k_layernorm = _FakeRMSNorm(head_dim)
        self.index_n_heads = n_heads
        self.index_kv_heads = 1
        self.index_head_dim = head_dim
        self.token_budget = budget
        self.compress_ratio = rate


class _FakeQSAAttention(nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, idx_heads, idx_dim, budget, rate, chunk, sparse):
        super().__init__()
        self.q_proj = nn.Linear(hidden, heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = _FakeRMSNorm(head_dim)
        self.k_norm = _FakeRMSNorm(head_dim)
        self.indexer = _FakeIndexer(hidden, idx_heads, idx_dim, budget, rate)
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.attention_dropout = 0.0
        self.config = SimpleNamespace(
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            _attn_implementation="sdpa",
            qsa_cp_query_chunk_size=chunk,
            qsa_sparse_gather=sparse,
        )

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        raise AssertionError("original forward must not run under CP")


def _build_rope(seq_len, rotary_dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    freqs = torch.arange(seq_len, device=device).float()[:, None] * inv_freq[None, :]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None]
    return cos, sin


def _reference_forward(module, hidden, cos, sin, validity):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    batch, seq_len, _ = hidden.shape
    heads = module.config.num_attention_heads
    kv_heads = module.config.num_key_value_heads
    head_dim = module.head_dim
    query, gate = module.q_proj(hidden).view(batch, seq_len, heads, head_dim * 2).chunk(2, -1)
    gate = gate.reshape(batch, seq_len, -1)
    query = module.q_norm(query).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden).view(batch, seq_len, kv_heads, head_dim)).transpose(1, 2)
    value = module.v_proj(hidden).view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    key = key.repeat_interleave(heads // kv_heads, dim=1)
    value = value.repeat_interleave(heads // kv_heads, dim=1)

    indexer = module.indexer
    query_width = indexer.index_n_heads * indexer.index_head_dim
    projected = indexer.index_qk_proj(hidden)
    index_q, raw_keys = projected.split([query_width, indexer.index_head_dim], dim=-1)
    index_q = index_q.view(batch, seq_len, indexer.index_n_heads, indexer.index_head_dim)
    index_q = apply_query_rope(indexer.q_layernorm(index_q), cos, sin)

    keys = torch.arange(seq_len, device=hidden.device)
    causal = keys[None, :] <= keys[:, None]
    mask4d = causal[None, None] & validity[:, None, None, :].bool()
    selected = qsa_reference_mask(
        index_q,
        raw_keys,
        cos,
        sin,
        mask4d,
        indexer.k_layernorm.weight,
        rms_norm_eps=indexer.k_layernorm.eps,
        token_budget=indexer.token_budget,
        compress_ratio=indexer.compress_ratio,
    )
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask4d & selected, scale=module.scaling)
    out = out.transpose(1, 2).reshape(batch, seq_len, -1) * torch.sigmoid(gate)
    return module.o_proj(out)


class _GlooAllGather(torch.autograd.Function):
    """gloo-compatible stand-in for ``kv_allgather`` (reduce-scatter via all_reduce)."""

    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim = dim
        ctx.group = group
        shards = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
        dist.all_gather(shards, x.contiguous(), group=group)
        return torch.cat(shards, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        total = grad_output.contiguous().clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=ctx.group)
        world = dist.get_world_size(ctx.group)
        rank = dist.get_rank(ctx.group)
        per_rank = total.shape[ctx.dim] // world
        index = [slice(None)] * total.dim()
        index[ctx.dim] = slice(rank * per_rank, (rank + 1) * per_rank)
        return total[tuple(index)], None, None


def _qsa_worker(rank, world, port, kv_size, ulysses_size, sparse, seed):
    _init_gloo(rank, world, port)
    kv_rank = rank // ulysses_size
    try:
        ulysses_group = kv_group = None
        for kv in range(kv_size):
            members = [kv * ulysses_size + u for u in range(ulysses_size)]
            group = dist.new_group(members)
            if rank in members:
                ulysses_group = group
        for u in range(ulysses_size):
            members = [kv * ulysses_size + u for kv in range(kv_size)]
            group = dist.new_group(members)
            if rank in members:
                kv_group = group
        cp_group = dist.group.WORLD

        # Identical module and inputs on every rank.
        torch.manual_seed(seed)
        module = _FakeQSAAttention(
            hidden=32,
            heads=6,
            kv_heads=2,
            head_dim=16,
            idx_heads=4,
            idx_dim=16,
            budget=24,
            rate=4,
            chunk=16,
            sparse=sparse,
        )
        seq_len = 96
        assert seq_len % (world * module.indexer.compress_ratio) == 0
        torch.manual_seed(seed + 1)
        hidden_full = torch.randn(1, seq_len, 32)
        validity = torch.ones(1, seq_len, dtype=torch.bool)
        validity[0, seq_len - 8 :] = False  # right padding
        cos, sin = _build_rope(seq_len, module.head_dim, hidden_full.device)

        reference_hidden = hidden_full.clone().requires_grad_(True)
        reference = _reference_forward(module, reference_hidden, cos, sin, validity)
        torch.manual_seed(seed + 2)
        grad_full = torch.randn_like(reference)
        reference.backward(grad_full)

        shard = seq_len // world
        local_hidden = hidden_full[:, rank * shard : (rank + 1) * shard].clone().requires_grad_(True)
        local_cos = cos[:, rank * shard : (rank + 1) * shard]
        local_sin = sin[:, rank * shard : (rank + 1) * shard]

        original_contexts = (
            qsa_mod.get_kv_allgather_context,
            qsa_mod.get_flattened_cp_context,
            qsa_mod.kv_allgather,
        )
        qsa_mod.get_kv_allgather_context = lambda: (kv_group, kv_rank, kv_size)
        qsa_mod.get_flattened_cp_context = lambda: (cp_group, rank, world)
        qsa_mod.kv_allgather = lambda x, group, gather_dim=2: _GlooAllGather.apply(x, gather_dim, group)
        try:
            forward = qsa_mod.get_qwen4_exp_ulysses_forward(_FakeQSAAttention.forward, ulysses_group)
            output, _ = forward(module, local_hidden, (local_cos, local_sin), validity, None)
        finally:
            (
                qsa_mod.get_kv_allgather_context,
                qsa_mod.get_flattened_cp_context,
                qsa_mod.kv_allgather,
            ) = original_contexts

        expected = reference[:, rank * shard : (rank + 1) * shard]
        assert torch.allclose(output, expected, atol=1e-4, rtol=1e-4), (
            f"forward mismatch (max diff {(output - expected).abs().max()})"
        )

        output.backward(grad_full[:, rank * shard : (rank + 1) * shard])
        expected_grad = reference_hidden.grad[:, rank * shard : (rank + 1) * shard]
        assert torch.allclose(local_hidden.grad, expected_grad, atol=1e-4, rtol=1e-4), (
            f"backward mismatch (max diff {(local_hidden.grad - expected_grad).abs().max()})"
        )
    finally:
        dist.destroy_process_group()


class Qwen4ExpQSACPForwardTests(unittest.TestCase):
    def _run(self, kv_size, ulysses_size, sparse, port):
        world = kv_size * ulysses_size
        mp.spawn(
            _qsa_worker,
            args=(world, port, kv_size, ulysses_size, sparse, 77 + world),
            nprocs=world,
            join=True,
        )

    def test_kv2_ulysses2_sparse(self):
        self._run(2, 2, True, 29621)

    def test_kv2_ulysses2_dense(self):
        self._run(2, 2, False, 29622)

    def test_kv4_ulysses1_sparse(self):
        self._run(4, 1, True, 29623)

    def test_kv1_ulysses4_sparse_with_head_padding(self):
        self._run(1, 4, True, 29624)

    def test_kv1_ulysses4_dense_with_head_padding(self):
        self._run(1, 4, False, 29625)


class QsaBlockKeysSelectionTests(unittest.TestCase):
    """Single-process check: pre-pooled block keys select the same tokens as raw keys."""

    def test_block_keys_match_raw_keys_selection(self):
        from fsdp_turbo.ops.qsa import (
            qsa_compress_block_keys,
            qsa_selected_token_mask,
        )

        torch.manual_seed(99)
        batch, seq_len, heads, head_dim, rate, budget = 1, 64, 4, 16, 4, 12
        index_queries = torch.randn(batch, seq_len, heads, head_dim)
        raw_keys = torch.randn(batch, seq_len, head_dim)
        cos, sin = _build_rope(seq_len, head_dim, raw_keys.device)
        norm_weight = torch.randn(head_dim)

        keys = torch.arange(seq_len)
        causal = keys[None, :] <= keys[:, None]
        validity = torch.ones(1, seq_len, dtype=torch.bool)
        validity[0, seq_len - 6 :] = False
        mask4d = causal[None, None] & validity[:, None, None, :].bool()

        raw_ids = qsa_selected_token_mask(
            index_queries,
            raw_keys,
            cos,
            sin,
            mask4d,
            norm_weight,
            rms_norm_eps=1e-6,
            token_budget=budget,
            compress_ratio=rate,
            return_token_ids=True,
        )
        block_keys = qsa_compress_block_keys(raw_keys, cos, sin, norm_weight, rms_norm_eps=1e-6, compress_ratio=rate)
        block_ids = qsa_selected_token_mask(
            index_queries,
            None,
            None,
            None,
            mask4d,
            None,
            rms_norm_eps=1e-6,
            token_budget=budget,
            compress_ratio=rate,
            block_keys=block_keys,
            return_token_ids=True,
        )

        def _row_sets(ids):
            return [set(row[row >= 0].tolist()) for row in ids[0]]

        raw_sets = _row_sets(raw_ids)
        block_sets = _row_sets(block_ids)
        assert len(raw_sets) == len(block_sets)
        for i, (a, b) in enumerate(zip(raw_sets, block_sets)):
            assert a == b, f"selection mismatch at query {i}: {a} != {b}"

    def test_backend_fallback_matches_public_entry(self):
        from fsdp_turbo.ops.qsa import qsa_compress_block_keys, qsa_selected_token_mask
        from fsdp_turbo.ops.triton.qsa_indexer import (
            qsa_compress_block_keys_triton,
            qsa_selected_token_mask_triton,
        )

        torch.manual_seed(47)
        batch, length, heads, dim, rate = 2, 32, 2, 16, 4
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                queries = torch.randn(batch, length, heads, dim).to(dtype)
                raw = torch.randn(batch, length, dim).to(dtype)
                cos, sin = _build_rope(length, dim, raw.device)
                weight = torch.randn(dim).to(dtype)
                keys = qsa_compress_block_keys(raw, cos, sin, weight, rms_norm_eps=1e-6, compress_ratio=rate)
                fallback_keys = qsa_compress_block_keys_triton(
                    raw,
                    cos,
                    sin,
                    weight,
                    rms_norm_eps=1e-6,
                    compress_ratio=rate,
                )
                torch.testing.assert_close(keys, fallback_keys, rtol=0, atol=0)
                causal = torch.ones(length, length, dtype=torch.bool).tril()
                boolean = causal[None, None].expand(batch, 1, length, length)
                additive = torch.where(boolean, 0.0, float("-inf"))
                for mask in (boolean, additive):
                    for return_ids in (False, True):
                        for compressed in (False, True):
                            args = (
                                queries,
                                None if compressed else raw,
                                None if compressed else cos,
                                None if compressed else sin,
                                mask,
                                None if compressed else weight,
                            )
                            kwargs = dict(
                                rms_norm_eps=1e-6,
                                token_budget=8,
                                compress_ratio=rate,
                                block_keys=keys if compressed else None,
                                return_token_ids=return_ids,
                            )
                            actual = qsa_selected_token_mask(*args, **kwargs)
                            expected = qsa_selected_token_mask_triton(*args, **kwargs)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

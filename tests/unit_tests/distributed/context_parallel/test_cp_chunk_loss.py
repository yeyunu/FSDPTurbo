# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for the CP chunk loss and the flash-causal attention path."""

import os
import socket
import types
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
import torch.nn.functional as F


def _find_free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_gloo(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _shutdown_gloo():
    dist.destroy_process_group()


class _FlashModule:
    """Minimal stand-in for KimiMLAAttention on the flash path."""

    is_causal = True
    training = False

    def __init__(self):
        self.config = types.SimpleNamespace()
        self.num_key_value_groups = 1


def _reference_causal_attention(query, key, value, scaling):
    """Dense causal attention over [batch, heads, seq, head_dim] tensors."""
    scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
    seq_len = query.shape[-2]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(probs, value)


def _dense_lm_reference(labels, hidden, weight, ignore_index):
    """Full-sequence shifted cross entropy; returns (loss, hidden_grad, weight_grad)."""
    hidden = hidden.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    shifted = torch.full_like(labels, ignore_index)
    shifted[..., :-1] = labels[..., 1:]
    logits = F.linear(hidden, weight).float()
    per_token = F.cross_entropy(
        logits.view(-1, weight.shape[0]), shifted.reshape(-1), ignore_index=ignore_index, reduction="none"
    )
    valid = shifted.ne(ignore_index)
    loss = (per_token * valid.reshape(-1)).sum() / valid.sum().clamp_min(1)
    loss.backward()
    return loss.detach(), hidden.grad, weight.grad


def _chunk_loss_worker(rank, world_size, port, seq_len, chunk_size, check_num_items=False):
    _init_gloo(rank, world_size, port)
    try:
        from fsdp_turbo.distributed.context_parallel.batch import pad_tensor_to_group, split_tensor_to_group
        from fsdp_turbo.distributed.context_parallel.loss import cp_chunk_lm_loss
        from fsdp_turbo.memory.chunk_loss.chunk_loss import chunk_lm_loss

        torch.manual_seed(1234)
        batch, hidden_dim, vocab, ignore_index = 1, 8, 17, -100
        group = dist.group.WORLD

        complete_hidden = torch.randn(batch, seq_len, hidden_dim)
        complete_labels = torch.randint(0, vocab, (batch, seq_len))
        complete_labels[..., -1] = ignore_index  # the last position predicts nothing
        if seq_len >= 6:
            complete_labels[..., 3] = ignore_index
        head_weight = torch.randn(vocab, hidden_dim)

        # Reference: dense loss over the complete (unpadded) sequence.
        ref_loss, ref_hidden_grad, ref_weight_grad = _dense_lm_reference(
            complete_labels, complete_hidden, head_weight, ignore_index
        )

        # Local shard exactly the way shard_fused_sequence_inputs produces it.
        padded_hidden = pad_tensor_to_group(complete_hidden, group, dim=1, pad_value=0)
        local_hidden = split_tensor_to_group(padded_hidden, group, dim=1)
        local_len = local_hidden.size(1)

        # 1) Chunked CP loss against the dense reference.
        chunk_hidden = local_hidden.clone().requires_grad_(True)
        chunk_weight = head_weight.clone().requires_grad_(True)
        chunk_loss = cp_chunk_lm_loss(
            chunk_hidden,
            complete_labels,
            chunk_weight,
            group,
            ignore_index=ignore_index,
            chunk_size=chunk_size,
            chunk_loss_fn=chunk_lm_loss,
        )
        torch.testing.assert_close(chunk_loss.detach(), ref_loss.detach(), rtol=1e-5, atol=1e-5)

        chunk_loss.backward()
        # The backward multiplies by world_size to pre-compensate FSDP's
        # gradient averaging across CP ranks; divide it back out for the
        # comparison. Padded tail positions carry zero reference gradient.
        expected_grad = F.pad(ref_hidden_grad, (0, 0, 0, padded_hidden.size(1) - seq_len))
        torch.testing.assert_close(
            chunk_hidden.grad / world_size,
            expected_grad[..., rank * local_len : (rank + 1) * local_len, :],
            rtol=1e-4,
            atol=1e-5,
        )
        # lm_head gradients accumulate only the local tokens; summing the
        # per-rank contributions recovers the dense reference gradient, which
        # is exactly what FSDP's reduction over the CP group does.
        local_weight_grad = chunk_weight.grad / world_size
        dist.all_reduce(local_weight_grad, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(local_weight_grad, ref_weight_grad, rtol=1e-4, atol=1e-5)

        if check_num_items:
            # A caller-provided global item count skips the cross-rank item
            # reduction; the int and tensor forms must agree exactly.
            def run_loss(num_items):
                return cp_chunk_lm_loss(
                    local_hidden.clone().requires_grad_(True),
                    complete_labels,
                    head_weight.clone().requires_grad_(True),
                    group,
                    num_items_in_batch=num_items,
                    ignore_index=ignore_index,
                    chunk_size=chunk_size,
                    chunk_loss_fn=chunk_lm_loss,
                )

            torch.testing.assert_close(run_loss(3), run_loss(torch.tensor(3)), rtol=0, atol=0)
    finally:
        _shutdown_gloo()


def _fsdp_sharded_head_worker(rank, world_size, port, seq_len, chunk_size):
    """Chunk loss through an FSDP2-sharded head must reduce the head gradient."""
    _init_gloo(rank, world_size, port)
    try:
        import functools

        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.tensor import DTensor

        from fsdp_turbo.distributed.context_parallel.batch import pad_tensor_to_group, split_tensor_to_group
        from fsdp_turbo.memory.chunk_loss.chunkloss_lm_head import apply_chunkloss_module
        from fsdp_turbo.models.kimi.kimi_k3_cp import _cp_chunk_loss_fn

        torch.manual_seed(2024)
        batch, hidden_dim, vocab, ignore_index = 1, 8, 16, -100
        group = dist.group.WORLD
        complete_hidden = torch.randn(batch, seq_len, hidden_dim)
        complete_labels = torch.randint(0, vocab, (batch, seq_len))
        complete_labels[..., -1] = ignore_index
        if seq_len >= 6:
            complete_labels[..., 3] = ignore_index

        # Dense reference: full-sequence mean cross entropy and head gradient.
        torch.manual_seed(77)
        ref_weight = torch.randn(vocab, hidden_dim)
        ref_loss, _, ref_grad = _dense_lm_reference(complete_labels, complete_hidden, ref_weight, ignore_index)

        padded_hidden = pad_tensor_to_group(complete_hidden, group, dim=1, pad_value=0)
        local_hidden = split_tensor_to_group(padded_hidden, group, dim=1)

        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        head = torch.nn.Linear(hidden_dim, vocab, bias=False)
        with torch.no_grad():
            head.weight.copy_(ref_weight)
        # Same pattern as the Kimi example: install the chunk-loss forward
        # before FSDP wraps the head, then call the module.
        apply_chunkloss_module(head)
        fully_shard(head, mesh=mesh)

        loss_func = functools.partial(
            _cp_chunk_loss_fn,
            group=group,
            num_items_in_batch=None,
            ignore_index=ignore_index,
            chunk_size=chunk_size,
        )
        loss = head(local_hidden, loss_func, complete_labels)
        torch.testing.assert_close(loss.detach(), ref_loss.detach(), rtol=1e-5, atol=1e-5)
        loss.backward()

        grad = head.weight.grad
        assert isinstance(grad, DTensor), f"expected a sharded DTensor grad, got {type(grad)}"
        local_grad = grad.to_local()
        shard = vocab // world_size
        expected = ref_grad[rank * shard : (rank + 1) * shard]
        torch.testing.assert_close(local_grad, expected, rtol=1e-4, atol=1e-5)
    finally:
        _shutdown_gloo()


def _ulysses_roundtrip_worker(rank, world_size, port):
    """Roundtrip the flash kernel through the Ulysses wrapper."""
    _init_gloo(rank, world_size, port)
    try:
        from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_attention import get_ulysses_attention_fn
        from fsdp_turbo.models.kimi.kimi_flash_attention import kimi_flash_attention_forward

        torch.manual_seed(99)
        batch, heads, local_len, head_dim = 1, 2, 4, 3
        scaling = head_dim**-0.5
        group = dist.group.WORLD

        full_q = torch.randn(batch, heads, local_len * world_size, head_dim)
        full_k = torch.randn(batch, heads, local_len * world_size, head_dim)
        full_v = torch.randn(batch, heads, local_len * world_size, head_dim)

        def local_shard(tensor):
            return tensor[..., rank * local_len : (rank + 1) * local_len, :].contiguous()

        # Reference: dense causal attention over the complete sequence. The
        # wrapper must return every head for this rank's sequence shard (heads
        # are only sharded during the attention itself).
        ref = _reference_causal_attention(full_q, full_k, full_v, scaling)  # [B, H, S, D]
        expected = ref.transpose(1, 2)[:, rank * local_len : (rank + 1) * local_len].contiguous()

        # The kernel carries native causality.
        flasher = get_ulysses_attention_fn(kimi_flash_attention_forward, group)
        output, weights = flasher(
            _FlashModule(),
            local_shard(full_q),
            local_shard(full_k),
            local_shard(full_v),
            attention_mask=None,
            scaling=scaling,
        )
        assert weights is None
        assert output.shape == (batch, local_len, heads, head_dim), tuple(output.shape)
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
    finally:
        _shutdown_gloo()


class CpChunkLossTests(unittest.TestCase):
    def _spawn(self, worker, world_size, **kwargs):
        port = _find_free_port()
        torch_mp.spawn(worker, args=(world_size, port, *kwargs.values()), nprocs=world_size, join=True)

    def test_chunk_loss_matches_dense_reference(self):
        # seq_len=9 forces right padding to 10 before the CP split.
        for seq_len in (10, 9):
            with self.subTest(seq_len=seq_len):
                self._spawn(_chunk_loss_worker, world_size=2, seq_len=seq_len, chunk_size=4)

    def test_chunk_loss_accepts_integer_num_items_in_batch(self):
        self._spawn(_chunk_loss_worker, world_size=2, seq_len=10, chunk_size=4, check_num_items=True)

    def test_fsdp_sharded_head_gradient_matches_dense_reference(self):
        # Regression for the reviewer's FSDP2 finding: the chunked loss must
        # run through the FSDP-managed head call so its gradient is reduced.
        self._spawn(_fsdp_sharded_head_worker, world_size=2, seq_len=10, chunk_size=4)

    def test_attention_roundtrip_through_ulysses(self):
        self._spawn(_ulysses_roundtrip_worker, world_size=2)


class KimiFlashAttentionTests(unittest.TestCase):
    def _run_flash(self, module, q, k, v, attention_mask=None, **kwargs):
        from fsdp_turbo.models.kimi.kimi_flash_attention import kimi_flash_attention_forward

        output, weights = kimi_flash_attention_forward(
            module, q, k, v, attention_mask=attention_mask, scaling=q.shape[-1] ** -0.5, **kwargs
        )
        return output, weights

    def test_flash_rejects_dense_mask(self):
        torch.manual_seed(13)
        module = _FlashModule()
        q, k, v = (torch.randn(1, 2, 5, 4, dtype=torch.float64) for _ in range(3))
        dense_mask = torch.zeros(1, 1, 5, 5, dtype=torch.float64)

        with self.assertRaisesRegex(ValueError, "dense 4-D mask"):
            self._run_flash(module, q, k, v, attention_mask=dense_mask)

    def test_flash_expands_gqa_key_value_heads(self):
        torch.manual_seed(14)
        module = _FlashModule()
        q = torch.randn(1, 4, 6, 3, dtype=torch.float64)
        kv_k = torch.randn(1, 2, 6, 3, dtype=torch.float64)
        kv_v = torch.randn(1, 2, 6, 3, dtype=torch.float64)
        k = kv_k.repeat_interleave(2, dim=1)
        v = kv_v.repeat_interleave(2, dim=1)

        output, _ = self._run_flash(module, q, kv_k, kv_v)
        self.assertEqual(output.shape, (1, 6, 4, 3))
        scores = (q @ k.transpose(-1, -2)) * (3**-0.5)
        causal = torch.ones(6, 6, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        expected_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        expected = (expected_probs @ v).transpose(1, 2)
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


class FlashCPWiringTests(unittest.TestCase):
    """The flash kernel must sit behind a CP wrapper whenever CP is active."""

    def test_unwrapped_call_without_wrapper_names_the_config(self):
        import fsdp_turbo.models.kimi.kimi_flash_attention as kfa

        module = _FlashModule()
        q, k, v = (torch.randn(1, 2, 4, 3) for _ in range(3))
        with mock.patch.object(kfa, "_active_context_parallel_size", return_value=2):
            with self.assertRaises(ValueError) as ctx:
                kfa.kimi_flash_attention_forward(module, q, k, v)
        message = str(ctx.exception)
        self.assertIn("cp_plan", message)
        self.assertIn("full_attention", message)


class ChunkLossConfigTests(unittest.TestCase):
    def test_chunk_loss_is_allowed_with_context_parallel(self):
        from fsdp_turbo.fsdp_turbo_config import CPPlanConfig, DistributedConfig, FSDPTurboConfig, MemoryConfig

        patches = {"target_functions": "package.module.attention", "type": "full_attention"}
        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                ulysses_parallel_size=2, cp_plan=CPPlanConfig(ulysses_function_patches=patches)
            ),
            memory=MemoryConfig(chunk_loss=True, chunk_loss_size=1024),
        )
        self.assertTrue(config.memory.chunk_loss)
        self.assertEqual(config.memory.chunk_loss_size, 1024)

    def test_chunk_loss_size_must_be_positive(self):
        from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, MemoryConfig

        with self.assertRaisesRegex(ValueError, "chunk_loss_size"):
            FSDPTurboConfig(memory=MemoryConfig(chunk_loss=True, chunk_loss_size=0))


class ChunkBatchGuardTests(unittest.TestCase):
    def test_chunk_batch_context_parallel_guard(self):
        from fsdp_turbo.fsdp_turbo_config import (
            ChunkBatchPlanConfig,
            CPPlanConfig,
            DistributedConfig,
            FSDPTurboConfig,
            MemoryConfig,
        )

        patches = {
            "ulysses_function_patches": {
                "target_functions": "package.module.attention",
                "type": "full_attention",
            }
        }
        with self.assertRaisesRegex(ValueError, "chunk_batch"):
            FSDPTurboConfig(
                distributed=DistributedConfig(ulysses_parallel_size=2, cp_plan=CPPlanConfig(**patches)),
                memory=MemoryConfig(chunk_batch=True),
            )

        config = FSDPTurboConfig(
            memory=MemoryConfig(
                chunk_batch=True,
                chunk_batch_plan=ChunkBatchPlanConfig(
                    chunk_mbs=2,
                    apply_modules=["model.layers.{*}"],
                    chunk_arg_indexes=[0],
                ),
            )
        )
        self.assertTrue(config.memory.chunk_batch)


if __name__ == "__main__":
    unittest.main()

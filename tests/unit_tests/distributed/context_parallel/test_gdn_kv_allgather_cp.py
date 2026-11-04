# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU/gloo tests for the GDN KV-AllGather context-parallel path.

The FLA chunk kernels need an accelerator, so the scan entry is replaced by a
distributed reference built on FLA's own ``naive_recurrent_gated_delta_rule``
(gather segments, run the full-sequence recurrence, slice back).  This
validates everything around the scan — Ulysses head all-to-all, conv-tail
exchange, conv/decay parameter slicing, GQA-free reshapes, the inverse
all-to-all and the gated norm — against a single-process reference.
``build_cp_context`` itself is pure PyTorch and is tested unmocked.
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F

import fsdp_turbo.modules.gated_delta_net as gdn_mod


def _init_gloo(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)


def _cpu_causal_conv1d(x, weight, bias=None, activation=None, seq_idx=None):
    """x [B, C, T], weight [C, W]: causal depthwise conv with zero left padding."""
    kernel = weight.shape[-1]
    padded = F.pad(x, (kernel - 1, 0))
    out = F.conv1d(padded, weight.unsqueeze(1), bias=bias, groups=x.shape[1])
    if activation == "silu":
        out = F.silu(out)
    return out


class _FakeGatedNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, z):
        dtype = x.dtype
        x = x.float()
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1 + self.weight.float())
        return (out * torch.sigmoid(z.float())).to(dtype)


class _FakeGDN(nn.Module):
    def __init__(self, hidden, k_heads, v_heads, head_k, head_v, kernel):
        super().__init__()
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.head_k_dim = head_k
        self.head_v_dim = head_v
        self.key_dim = k_heads * head_k
        self.value_dim = v_heads * head_v
        self.conv_kernel_size = kernel
        self.activation = "silu"
        self.layer_idx = 0
        conv_dim = 2 * self.key_dim + self.value_dim
        self.in_proj_qkv = nn.Linear(hidden, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, v_heads, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel, groups=conv_dim, padding=kernel - 1)
        self.A_log = nn.Parameter(torch.randn(v_heads))
        self.dt_bias = nn.Parameter(torch.randn(v_heads))
        self.norm = _FakeGatedNorm(head_v)
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)


def _naive_scan_inputs(module, mixed_qkv, b, a):
    """Reshape post-conv projections into kernel-layout tensors (full sequence)."""
    batch = mixed_qkv.shape[0]
    length = mixed_qkv.shape[1]
    q, k, v = torch.split(mixed_qkv, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    q = q.view(batch, length, module.num_k_heads, module.head_k_dim)
    k = k.view(batch, length, module.num_k_heads, module.head_k_dim)
    v = v.view(batch, length, module.num_v_heads, module.head_v_dim)
    beta = b.sigmoid()
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    return q, k, v, g, beta


def _naive_gdn(module, q, k, v, g, beta):
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

    # use_qk_l2norm_in_kernel=True: the kernel normalizes q/k internally.
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    out, _ = naive_recurrent_gated_delta_rule(q, k, v, beta=beta, g=g)
    return out


def _reference_gdn_forward(module, hidden, validity):
    hidden = hidden * validity[:, :, None].to(hidden.dtype)
    batch, length, _ = hidden.shape
    mixed = module.in_proj_qkv(hidden)
    z = module.in_proj_z(hidden)
    b = module.in_proj_b(hidden)
    a = module.in_proj_a(hidden)
    mixed = _cpu_causal_conv1d(
        mixed.transpose(1, 2),
        module.conv1d.weight.squeeze(1),
        module.conv1d.bias,
        activation=module.activation,
    ).transpose(1, 2)
    q, k, v, g, beta = _naive_scan_inputs(module, mixed, b, a)
    out = _naive_gdn(module, q, k, v, g, beta)
    out = module.norm(out.reshape(-1, module.head_v_dim), z.reshape(-1, module.head_v_dim))
    return module.out_proj(out.reshape(batch, length, -1))


class _GatherSegments(torch.autograd.Function):
    """Differentiable segment gather (backward: all_reduce SUM then slice)."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        shards = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
        dist.all_gather(shards, x.contiguous(), group=group)
        return torch.cat(shards, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        total = grad_output.contiguous().clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=ctx.group)
        world = dist.get_world_size(ctx.group)
        rank = dist.get_rank(ctx.group)
        seg = total.shape[1] // world
        return total[:, rank * seg : (rank + 1) * seg], None


def _mock_distributed_scan(kv_group, kv_size):
    """Reference CP scan: gather segments, run the full recurrence, slice back."""

    def scan(query, key, value, g, beta, kv_group=kv_group, total_seq_len=None):
        gathered = [_GatherSegments.apply(tensor, kv_group) for tensor in (query, key, value, g, beta)]
        out = _naive_gdn(None, *gathered)
        seg_len = query.shape[1]
        kv_rank = dist.get_rank(kv_group)
        return out[:, kv_rank * seg_len : (kv_rank + 1) * seg_len]

    return scan


def _cpu_chunk_gdn(
    query,
    key,
    value,
    g=None,
    beta=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
    out, final = naive_recurrent_gated_delta_rule(
        query,
        key,
        value,
        beta=beta,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    return out, final


def _mock_resolve_gdn_func(self, name, module_name=None):
    return {"causal_conv1d_fn": _cpu_causal_conv1d, "chunk_gated_delta_rule": _cpu_chunk_gdn}.get(name)


def _gdn_worker(rank, world, port, kv_size, ulysses_size, seed):
    _init_gloo(rank, world, port)
    ulysses_rank = rank % ulysses_size
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

        torch.manual_seed(seed)
        module = _FakeGDN(hidden=24, k_heads=4, v_heads=4, head_k=8, head_v=8, kernel=4)
        seq_len = 64
        torch.manual_seed(seed + 1)
        hidden_full = torch.randn(1, seq_len, 24)
        validity = torch.ones(1, seq_len, dtype=torch.bool)
        validity[0, seq_len - 6 :] = False

        reference_hidden = hidden_full.clone().requires_grad_(True)
        reference = _reference_gdn_forward(module, reference_hidden, validity)
        torch.manual_seed(seed + 2)
        grad_full = torch.randn_like(reference)
        reference.backward(grad_full)

        shard = seq_len // world
        local_hidden = (
            (
                hidden_full[:, rank * shard : (rank + 1) * shard]
                * validity[:, rank * shard : (rank + 1) * shard, None].to(hidden_full.dtype)
            )
            .clone()
            .requires_grad_(True)
        )
        # The CP forward applies the padding mask itself; feed unmasked shards.
        local_hidden = hidden_full[:, rank * shard : (rank + 1) * shard].clone().requires_grad_(True)
        local_validity = validity[:, rank * shard : (rank + 1) * shard]

        originals = (
            gdn_mod.get_ulysses_context,
            gdn_mod.get_kv_allgather_context,
            gdn_mod.gdn_kv_allgather_scan,
            gdn_mod._resolve_gdn_func,
        )
        gdn_mod.get_ulysses_context = lambda: (ulysses_group, ulysses_rank, ulysses_size)
        gdn_mod.get_kv_allgather_context = lambda: (kv_group, kv_rank, kv_size)
        gdn_mod.gdn_kv_allgather_scan = _mock_distributed_scan(kv_group, kv_size)
        gdn_mod._resolve_gdn_func = _mock_resolve_gdn_func
        try:
            output = gdn_mod.GatedDeltaNet.forward(module, local_hidden, None, local_validity)
        finally:
            (
                gdn_mod.get_ulysses_context,
                gdn_mod.get_kv_allgather_context,
                gdn_mod.gdn_kv_allgather_scan,
                gdn_mod._resolve_gdn_func,
            ) = originals

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


def _cp_context_worker(rank, world, port):
    _init_gloo(rank, world, port)
    try:
        from fla.ops.cp import build_cp_context

        cu_seqlens = torch.tensor([0, world * 16], dtype=torch.long)
        context = build_cp_context(cu_seqlens, group=dist.group.WORLD)
        assert context.cu_seqlens.tolist() == [0, 16], context.cu_seqlens
        assert context.num_seqs == 1
        assert context.pre_num_ranks == rank
        assert context.is_first_rank == (rank == 0)
        assert context.post_num_ranks == world - 1 - rank
        assert context.is_last_rank == (rank == world - 1)
    finally:
        dist.destroy_process_group()


class GDNKVAllGatherCPTests(unittest.TestCase):
    def test_gdn_forward_and_backward_kv2_ulysses2(self):
        mp.spawn(_gdn_worker, args=(4, 29631, 2, 2, 2024), nprocs=4, join=True)

    def test_gdn_forward_kv4_ulysses1(self):
        mp.spawn(_gdn_worker, args=(4, 29632, 4, 1, 2025), nprocs=4, join=True)

    def test_gdn_forward_kv1_ulysses4(self):
        mp.spawn(_gdn_worker, args=(4, 29633, 1, 4, 2026), nprocs=4, join=True)

    def test_fla_build_cp_context_metadata(self):
        mp.spawn(_cp_context_worker, args=(4, 29634), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()

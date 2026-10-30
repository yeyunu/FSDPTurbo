# 2025 The Qwen team Alibaba Group and the HuggingFace Inc. team. All rights reserved
"""Portable gated delta rule with autograd and explicit sequence state."""

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.registry import register_op


def _l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _chunk_gdn(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    scale=None,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _recurrent_gdn(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
    scale=None,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


@register_op("chunk_gated_delta_rule", "cpu")
def chunk_gdn_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    head_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute chunk GDN in BTHD layout, restarting state at packed boundaries."""
    if head_first:
        q, k, v, g, beta = (value.transpose(1, 2) for value in (q, k, v, g, beta))
    if cu_seqlens is None:
        output, state = _chunk_gdn(
            q,
            k,
            v,
            g,
            beta,
            chunk_size,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
            scale=scale,
        )
    else:
        if q.shape[0] != 1:
            raise ValueError("Packed GDN requires batch size one.")
        offsets = cu_seqlens.tolist()
        outputs, states = [], []
        for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            sequence_state = None if initial_state is None else initial_state[index : index + 1]
            values = [value[:, start:end] for value in (q, k, v, g, beta)]
            output, state = _chunk_gdn(
                *values,
                chunk_size,
                sequence_state,
                output_final_state,
                use_qk_l2norm_in_kernel,
                scale=scale,
            )
            outputs.append(output)
            if output_final_state:
                states.append(state)
        output = torch.cat(outputs, dim=1)
        state = torch.cat(states, dim=0) if output_final_state else None
    return (output.transpose(1, 2) if head_first else output), state


@register_op("fused_recurrent_gated_delta_rule", "cpu")
def recurrent_gdn_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute recurrent GDN with packed boundaries and differentiable states.

    Inputs use BTHD layout unless ``head_first`` selects BHTD. Packed inputs
    require batch size one; ``cu_seqlens`` identifies independent sequences,
    each with its own initial and final state.
    """
    if head_first:
        q, k, v, g, beta = (value.transpose(1, 2) for value in (q, k, v, g, beta))
    if cu_seqlens is None:
        output, state = _recurrent_gdn(
            q, k, v, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, scale=scale
        )
    else:
        if q.shape[0] != 1:
            raise ValueError("Packed GDN requires batch size one.")
        offsets = cu_seqlens.tolist()
        outputs, states = [], []
        for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            sequence_state = None if initial_state is None else initial_state[index : index + 1]
            values = [value[:, start:end] for value in (q, k, v, g, beta)]
            output, state = _recurrent_gdn(
                *values, sequence_state, output_final_state, use_qk_l2norm_in_kernel, scale=scale
            )
            outputs.append(output)
            if output_final_state:
                states.append(state)
        output = torch.cat(outputs, dim=1)
        state = torch.cat(states, dim=0) if output_final_state else None
    return (output.transpose(1, 2) if head_first else output), state

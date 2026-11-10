# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Legacy one-dimensional KDA state CP; two-dimensional CP uses P2P."""

from collections.abc import Callable
from functools import wraps

import torch
import torch.distributed as dist

from fsdp_turbo.ops.kcp import build_kcp_context, kcp_causal_conv1d, kcp_chunk_kda


def _apply_mask(tensor: torch.Tensor, local_mask: torch.Tensor | None) -> torch.Tensor:
    if local_mask is None:
        return tensor
    return tensor * local_mask[..., None].to(device=tensor.device, dtype=tensor.dtype)


def _sliced_cp_convolution(conv, projected, context, head_start, head_end, head_dim):
    start, end = head_start * head_dim, head_end * head_dim
    return kcp_causal_conv1d(
        projected,
        conv.weight[start:end].squeeze(1),
        None if conv.bias is None else conv.bias[start:end],
        conv.activation,
        context,
    )


def get_fla_state_kda_forward(
    original_forward: Callable[..., torch.Tensor],
    group: dist.ProcessGroup,
    *,
    ulysses_group: dist.ProcessGroup | None = None,
    kda_backend: str = "kcp",
) -> Callable[..., torch.Tensor]:
    """Build legacy KCP for contiguous one-dimensional sequence shards.

    The Ulysses keyword is retained for existing callers with a singleton group;
    a larger U group is rejected. New two-dimensional CP uses the P2P adapter.
    Batch size one, equal sequence shards and training without cache are required.
    """

    @wraps(original_forward)
    def forward(
        module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params=None,
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3 or hidden_states.size(0) != 1:
            raise ValueError("KDA state CP requires hidden_states with shape [1, local_sequence, hidden].")
        if cache_params is not None:
            raise ValueError("KDA state CP does not support cache inputs.")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("KDA state CP does not support pre-packed cu_seqlens inputs.")
        kernel_size = module.q_conv1d.weight.size(-1)
        if kernel_size <= 1:
            raise ValueError("KDA state CP convolution kernel size must be greater than 1.")
        if attention_mask is not None and attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("KDA state CP attention_mask must match the local [batch, sequence] shape.")

        state_size = dist.get_world_size(group)
        ulysses_size = 1 if ulysses_group is None else dist.get_world_size(ulysses_group)
        if ulysses_size > 1:
            raise ValueError("2D Ulysses-KCP AllGather is retired; use hybrid_p2p.")
        local_heads = module.num_heads
        head_start, head_end = 0, local_heads
        batch_size, local_length, _ = hidden_states.shape
        hidden_states = _apply_mask(hidden_states, attention_mask)

        query = module.q_proj(hidden_states)
        key = module.k_proj(hidden_states)
        value = module.v_proj(hidden_states)
        decay = module.f_b_proj(module.f_a_proj(hidden_states))
        beta = module.b_proj(hidden_states).float().sigmoid()
        state_length = local_length
        global_cu_seqlens = torch.tensor([0, state_length * state_size], dtype=torch.long, device=hidden_states.device)
        context = build_kcp_context(global_cu_seqlens, group, kernel_size)

        query = _sliced_cp_convolution(module.q_conv1d, query, context, head_start, head_end, module.head_k_dim)
        key = _sliced_cp_convolution(module.k_conv1d, key, context, head_start, head_end, module.head_k_dim)
        value = _sliced_cp_convolution(module.v_conv1d, value, context, head_start, head_end, module.head_dim)
        output = kcp_chunk_kda(
            q=query.view(batch_size, state_length, local_heads, module.head_k_dim),
            k=key.view(batch_size, state_length, local_heads, module.head_k_dim),
            v=value.view(batch_size, state_length, local_heads, module.head_dim),
            g=decay.view(batch_size, state_length, local_heads, module.head_k_dim),
            beta=beta.view(batch_size, state_length, local_heads),
            A_log=module.A_log[head_start:head_end],
            dt_bias=module.dt_bias.reshape(module.num_heads, -1)[head_start:head_end].reshape(-1),
            lower_bound=module.gate_lower_bound,
            disable_recompute=getattr(module.config, "skip_kda_recompute", False),
            context=context,
            backend=kda_backend,
        )
        output = output.reshape(batch_size, state_length, local_heads * module.head_dim)
        output = output.reshape(batch_size, local_length, module.num_heads, module.head_dim)

        if module.use_full_rank_gate:
            output_gate = module.g_proj(hidden_states)
        else:
            output_gate = module.g_b_proj(module.g_a_proj(hidden_states))
        output_gate = output_gate.view(batch_size, local_length, module.num_heads, module.head_dim)
        output = module.o_norm(output, output_gate)
        output = module.o_proj(output.reshape(batch_size, local_length, -1))
        return _apply_mask(output, attention_mask)

    return forward

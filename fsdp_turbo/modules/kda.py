# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Ulysses context parallelism for Kimi Delta Attention (KDA)."""

from functools import wraps
import importlib

import torch
import torch.distributed as dist

from fsdp_turbo.distributed.context_parallel.batch import split_tensor_to_group
from fsdp_turbo.distributed.context_parallel.ulysses.utils import ulysses_all_to_all


def _local_attention_mask(attention_mask, hidden_states, group):
    """Return the rank-local two-dimensional padding mask when one is present."""
    if attention_mask is None:
        return None
    if attention_mask.dim() != 2:
        return None
    local_seq_len = hidden_states.size(1)
    if attention_mask.size(1) == local_seq_len:
        return attention_mask
    if attention_mask.size(1) == local_seq_len * dist.get_world_size(group):
        return split_tensor_to_group(attention_mask, group, dim=1)
    raise ValueError(
        "KDA Ulysses attention_mask must contain either the local or complete sequence; "
        f"got mask={tuple(attention_mask.shape)}, hidden_states={tuple(hidden_states.shape)}."
    )


def _sequence_to_local_channels(tensor, group):
    """Exchange local sequence shards for complete-sequence channel shards."""
    return ulysses_all_to_all(tensor, group, scatter_dim=2, gather_dim=1)


def _local_channels_to_sequence(tensor, group):
    """Restore local sequence shards from complete-sequence channel shards."""
    return ulysses_all_to_all(tensor, group, scatter_dim=1, gather_dim=2)


def _sliced_short_convolution(conv, projected, rank):
    """Run ShortConvolution with the local contiguous channel weights.

    Ulysses partitions projection channels before convolution so each rank owns
    complete sequences for a subset of heads.  ShortConvolution is depthwise,
    so its channel-indexed weights can be sliced while still reusing the
    module's original forward path.  This keeps the causal-convolution backend,
    activation, and autograd behavior identical to the non-CP path.
    """
    local_channels = projected.size(-1)
    start = rank * local_channels
    end = start + local_channels
    weight = conv.weight[start:end]
    if conv.bias is not None:
        raise ValueError("Ulysses sliced ShortConvolution requires bias=False.")

    output, _ = conv(
        x=projected,
        weight=weight,
        cache=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    return output


def _get_chunk_kda(module):
    modeling_module = importlib.import_module(type(module).__module__)
    chunk_kda = getattr(modeling_module, "chunk_kda", None)
    if not callable(chunk_kda):
        raise RuntimeError(
            f"KDA module {type(module).__module__}.{type(module).__name__} does not expose its chunk_kda kernel."
        )
    return chunk_kda


def get_ulysses_kda_forward(original_forward, group):
    """Build a KimiDeltaAttention forward using sequence/head all-to-all."""

    @wraps(original_forward)
    def forward(
        module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params=None,
        **kwargs,
    ):
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return original_forward(
                module,
                hidden_states,
                attention_mask=attention_mask,
                cache_params=cache_params,
                **kwargs,
            )
        if cache_params is not None:
            raise ValueError("Ulysses CP for Kimi KDA does not support cache inputs.")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("Ulysses CP for Kimi KDA does not support pre-packed cu_seqlens inputs.")
        if module.num_heads % world_size:
            raise ValueError(f"ulysses size ({world_size}) must divide KDA num_heads ({module.num_heads}).")

        local_mask = _local_attention_mask(attention_mask, hidden_states, group)
        if local_mask is not None:
            hidden_states = hidden_states * local_mask[..., None].to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        batch_size, local_seq_len, _ = hidden_states.shape
        rank = dist.get_rank(group)
        local_num_heads = module.num_heads // world_size

        query = _sequence_to_local_channels(module.q_proj(hidden_states), group)
        key = _sequence_to_local_channels(module.k_proj(hidden_states), group)
        value = _sequence_to_local_channels(module.v_proj(hidden_states), group)
        decay = _sequence_to_local_channels(module.f_b_proj(module.f_a_proj(hidden_states)), group)
        beta = _sequence_to_local_channels(module.b_proj(hidden_states), group)

        query = _sliced_short_convolution(module.q_conv1d, query, rank)
        key = _sliced_short_convolution(module.k_conv1d, key, rank)
        value = _sliced_short_convolution(module.v_conv1d, value, rank)

        full_seq_len = query.size(1)
        query = query.view(batch_size, full_seq_len, local_num_heads, module.head_k_dim)
        key = key.view(batch_size, full_seq_len, local_num_heads, module.head_k_dim)
        value = value.view(batch_size, full_seq_len, local_num_heads, module.head_dim)
        decay = decay.view(batch_size, full_seq_len, local_num_heads, module.head_dim)
        beta = beta.float().sigmoid()

        head_start = rank * local_num_heads
        head_end = head_start + local_num_heads
        a_log = module.A_log[head_start:head_end]
        dt_bias = module.dt_bias.view(module.num_heads, module.head_dim)[head_start:head_end].reshape(-1)

        output, _ = _get_chunk_kda(module)(
            q=query,
            k=key,
            v=value,
            g=decay,
            beta=beta,
            A_log=a_log,
            dt_bias=dt_bias,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            safe_gate=module.gate_lower_bound is not None,
            lower_bound=module.gate_lower_bound,
            transpose_state_layout=True,
            cu_seqlens=None,
            skip_recompute=getattr(module.config, "skip_kda_recompute", False),
        )

        output = output.reshape(batch_size, full_seq_len, -1)
        output = _local_channels_to_sequence(output, group)
        output = output.view(batch_size, local_seq_len, module.num_heads, module.head_dim)

        if module.use_full_rank_gate:
            output_gate = module.g_proj(hidden_states)
        else:
            output_gate = module.g_b_proj(module.g_a_proj(hidden_states))
        output_gate = output_gate.view(batch_size, local_seq_len, module.num_heads, module.head_dim)
        output = module.o_norm(output, output_gate)
        output = module.o_proj(output.reshape(batch_size, local_seq_len, -1))
        if local_mask is not None:
            output = output * local_mask[..., None].to(device=output.device, dtype=output.dtype)
        return output

    return forward

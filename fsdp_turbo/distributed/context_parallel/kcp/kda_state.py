# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""FLA state context parallelism for Kimi Delta Attention (KDA)."""

from functools import wraps

import torch
import torch.distributed as dist

from fsdp_turbo.modules.kcp import build_kcp_context, kcp_causal_conv1d, kcp_chunk_kda


def _apply_mask(
    tensor: torch.Tensor,
    local_mask: torch.Tensor | None,
) -> torch.Tensor:
    if local_mask is None:
        return tensor
    return tensor * local_mask[..., None].to(device=tensor.device, dtype=tensor.dtype)


def get_fla_state_kda_forward(original_forward, group):
    """Build a Kimi KDA forward that keeps full heads and shards sequence state."""

    @wraps(original_forward)
    def forward(
        module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params=None,
        **kwargs,
    ):
        kernel_size = module.q_conv1d.weight.size(-1)
        if hidden_states.dim() != 3:
            raise ValueError("KDA state CP requires hidden_states with shape [batch, local_sequence, hidden].")
        batch_size, local_sequence_length, _ = hidden_states.shape
        if batch_size != 1:
            raise ValueError(f"KDA state CP requires batch size 1; got batch size {batch_size}.")
        if kernel_size <= 1:
            raise ValueError(f"KDA state CP convolution kernel size must be greater than 1; got {kernel_size}.")
        if cache_params is not None:
            raise ValueError("KDA state CP does not support cache inputs.")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("KDA state CP does not support pre-packed cu_seqlens inputs.")

        world_size = dist.get_world_size(group)
        local_mask = attention_mask
        hidden_states = _apply_mask(hidden_states, local_mask)

        query = module.q_proj(hidden_states)
        key = module.k_proj(hidden_states)
        value = module.v_proj(hidden_states)
        decay = module.f_b_proj(module.f_a_proj(hidden_states))
        beta = module.b_proj(hidden_states).float().sigmoid()

        global_cu_seqlens = torch.tensor(
            [0, local_sequence_length * world_size],
            dtype=torch.long,
            device=hidden_states.device,
        )
        context = build_kcp_context(global_cu_seqlens, group, kernel_size)

        query = kcp_causal_conv1d(
            query,
            module.q_conv1d.weight.squeeze(1),
            module.q_conv1d.bias,
            module.q_conv1d.activation,
            context,
        )
        key = kcp_causal_conv1d(
            key,
            module.k_conv1d.weight.squeeze(1),
            module.k_conv1d.bias,
            module.k_conv1d.activation,
            context,
        )
        value = kcp_causal_conv1d(
            value,
            module.v_conv1d.weight.squeeze(1),
            module.v_conv1d.bias,
            module.v_conv1d.activation,
            context,
        )

        query = query.view(
            batch_size,
            local_sequence_length,
            module.num_heads,
            module.head_k_dim,
        )
        key = key.view(
            batch_size,
            local_sequence_length,
            module.num_heads,
            module.head_k_dim,
        )
        value = value.view(
            batch_size,
            local_sequence_length,
            module.num_heads,
            module.head_dim,
        )
        decay = decay.view(
            batch_size,
            local_sequence_length,
            module.num_heads,
            module.head_k_dim,
        )
        beta = beta.view(batch_size, local_sequence_length, module.num_heads)

        output = kcp_chunk_kda(
            q=query,
            k=key,
            v=value,
            g=decay,
            beta=beta,
            A_log=module.A_log,
            dt_bias=module.dt_bias,
            lower_bound=module.gate_lower_bound,
            disable_recompute=getattr(module.config, "skip_kda_recompute", False),
            context=context,
        )

        if module.use_full_rank_gate:
            output_gate = module.g_proj(hidden_states)
        else:
            output_gate = module.g_b_proj(module.g_a_proj(hidden_states))
        output_gate = output_gate.view(
            batch_size,
            local_sequence_length,
            module.num_heads,
            module.head_dim,
        )
        output = module.o_norm(output, output_gate)
        output = module.o_proj(output.reshape(batch_size, local_sequence_length, -1))
        return _apply_mask(output, local_mask)

    return forward

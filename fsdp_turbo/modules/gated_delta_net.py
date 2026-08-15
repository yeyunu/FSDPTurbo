# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from fsdp_turbo.distributed.context_parallel.batch import split_tensor_to_group
from fsdp_turbo.distributed.context_parallel.ulysses.utils import (
    get_ulysses_context,
    ulysses_all_to_all,
)


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]):
    """Zero padded hidden states when a two-dimensional mask is available."""
    if attention_mask is not None and attention_mask.dim() == 2:
        return hidden_states.mul(attention_mask[:, :, None].to(device=hidden_states.device, dtype=hidden_states.dtype))
    return hidden_states


def ulysses_gdn_all_to_all_heads(
    module,
    mixed_qkv,
    b,
    a,
    batch_size,
    seq_len,
    ulysses_group,
    ulysses_size,
):
    """Exchange GDN sequence shards for local Q/K/V head shards."""
    if module.num_k_heads % ulysses_size != 0 or module.num_v_heads % ulysses_size != 0:
        raise ValueError(
            f"ulysses size ({ulysses_size}) must divide num_k_heads ({module.num_k_heads}) "
            f"and num_v_heads ({module.num_v_heads})."
        )

    local_num_k_heads = module.num_k_heads // ulysses_size
    local_num_v_heads = module.num_v_heads // ulysses_size
    local_key_dim = module.head_k_dim * local_num_k_heads
    local_value_dim = module.head_v_dim * local_num_v_heads

    q_proj, k_proj, v_proj = torch.split(mixed_qkv, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    q_proj = ulysses_all_to_all(q_proj, ulysses_group, scatter_dim=2, gather_dim=1)
    k_proj = ulysses_all_to_all(k_proj, ulysses_group, scatter_dim=2, gather_dim=1)
    v_proj = ulysses_all_to_all(v_proj, ulysses_group, scatter_dim=2, gather_dim=1)

    b = b.reshape(batch_size, seq_len, module.num_v_heads)
    a = a.reshape(batch_size, seq_len, module.num_v_heads)
    b = ulysses_all_to_all(b, ulysses_group, scatter_dim=2, gather_dim=1)
    a = ulysses_all_to_all(a, ulysses_group, scatter_dim=2, gather_dim=1)

    metadata = {
        "local_num_k_heads": local_num_k_heads,
        "local_num_v_heads": local_num_v_heads,
        "local_key_dim": local_key_dim,
        "local_value_dim": local_value_dim,
    }
    return torch.cat((q_proj, k_proj, v_proj), dim=-1), b, a, metadata


def ulysses_gdn_slice_a_log_dt_bias(module, a, ulysses_rank, local_num_v_heads):
    """Select local recurrent-head parameters and construct the decay."""
    offset = ulysses_rank * local_num_v_heads
    head_slice = slice(offset, offset + local_num_v_heads)
    return -module.A_log[head_slice].float().exp() * F.softplus(a.float() + module.dt_bias[head_slice])


def ulysses_gdn_get_conv_params(module, ulysses_rank, local_key_dim, local_value_dim):
    """Return local Q/K/V slices of a fused depthwise convolution."""
    return (
        _slice_gdn_conv1d_param(
            module.conv1d.weight,
            module,
            ulysses_rank,
            local_key_dim,
            local_value_dim,
            "weight",
        ),
        _slice_gdn_conv1d_param(
            module.conv1d.bias,
            module,
            ulysses_rank,
            local_key_dim,
            local_value_dim,
            "bias",
        ),
    )


def _slice_gdn_conv1d_param(
    param,
    module,
    ulysses_rank,
    local_key_dim,
    local_value_dim,
    param_name,
):
    if param is None:
        return None
    expected_dim = module.key_dim * 2 + module.value_dim
    if param.shape[0] != expected_dim:
        raise ValueError(
            f"conv1d {param_name} dim ({param.shape[0]}) must match (2 * key_dim + value_dim) ({expected_dim})"
        )
    key_offset = ulysses_rank * local_key_dim
    value_offset = ulysses_rank * local_value_dim
    query = param[key_offset : key_offset + local_key_dim]
    key = param[module.key_dim + key_offset : module.key_dim + key_offset + local_key_dim]
    value = param[2 * module.key_dim + value_offset : 2 * module.key_dim + value_offset + local_value_dim]
    return torch.cat([query, key, value], dim=0)


def ulysses_gdn_all_to_all_output(core_attn_out, ulysses_group):
    """Restore local sequence ordering from local recurrent-head outputs."""
    core_attn_out = core_attn_out.transpose(1, 2).contiguous()
    core_attn_out = ulysses_all_to_all(core_attn_out, ulysses_group, scatter_dim=2, gather_dim=1)
    return core_attn_out.transpose(1, 2).contiguous()


def _resolve_gdn_func(self, func_name, module_func_name=None):
    """Resolve a GDN function from instance method (5.15.0) or module-level import (5.3.0).

    In transformers 5.3.0, GDN functions are instance methods (e.g. ``self.chunk_gated_delta_rule``).
    In transformers 5.15.0, they are module-level functions with a ``torch_`` prefix
    (e.g. ``torch_chunk_gated_delta_rule``). When ``module_func_name`` is not specified,
    it defaults to ``func_name`` (for functions whose name is unchanged across versions).
    """
    func = getattr(self, func_name, None)
    if func is not None:
        return func
    if module_func_name is None:
        module_func_name = func_name
    if "Moe" in type(self).__name__:
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling_module
    else:
        import transformers.models.qwen3_5.modeling_qwen3_5 as modeling_module
    return getattr(modeling_module, module_func_name, None)


class GatedDeltaNet(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        ulysses_group, ulysses_rank, ulysses_size = get_ulysses_context()

        if ulysses_size > 1 and cache_params is not None:
            label = "Qwen3.5 MoE" if "Moe" in type(self).__name__ else "Qwen3.5"
            raise ValueError(f"Ulysses CP for {label} GatedDeltaNet does not support cache yet.")

        if (
            ulysses_size > 1
            and attention_mask is not None
            and attention_mask.dim() == 2
            and attention_mask.size(1) == hidden_states.size(1) * ulysses_size
        ):
            attention_mask = split_tensor_to_group(
                attention_mask,
                ulysses_group,
                dim=1,
            )
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        conv_state = None
        recurrent_state = None
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if ulysses_size > 1:
            mixed_qkv, b, a, context = ulysses_gdn_all_to_all_heads(
                self,
                mixed_qkv,
                b,
                a,
                batch_size,
                seq_len,
                ulysses_group,
                ulysses_size,
            )
            local_num_k_heads = context["local_num_k_heads"]
            local_num_v_heads = context["local_num_v_heads"]
            local_key_dim = context["local_key_dim"]
            local_value_dim = context["local_value_dim"]
        else:
            local_num_k_heads = self.num_k_heads
            local_num_v_heads = self.num_v_heads
            local_key_dim = self.key_dim
            local_value_dim = self.value_dim

        local_conv_dim = local_key_dim * 2 + local_value_dim
        full_seq_len = seq_len * ulysses_size if ulysses_size > 1 else seq_len

        mixed_qkv = mixed_qkv.transpose(1, 2)
        if ulysses_size > 1:
            conv_weight, conv_bias = ulysses_gdn_get_conv_params(
                self,
                ulysses_rank,
                local_key_dim,
                local_value_dim,
            )
        else:
            conv_weight = self.conv1d.weight
            conv_bias = self.conv1d.bias

        if use_precomputed_states and seq_len == 1:
            causal_conv1d_update = _resolve_gdn_func(self, "causal_conv1d_update")
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weight.squeeze(1),
                conv_bias,
                self.activation,
            )
        else:
            if use_precomputed_states:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(new_conv_state, self.layer_idx)
            causal_conv1d_fn = _resolve_gdn_func(self, "causal_conv1d_fn")
            if causal_conv1d_fn is not None:
                mixed_qkv = causal_conv1d_fn(
                    mixed_qkv,
                    conv_weight.squeeze(1),
                    conv_bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(
                    F.conv1d(
                        mixed_qkv,
                        conv_weight,
                        bias=conv_bias,
                        padding=self.conv1d.padding[0],
                        groups=local_conv_dim,
                    )[:, :, : mixed_qkv.shape[-1]]
                )
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [local_key_dim, local_key_dim, local_value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, full_seq_len, local_num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, full_seq_len, local_num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, full_seq_len, local_num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        if ulysses_size > 1:
            g = ulysses_gdn_slice_a_log_dt_bias(self, a, ulysses_rank, local_num_v_heads)
        else:
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if local_num_v_heads // local_num_k_heads > 1:
            query = query.repeat_interleave(local_num_v_heads // local_num_k_heads, dim=2)
            key = key.repeat_interleave(local_num_v_heads // local_num_k_heads, dim=2)

        if use_precomputed_states and seq_len == 1:
            recurrent_gated_delta_rule = _resolve_gdn_func(
                self, "recurrent_gated_delta_rule", "torch_recurrent_gated_delta_rule"
            )
            core_attn_out, last_recurrent_state = recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            chunk_gated_delta_rule = _resolve_gdn_func(self, "chunk_gated_delta_rule", "torch_chunk_gated_delta_rule")
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        if ulysses_size > 1:
            core_attn_out = ulysses_gdn_all_to_all_output(core_attn_out, ulysses_group)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

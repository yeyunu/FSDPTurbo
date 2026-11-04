# Copyright c) 2025 NVIDIA CORPORATION AFFILIATES. All rights reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Optional


# pylint: disable=no-name-in-module
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from fsdp_turbo.distributed.context_parallel.batch import split_tensor_to_group
from fsdp_turbo.distributed.context_parallel.kv_allgather.conv_tail import conv_tail_exchange
from fsdp_turbo.distributed.context_parallel.kv_allgather.utils import get_kv_allgather_context
from fsdp_turbo.distributed.context_parallel.ulysses.utils import (
    get_ulysses_context,
    ulysses_all_to_all,
)


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]):
    """
    # pylint: disable=import-errorZero padded hidden states when a two-dimensional mask is available."""
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
    """Resolve a GDN function from instance method or module-level import.

    Searches every class in ``type(self).__mro__`` so that subclasses (which
    keep the stock transformers class in their MRO) still resolve the
    module-level reference defined in the transformers modeling module.
    """
    func = getattr(self, func_name, None)
    if func is not None:
        return func

    if module_func_name is None:
        module_func_name = func_name

    for klass in type(self).__mro__:
        modeling_module = __import__(klass.__module__, fromlist=[""])
        func = getattr(modeling_module, module_func_name, None)
        if func is not None:
            return func
    return None


def gdn_kv_allgather_scan(query, key, value, g, beta, kv_group, total_seq_len):
    """Chunked GDN scan with FLA's native context parallelism on the KV axis.

    Each rank holds one contiguous segment (after the Ulysses head all-to-all)
    of the global sequence described by ``cu_seqlens = [0, total_seq_len]``.
    FLA's CP pre-process computes every rank's local prefix state, exchanges
    the packed ``[S_ext | M]`` tensors (traffic independent of sequence
    length) and merges the chain to recover this rank's initial state, with
    forward and backward both implemented inside its autograd function.
    """
    try:
        from fla.ops.cp import build_cp_context
    except ImportError as error:
        raise RuntimeError(
            "GDN KV-AllGather CP requires a flash-linear-attention build with "
            "fla.ops.cp (KCP support); the installed copy does not provide it."
        ) from error

    from fsdp_turbo.ops.gdn import chunk_gdn

    cu_seqlens = torch.tensor([0, total_seq_len], dtype=torch.long, device=query.device)
    cp_context = build_cp_context(cu_seqlens, group=kv_group)
    core_attn_out, _ = chunk_gdn(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cp_context=cp_context,
    )
    return core_attn_out


class GatedDeltaNet(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        ulysses_group=None,
        **kwargs,
    ):
        if ulysses_group is None:
            ulysses_group, ulysses_rank, ulysses_size = get_ulysses_context()
        else:
            ulysses_rank = dist.get_rank(ulysses_group)
            ulysses_size = dist.get_world_size(ulysses_group)
        kv_group, _kv_rank, kv_size = get_kv_allgather_context()

        if ulysses_size > 1 and cache_params is not None:
            label = "Qwen3.5 MoE" if "Moe" in type(self).__name__ else "Qwen3.5"
            raise ValueError(f"Ulysses CP for {label} GatedDeltaNet does not support cache yet.")
        if kv_size > 1 and cache_params is not None:
            raise ValueError("KV-AllGather CP for GatedDeltaNet does not support cache.")

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
        if kv_size > 1 and batch_size != 1:
            raise ValueError(
                "GDN KV-AllGather CP requires batch size 1 (FLA context parallelism "
                "operates on flattened varlen sequences)."
            )
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

        conv_borrowed = False
        if kv_size > 1:
            # Borrow the previous segment's last conv_kernel-1 tokens (same
            # head shard: the KV-AllGather group fixes the Ulysses coordinate)
            # so the causal conv sees true cross-segment context.
            mixed_qkv, conv_borrowed = conv_tail_exchange(mixed_qkv, kv_group, self.conv_kernel_size - 1)
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
        if conv_borrowed:
            # Outputs for the borrowed prefix tokens belong to the previous
            # segment; drop them so local tokens keep their true context.
            mixed_qkv = mixed_qkv[:, :, self.conv_kernel_size - 1 :]

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

        if kv_size > 1:
            # FLA context-parallel scan across the KV-AllGather axis: the
            # global sequence is [0, seq_len * (kv * ulysses)] and this rank's
            # input is its contiguous segment of it.
            core_attn_out = gdn_kv_allgather_scan(
                query,
                key,
                value,
                g=g,
                beta=beta,
                kv_group=kv_group,
                total_seq_len=seq_len * kv_size * ulysses_size,
            )
            last_recurrent_state = None
        elif use_precomputed_states and seq_len == 1:
            recurrent_gated_delta_rule = _resolve_gdn_func(
                self, "recurrent_gated_delta_rule", "torch_recurrent_gated_delta_rule"
            )
            if recurrent_gated_delta_rule is None:
                raise RuntimeError(
                    f"Could not resolve recurrent_gated_delta_rule for {type(self).__name__}; "
                    "no module in its MRO defines 'torch_recurrent_gated_delta_rule'."
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
            if chunk_gated_delta_rule is None:
                raise RuntimeError(
                    f"Could not resolve chunk_gated_delta_rule for {type(self).__name__}; "
                    "no module in its MRO defines 'torch_chunk_gated_delta_rule'."
                )
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

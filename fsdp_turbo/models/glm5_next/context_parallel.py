# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Ulysses context parallelism for the GLM-5.3-Flash attention layers.

GLM-5.3-Flash has two attention types, and both need a dedicated Ulysses
forward with the "all-to-all heads -> full sequence" strategy:

- :class:`Glm5NextTextAttention` (MLA + DSA indexer), see
  ``get_ulysses_dsa_forward``;
- :class:`Glm5NextTextLinearAttention` (fused KDA variant), see
  ``get_ulysses_linear_attention_forward``.

DSA strategy
------------

``Glm5NextTextAttention`` couples MLA projections with a DeepSeek Sparse
Attention (DSA) indexer that runs *inside* the attention forward: it scores
compressed k-pools of the candidate keys and returns per-query top-k token
indices, which ``build_attention_mask_from_topk`` then turns into the dense
causal-plus-sparse mask. With a rank-sharded sequence this is the piece that
breaks under any wrapper that only exchanges q/k/v: the indexer would score
rank-local tokens and select positions in ``[0, local_seq_len)``, which are
meaningless once q/k/v cover the complete sequence. The stock
``type: full_attention`` patch therefore cannot be used for these layers.

The dedicated forward:

1. runs the MLA q/kv projections on the rank-local sequence shard,
2. all-gathers ``hidden_states`` / ``q_resid`` / padding mask and runs the
   indexer on the *complete* sequence -- identically on every rank, so the
   resulting ``topk_indices`` are global (the indexer is ``@torch.no_grad``
   and cheap next to MLA attention; every rank recomputes it redundantly),
3. exchanges heads for sequence (each rank ends up with
   ``num_heads / world_size`` heads over the whole sequence),
4. runs the attention over the complete sequence:
   - ``dsa_implementation='dense'``: build the global sparse mask from the
     global top-k indices and run the unmodified attention interface
     (``eager_attention_forward`` and its NPU fused dispatch see exactly the
     shapes they see in a non-CP run),
   - ``dsa_implementation='sfa'``: gather the shared compressed latent whole
     (the ``npu_sparse_flash_attention`` operator consumes one latent for all
     heads) and call ``sparse_attention_from_topk`` with the head-sharded
     query -- no dense [B, S, kv_len] mask is ever built,
5. all-to-all back to the local sequence shard and finishes with ``o_proj``.

Cross-layer top-k sharing stays intact: a full indexer layer returns the
global indices when ``next_skip_topk`` is set, and the ``shared`` layer
reuses them directly -- after the head/sequence exchange every rank holds
every query row, so global indices are exactly what both consumers want.

Linear-attention strategy
-------------------------

``Glm5NextTextLinearAttention`` differs from the Kimi K3 ``KimiDeltaAttention``
targeted by :mod:`fsdp_turbo.distributed.context_parallel.ulysses.kda`: GLM-5.3-Flash fuses the q/k/v
projections into one ``qkv_proj`` and the causal short convolutions into one
grouped ``conv1d`` over the concatenated channels, and keeps the forget gate in
a ``forget_gate`` sub-module. The dedicated forward:

1. projects qkv / forget-gate / beta on the rank-local sequence shard,
2. exchanges the per-head channel blocks for complete sequences (each rank
   ends up with ``num_heads / world_size`` heads over the whole sequence),
3. slices the grouped conv weights, A_log and dt_bias to the local heads,
4. runs the unmodified ``chunk_kda`` kernel on the complete sequence,
5. all-to-all back to sequence shards and finishes with the local gate/o_norm.

The recurrent state never crosses ranks: every head sees every token, so the
per-head delta-rule recursion stays mathematically identical to the non-CP run.

The mask/sequence-exchange helpers are shared with :mod:`fsdp_turbo.distributed.context_parallel.ulysses.kda`
(identical Ulysses padding-mask semantics).
"""

from functools import wraps
import importlib

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

from fsdp_turbo.distributed.context_parallel.ulysses.utils import ulysses_all_to_all
from fsdp_turbo.distributed.context_parallel.ulysses.kda import (
    _local_attention_mask,
    _local_channels_to_sequence,
    _sequence_to_local_channels,
)


def _modeling_module(module):
    """Import the modeling module that owns this attention class."""
    return importlib.import_module(type(module).__module__)


def _gather_full_sequence(tensor, group):
    """All-gather rank-local sequence shards into the complete sequence (dim=1)."""
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor
    shards = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(shards, tensor.contiguous(), group=group)
    return torch.cat(shards, dim=1).contiguous()


def get_ulysses_dsa_forward(original_forward, group):
    """Build a Glm5NextTextAttention forward using sequence/head all-to-all."""

    @wraps(original_forward)
    def forward(
        module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs,
    ):
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return original_forward(
                module,
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                prev_topk_indices=prev_topk_indices,
                **kwargs,
            )
        if past_key_values is not None:
            raise ValueError("Ulysses CP for GLM5 DSA attention does not support cache inputs.")
        if kwargs.get("output_attentions"):
            raise NotImplementedError("Ulysses CP for GLM5 DSA attention cannot return attention weights.")
        use_sfa = getattr(module, "dsa_implementation", "dense") == "sfa"
        if module.num_heads % world_size:
            raise ValueError(f"ulysses size ({world_size}) must divide attention num_heads ({module.num_heads}).")
        if module.num_key_value_groups != 1:
            raise ValueError(
                "Ulysses CP for GLM5 DSA attention requires per-head expanded K/V (num_key_value_groups == 1)."
            )

        batch_size, local_seq_len = hidden_states.shape[:2]
        full_seq_len = local_seq_len * world_size

        # --- local MLA projections (identical to the dense path) -------------
        query_shape = (batch_size, local_seq_len, -1, module.qk_head_dim)
        q_resid = module.q_a_layernorm(module.q_a_proj(hidden_states))
        query_states = module.q_b_proj(q_resid).view(query_shape).transpose(1, 2)

        compressed_kv = module.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1)
        k_pass = module.kv_a_layernorm(kv_pass).view(batch_size, 1, local_seq_len, module.kv_lora_rank)

        # --- global top-k selection -------------------------------------------
        # The indexer must score pools of the complete sequence; every rank
        # gathers the same inputs and runs it redundantly, so all ranks agree
        # bit-for-bit and the indices are valid across the whole sequence.
        local_mask = _local_attention_mask(attention_mask, hidden_states, group)
        if local_mask is None:
            local_mask = torch.ones(batch_size, local_seq_len, dtype=torch.bool, device=hidden_states.device)
        full_mask = _gather_full_sequence(local_mask.detach(), group)

        if module.indexer is not None:
            full_hidden = _gather_full_sequence(hidden_states.detach(), group)
            full_q_resid = _gather_full_sequence(q_resid.detach(), group)
            topk_indices = module.indexer(
                hidden_states=full_hidden,
                q_resid=full_q_resid,
                attention_mask=full_mask,
                past_key_values=None,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            if prev_topk_indices.size(1) != full_seq_len:
                raise ValueError(
                    "prev_topk_indices are not global: the previous full DSA layer was not "
                    f"Ulysses-patched (got {tuple(prev_topk_indices.shape)}, expected "
                    f"q_length={full_seq_len})."
                )
            topk_indices = prev_topk_indices

        # --- attention over the complete sequence ------------------------------
        if use_sfa:
            # The SFA operator consumes the indexer's global top-k directly, so no
            # [B, S, kv_len] dense mask is ever built (64 GiB at S=128k -- the
            # dense path's OOM). It only accepts the absorbed MLA layout: one
            # shared latent as key/value, kv_head_num=1. The query is therefore
            # head-sharded by the same exchange as the dense path, while the
            # latent is gathered whole on every rank -- autograd-capable
            # collectives keep the gradients flowing to the local projections.
            query_states = ulysses_all_to_all(query_states, group, scatter_dim=1, gather_dim=2)
            latent_shards = all_gather(k_pass.contiguous(), group=group)
            full_k_pass = torch.cat(latent_shards, dim=2)

            # After the exchange rank r owns heads [r*N_local, (r+1)*N_local)
            # of the complete sequence, so the absorbed W_UK / W_UV rows must
            # be sliced to the same shard the query carries.
            local_num_heads = query_states.shape[1]
            head_offset = dist.get_rank(group) * local_num_heads
            attn_output = module.sparse_attention_from_topk(
                query_states=query_states,
                k_pass=full_k_pass,
                topk_indices=topk_indices,
                head_offset=head_offset,
                num_query_heads=local_num_heads,
            )
        else:
            k_rot = k_rot.view(batch_size, 1, local_seq_len, module.qk_rope_head_dim)
            key_states, value_states = module.expand_kv(k_pass, k_rot)

            # --- exchange heads for the complete sequence ---------------------
            query_states = ulysses_all_to_all(query_states, group, scatter_dim=1, gather_dim=2)
            key_states = ulysses_all_to_all(key_states, group, scatter_dim=1, gather_dim=2)
            value_states = ulysses_all_to_all(value_states, group, scatter_dim=1, gather_dim=2)

            # The indexer already encodes causality and padding into the top-k
            # selection, so the global indices produce the correct global mask.
            sparse_mask = module.build_attention_mask_from_topk(
                topk_indices=topk_indices,
                query_states=query_states,
                kv_length=key_states.shape[2],
            )

            modeling = _modeling_module(module)
            attention_interface = modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
                module.config._attn_implementation, modeling.eager_attention_forward
            )
            attn_output, _ = attention_interface(
                module,
                query_states,
                key_states,
                value_states,
                sparse_mask,
                dropout=0.0 if not module.training else module.attention_dropout,
                scaling=module.scaling,
                **kwargs,
            )

        # --- back to the local sequence shard and finish locally --------------
        # Both branches leave the output in [B, S_full, H_local, D] (the dense
        # attention interface ends with `.transpose(1, 2)` and the SFA path
        # natively produces BSND), so the *sequence* dim is the full one to
        # scatter (dim 1) and the *head* dim is the local one to gather back
        # (dim 2). Scattering/gathering the other way around silently permutes
        # heads and positions: the element count still reshapes into o_proj's
        # input.
        attn_output = ulysses_all_to_all(attn_output, group, scatter_dim=1, gather_dim=2)
        attn_output = attn_output.reshape(batch_size, local_seq_len, -1)
        attn_output = module.o_proj(attn_output)

        return attn_output, None, topk_indices if module.next_skip_topk else None

    return forward


def get_ulysses_linear_attention_forward(original_forward, group):
    """Build a Glm5NextTextLinearAttention forward using sequence/head all-to-all."""

    @wraps(original_forward)
    def forward(
        module,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        modeling = _modeling_module(module)
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return original_forward(
                module,
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )
        if cache_params is not None:
            raise ValueError("Ulysses CP for GLM5 linear attention does not support cache inputs.")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("Ulysses CP for GLM5 linear attention does not support pre-packed cu_seqlens inputs.")
        if getattr(module, "kda_implementation", None) != "triton":
            # The fused chunk kernel signature used below is the triton branch;
            # the eager chunk_kimi_delta_attention branch takes a transformed
            # gate instead of A_log/dt_bias and is not wired for CP.
            raise ValueError("Ulysses CP for GLM5 linear attention requires the triton chunk_kda implementation.")
        if getattr(module, "causal_conv1d_implementation", None) != "triton":
            raise ValueError("Ulysses CP for GLM5 linear attention requires the triton causal_conv1d implementation.")
        if module.num_heads % world_size:
            raise ValueError(f"ulysses size ({world_size}) must divide KDA num_heads ({module.num_heads}).")

        local_mask = _local_attention_mask(attention_mask, hidden_states, group)
        if local_mask is not None:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * local_mask[:, :, None]).to(dtype)

        batch_size, local_seq_len = hidden_states.shape[:2]
        rank = dist.get_rank(group)
        local_num_heads = module.num_heads // world_size
        head_dim = module.head_dim
        head_start = rank * local_num_heads

        # --- projections on the local shard ---------------------------------
        mixed_qkv = module.qkv_proj(hidden_states)  # [B, S_loc, 3 * qkv_dim_full]
        g_raw = module.forget_gate.raw(hidden_states)  # [B, S_loc, qkv_dim_full]
        beta = module.b_proj(hidden_states)  # [B, S_loc, num_heads]

        # --- exchange sequence shards for complete sequences -----------------
        # mixed_qkv channels are [q | k | v]; all-to-all scatters contiguous
        # channel blocks, so exchange each block separately to keep the
        # per-head slices aligned.
        qkv_full = module.qkv_dim  # full-size block (num_heads * head_dim)
        query = _sequence_to_local_channels(mixed_qkv[..., :qkv_full], group)
        key = _sequence_to_local_channels(mixed_qkv[..., qkv_full : 2 * qkv_full], group)
        value = _sequence_to_local_channels(mixed_qkv[..., 2 * qkv_full :], group)
        g_raw = _sequence_to_local_channels(g_raw, group)
        beta = _sequence_to_local_channels(beta, group)

        # --- grouped short convolutions on the complete sequence -------------
        # conv1d is depthwise (groups == channels), so slicing the channel-
        # indexed weights is exact; the causal window spans the full sequence,
        # which is only available after the all-to-all.
        conv_weight = module.conv1d.weight.squeeze(1).transpose(-1, -2).contiguous()  # [W, 3D]
        causal_conv1d = getattr(modeling, "causal_conv1d")
        q_channels = slice(head_start * head_dim, (head_start + local_num_heads) * head_dim)
        k_channels = slice(qkv_full + head_start * head_dim, qkv_full + (head_start + local_num_heads) * head_dim)
        v_channels = slice(
            2 * qkv_full + head_start * head_dim, 2 * qkv_full + (head_start + local_num_heads) * head_dim
        )

        full_seq_len = query.size(1)
        query, _ = causal_conv1d(
            x=query, weight=conv_weight[:, q_channels].contiguous(), bias=None, activation=module.activation
        )
        key, _ = causal_conv1d(
            x=key, weight=conv_weight[:, k_channels].contiguous(), bias=None, activation=module.activation
        )
        value, _ = causal_conv1d(
            x=value, weight=conv_weight[:, v_channels].contiguous(), bias=None, activation=module.activation
        )

        # --- chunked delta rule over the complete sequence --------------------
        hidden_shape = (batch_size, full_seq_len, local_num_heads, head_dim)
        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)
        g = g_raw.view(hidden_shape)
        beta = torch.sigmoid(beta)

        forget_gate = module.forget_gate
        a_log = forget_gate.A_log[head_start : head_start + local_num_heads]
        dt_bias = forget_gate.dt_bias.view(module.num_heads, head_dim)[
            head_start : head_start + local_num_heads
        ].reshape(-1)

        chunk_kda = getattr(modeling, "chunk_kda")
        core_attn_out, _ = chunk_kda(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=forget_gate.safe_gate_lower_bound is not None,
            lower_bound=forget_gate.safe_gate_lower_bound,
            A_log=a_log,
            dt_bias=dt_bias,
            skip_recompute=True,
        )

        # --- back to sequence shards and finish locally -----------------------
        core_attn_out = core_attn_out.reshape(batch_size, full_seq_len, -1)
        core_attn_out = _local_channels_to_sequence(core_attn_out, group)
        core_attn_out = core_attn_out.view(batch_size, local_seq_len, module.num_heads, head_dim)

        gate = module.g_b_proj(module.g_a_proj(hidden_states)).view(
            batch_size, local_seq_len, module.num_heads, head_dim
        )
        output = module.o_norm(core_attn_out, gate).reshape(batch_size, local_seq_len, -1)
        output = module.o_proj(output)
        if local_mask is not None:
            output = (output * local_mask[:, :, None]).to(output.dtype)
        return output

    return forward

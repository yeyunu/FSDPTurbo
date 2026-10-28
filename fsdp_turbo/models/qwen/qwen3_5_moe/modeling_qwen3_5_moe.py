# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Copyright 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""FSDPTurbo modeling for Qwen3.5-MoE."""

from functools import partial
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
from transformers.utils import can_return_tuple

from fsdp_turbo.distributed.context_parallel.batch import (
    cal_split_sizes,
    shard_fused_sequence_inputs,
    split_tensor_to_group,
)
from fsdp_turbo.distributed.context_parallel.communication import (
    gather_forward_split_backward,
    split_forward_gather_backward,
)
from fsdp_turbo.distributed.context_parallel.loss import cp_compute_lm_loss
from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_attention import ulysses_context_parallel_attention
from fsdp_turbo.modules.gated_delta_net import GatedDeltaNet
from fsdp_turbo.ops.rms_norm import rms_norm
from fsdp_turbo.ops.rotary_mul import rotary_mul
from fsdp_turbo.utils.model import register_model


class Qwen3_5MoeRMSNorm(hf.Qwen3_5MoeRMSNorm):
    """RMSNorm that dispatches to the fused ``ops.rms_norm`` on accelerator."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, 1.0 + self.weight.float(), self.eps).type_as(x)


class Qwen3_5MoeRMSNormGated(hf.Qwen3_5MoeRMSNormGated):
    """Gated RMSNorm that dispatches to the fused ``ops.rms_norm`` on accelerator."""

    def forward(self, hidden_states: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normed = rms_norm(hidden_states, self.weight.float(), self.variance_epsilon)
        return (normed * F.silu(gate.to(normed.dtype))).to(input_dtype)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split q/k into rotary / non-rotary parts and apply ``rotary_mul`` on the rotary part."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = rotary_mul(q_rot, cos, sin)
    k_embed = rotary_mul(k_rot, cos, sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply full-dim rotation in float32 then cast back to original dtype."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = rotary_mul(q, cos, sin)
    k_embed = rotary_mul(k, cos, sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen3_5MoeAttention(hf.Qwen3_5MoeAttention):
    """Attention layer using fused rotary embedding and fused q/k RMSNorm."""

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.ulysses_group = None
        self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        # MRoPE has three coordinate planes; FA packing detection needs one
        # text-position plane, not three concatenated sequences.
        position_ids = kwargs.get("position_ids")
        if position_ids is not None and position_ids.ndim == 3:
            kwargs["position_ids"] = position_ids[0]
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        raw_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, hf.eager_attention_forward
        )
        attention_interface = partial(raw_interface, self)
        attention_kwargs = dict(
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            **kwargs,
        )
        if self.ulysses_group is not None:
            attn_output, attn_weights = ulysses_context_parallel_attention(
                attention_interface,
                query_states,
                key_states,
                value_states,
                ulysses_group=self.ulysses_group,
                flash_attention=True,
                **attention_kwargs,
            )
        else:
            attn_output, attn_weights = attention_interface(query_states, key_states, value_states, **attention_kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3_5MoeGatedDeltaNet(hf.Qwen3_5MoeGatedDeltaNet):
    """Gated delta net layer using fused causal conv1d, fused GDN and fused gated RMSNorm.

    All operator functions are resolved from instance attributes so that both
    this class's own ``forward`` and the CP-aware ``GatedDeltaNet.forward``
    (installed by ``ulysses_function_patches``) share the same dispatch path.
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.cp_group = None
        self.use_fused_conv1d = getattr(config, "use_fused_causal_conv1d", False)
        self.use_fused_gdn = getattr(config, "use_fused_gdn", False)
        self.norm = Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        if self.use_fused_gdn:
            from fsdp_turbo.ops.gdn import chunk_gdn, fused_recurrent_gdn

            self.chunk_gated_delta_rule = chunk_gdn
            self.recurrent_gated_delta_rule = fused_recurrent_gdn
        else:
            self.chunk_gated_delta_rule = hf.torch_chunk_gated_delta_rule
            self.recurrent_gated_delta_rule = hf.torch_recurrent_gated_delta_rule

        if self.use_fused_conv1d:
            from fsdp_turbo.ops.causal_conv1d import causal_conv1d, causal_conv1d_update

            def _fused_conv1d_fn(x, weight, bias=None, activation=None, **kw):
                output, _ = causal_conv1d(
                    x.transpose(1, 2).contiguous(),
                    weight.transpose(0, 1),
                    bias=bias,
                    activation=activation,
                    cu_seqlens=kw.get("cu_seqlens"),
                )
                return output.transpose(1, 2).contiguous()

            def _fused_conv1d_update(x, cache, weight, bias=None, activation=None):
                output, cache = causal_conv1d_update(
                    x.transpose(1, 2).contiguous(),
                    cache,
                    weight,
                    bias=bias,
                    activation=activation,
                )
                return output.transpose(1, 2).contiguous()

            self.causal_conv1d_fn = _fused_conv1d_fn
            self.causal_conv1d_update = _fused_conv1d_update
        else:
            self.causal_conv1d_fn = hf.causal_conv1d_fn
            self.causal_conv1d_update = hf.causal_conv1d_update

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.cp_group is not None:
            return GatedDeltaNet.forward(
                self,
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                ulysses_group=self.cp_group,
                **kwargs,
            )
        hidden_states = hf.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )
            mixed_qkv = self.causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
        cu_seqlens = kwargs.pop("cu_seq_lens_q", None)
        if cu_seqlens is None:
            cu_seqlens = kwargs.pop("cu_seqlens", None)

        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


class Qwen3_5MoeDecoderLayer(hf.Qwen3_5MoeDecoderLayer):
    """Decoder layer that assembles FSDPTurbo attention / GDN / MoE / norm sublayers."""

    def __init__(self, config, layer_idx):
        hf.GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.block_type = self.layer_type
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        self.mlp = hf.Qwen3_5MoeSparseMoeBlock(config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3_5MoeTextModel(hf.Qwen3_5MoeTextModel):
    """Text model that rebuilds decoder layers and final norm with FSDPTurbo subclasses."""

    def __init__(self, config):
        hf.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = hf.Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.cp_group = None
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        if self.cp_group is not None:
            if past_key_values is not None or cache_position is not None or use_cache:
                raise NotImplementedError("Qwen3.5 CP does not support KV cache.")
            if any(
                kwargs.get(name) is not None
                for name in (
                    "cu_seqlens",
                    "cu_seq_lens_q",
                    "cu_seq_lens_k",
                    "seq_split_lens",
                )
            ):
                raise NotImplementedError("Qwen3.5 CP does not support packed sequences.")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            if position_ids is None:
                position_ids = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device).unsqueeze(0)
                position_ids = position_ids.expand(inputs_embeds.size(0), -1)
            inputs_embeds, position_ids, attention_mask = shard_fused_sequence_inputs(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                group=self.cp_group,
                gather_backward=True,
            )
            input_ids, use_cache = None, False
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )


class Qwen3_5MoeVisionAttention(hf.Qwen3_5MoeVisionAttention):
    """Vision attention using fused vision rotary embedding."""

    def __init__(self, config):
        super().__init__(config)
        self.cp_group = None

    def forward(
        self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, max_seqlen=None, **kwargs
    ):
        seq_length = hidden_states.shape[0]
        # Explicit head_dim also permits an empty local sequence shard.
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        raw_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, hf.eager_attention_forward
        )
        attention_interface = partial(raw_interface, self)

        # Native HF FA2 expects BHSD rather than MM's custom 1TND layout.
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if self.config._attn_implementation == "flash_attention_2":
            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            attention_kwargs = dict(
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
            if self.cp_group is None:
                attn_output, _ = attention_interface(query_states, key_states, value_states, **attention_kwargs)
            else:
                attn_output, _ = ulysses_context_parallel_attention(
                    attention_interface,
                    query_states,
                    key_states,
                    value_states,
                    ulysses_group=self.cp_group,
                    total_seq_len=int(cu_seqlens[-1]),
                    flash_attention=True,
                    **attention_kwargs,
                )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3_5MoeVisionBlock(hf.Qwen3_5MoeVisionBlock):
    def __init__(self, config):
        hf.GradientCheckpointingLayer.__init__(self)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5MoeVisionAttention(config)
        self.mlp = hf.Qwen3_5MoeVisionMLP(config)


class Qwen3_5MoeVisionPatchMerger(hf.Qwen3_5MoeVisionPatchMerger):
    def __init__(self, config, use_postshuffle_norm=False):
        super().__init__(config, use_postshuffle_norm=use_postshuffle_norm)
        self.cp_group = None

    def forward(self, x, *, sequence_length=None):
        if self.cp_group is None:
            return super().forward(x)
        gather_sizes = cal_split_sizes(sequence_length, dist.get_world_size(self.cp_group))
        # Keep MM's two normalization branches and operation order.
        if self.use_postshuffle_norm:
            x = gather_forward_split_backward(x, self.cp_group, dim=0, gather_sizes=gather_sizes)
            x = x.view(-1, self.hidden_size)
            # Split the merged tensor back for distributed processing.
            split_sizes = cal_split_sizes(x.size(0), dist.get_world_size(self.cp_group))
            x = split_forward_gather_backward(x, self.cp_group, dim=0, grad_scale="down", split_sizes=split_sizes)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = gather_forward_split_backward(x, self.cp_group, dim=0, gather_sizes=gather_sizes)
            x = x.view(-1, self.hidden_size)
            # Split the merged tensor back for distributed processing.
            split_sizes = cal_split_sizes(x.size(0), dist.get_world_size(self.cp_group))
            x = split_forward_gather_backward(x, self.cp_group, dim=0, grad_scale="down", split_sizes=split_sizes)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3_5MoeVisionModel(hf.Qwen3_5MoeVisionModel):
    """Vision tower that rebuilds blocks with fused rotary attention when configured."""

    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList([Qwen3_5MoeVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3_5MoeVisionPatchMerger(config, use_postshuffle_norm=False)
        self.cp_group = None

    def forward(self, hidden_states, grid_thw, **kwargs):
        if self.cp_group is None:
            return super().forward(hidden_states, grid_thw, **kwargs)
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        sequence_lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cpu()
        cu_seqlens = sequence_lengths.cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # MM's pure Ulysses path splits the whole concatenated sequence.
        rotary_pos_emb = split_tensor_to_group(rotary_pos_emb, self.cp_group, dim=0)
        split_sizes = cal_split_sizes(seq_len, dist.get_world_size(self.cp_group))
        hidden_states = split_forward_gather_backward(hidden_states, self.cp_group, dim=0, split_sizes=split_sizes)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states, sequence_length=seq_len)

        # Aggregate the full merged features.
        gather_sizes = cal_split_sizes(seq_len // self.spatial_merge_size**2, dist.get_world_size(self.cp_group))
        merged_hidden_states = gather_forward_split_backward(
            merged_hidden_states,
            self.cp_group,
            dim=0,
            grad_scale="up",
            gather_sizes=gather_sizes,
        )
        return hf.BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


class Qwen3_5MoeModel(hf.Qwen3_5MoeModel):
    """Multimodal wrapper that assembles FSDPTurbo text and vision submodels."""

    def __init__(self, config):
        hf.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.visual = Qwen3_5MoeVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3_5MoeTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


@register_model("qwen3_5_moe")
class Qwen3_5MoeForConditionalGeneration(hf.Qwen3_5MoeForConditionalGeneration):
    def __init__(self, config):
        hf.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.model = Qwen3_5MoeModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.cp_group = None
        self.post_init()

    def configure_context_parallel(self, *, cp_group, ulysses_group):
        """Bind externally created groups to the model's existing CP-aware modules."""

        if cp_group is None or ulysses_group is None:
            raise ValueError("Qwen3.5 CP requires explicit CP and Ulysses process groups.")
        if self.config.text_config._attn_implementation != "flash_attention_2":
            raise NotImplementedError("Qwen3.5 CP requires flash_attention_2.")
        if self.config.vision_config._attn_implementation != "flash_attention_2":
            raise NotImplementedError("Qwen3.5 vision CP requires flash_attention_2.")
        size = dist.get_world_size(cp_group)
        text_config = self.config.text_config
        if dist.get_world_size(ulysses_group) != size:
            raise ValueError("The Ulysses group and CP group must have the same size.")
        if text_config.linear_num_key_heads % size or text_config.linear_num_value_heads % size:
            raise ValueError("GDN key/value heads must be divisible by the CP group size.")
        if text_config.num_attention_heads % size or self.config.vision_config.num_heads % size:
            raise ValueError("Language and vision attention heads must be divisible by the Ulysses group size.")
        self.cp_group = cp_group
        self.model.language_model.cp_group = cp_group
        for layer in self.model.language_model.layers:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.cp_group = cp_group
            else:
                layer.self_attn.ulysses_group = ulysses_group
        self.model.visual.cp_group = ulysses_group
        self.model.visual.merger.cp_group = ulysses_group
        for block in self.model.visual.blocks:
            block.attn.cp_group = ulysses_group

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | hf.Qwen3_5MoeCausalLMOutputWithPast:
        if self.cp_group is not None:
            if getattr(self, "enable_chunk_loss", False) or kwargs.get("output_router_logits", False):
                raise NotImplementedError("Qwen3.5 CP with chunk loss or router auxiliary loss is not supported.")
            if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
                raise NotImplementedError("Qwen3.5 CP requires all local logits (logits_to_keep=0).")
            if past_key_values is not None or cache_position is not None or kwargs.get("use_cache", False):
                raise NotImplementedError("Qwen3.5 CP does not support KV cache.")
            kwargs["use_cache"] = False
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        loss = None
        logits = None
        if getattr(self, "enable_chunk_loss", False):
            # Training injects either legacy chunk loss or CCE through loss_function.
            # Both consume the full sequence; logits_to_keep only selects returned logits.
            loss = self.lm_head(hidden_states, self.loss_function, labels)
        else:
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None:
                if self.cp_group is not None:
                    loss = cp_compute_lm_loss(logits, labels, group=self.cp_group)
                else:
                    loss = self.loss_function(
                        logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
                    )

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = hf.load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

        return hf.Qwen3_5MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            router_logits=outputs.router_logits,
        )


__all__ = ["Qwen3_5MoeForConditionalGeneration"]

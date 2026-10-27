# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeCausalLMOutputWithPast,
    Qwen3_5MoeForConditionalGeneration as HFQwen3_5MoeForConditionalGeneration,
    load_balancing_loss_func,
)
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


class Qwen3_5MoeAttention(hf.Qwen3_5MoeAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.ulysses_group = None

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        if self.config._attn_implementation != "flash_attention_2":
            return super().forward(
                hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs
            )
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
        query_states, key_states = hf.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
        attention_interface = partial(flash_attention_forward, self)
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
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.cp_group = None

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if self.cp_group is None:
            return super().forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
        return GatedDeltaNet.forward(
            self,
            hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
            ulysses_group=self.cp_group,
            **kwargs,
        )


class Qwen3_5MoeDecoderLayer(hf.Qwen3_5MoeDecoderLayer):
    def __init__(self, config, layer_idx):
        hf.GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        self.mlp = hf.Qwen3_5MoeSparseMoeBlock(config)
        self.input_layernorm = hf.Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = hf.Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3_5MoeTextModel(hf.Qwen3_5MoeTextModel):
    def __init__(self, config):
        hf.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = hf.Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
    def __init__(self, config):
        super().__init__(config)
        self.cp_group = None

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        if self.config._attn_implementation != "flash_attention_2":
            return super().forward(hidden_states, cu_seqlens, rotary_pos_emb, position_embeddings, **kwargs)
        seq_length = hidden_states.shape[0]
        # Explicit head_dim also permits an empty local sequence shard.
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = hf.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        attention_interface = partial(flash_attention_forward, self)
        # Native HF FA2 expects BHSD rather than MM's custom 1TND layout.
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Flash Attention 2: Use cu_seqlens for variable length attention
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
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

        attn_output = attn_output.reshape(seq_length, self.dim).contiguous()
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
    def __init__(self, config):
        hf.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.patch_embed = hf.Qwen3_5MoeVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.rotary_pos_emb = hf.Qwen3_5MoeVisionRotaryEmbedding(config.hidden_size // config.num_heads // 2)
        self.blocks = nn.ModuleList([Qwen3_5MoeVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3_5MoeVisionPatchMerger(config, use_postshuffle_norm=False)
        self.gradient_checkpointing = False
        self.cp_group = None
        self.post_init()

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

        # 聚合完整的 merged 特征
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
    def __init__(self, config):
        hf.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.visual = Qwen3_5MoeVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3_5MoeTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class Qwen3_5MoeForConditionalGeneration(HFQwen3_5MoeForConditionalGeneration):
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
    ) -> tuple | Qwen3_5MoeCausalLMOutputWithPast:
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
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

        return Qwen3_5MoeCausalLMOutputWithPast(
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

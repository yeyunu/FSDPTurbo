# Copyright 2026 the HuggingFace Team. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Adapted from the Apache-2.0 GLM-MoE-DSA modeling and its MindSpeed-LLM optimization.
"""GLM53 training subclasses; no model monkeypatches or LLM framework imports."""

import math
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as hf
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from fsdp_turbo.ops.glm_indexer import attach_indexer_loss, indexer_kl_loss, lightning_indexer
from fsdp_turbo.ops.moe import grouped_matmul
from fsdp_turbo.ops.rms_norm import rms_norm
from fsdp_turbo.ops.rotary_mul import rotary_mul
from fsdp_turbo.ops.selected_attention import selected_attention

LossContext = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]
_HFExperts = getattr(hf, "GlmMoeDsaExperts", None) or hf.GlmMoeDsaNaiveMoe


def _rotary(query, key, cosine, sine, dimension, interleave=False):
    cosine, sine = cosine.unsqueeze(dimension), sine.unsqueeze(dimension)
    if interleave:
        query = torch.cat((query[..., ::2], query[..., 1::2]), dim=-1)
        key = torch.cat((key[..., ::2], key[..., 1::2]), dim=-1)
    return rotary_mul(query, cosine, sine), rotary_mul(key, cosine, sine)


class Glm53RMSNorm(hf.GlmMoeDsaRMSNorm):
    """Select the existing fused RMSNorm with a tensor-device CPU fallback."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Preserve the LLM activation dtype after normalization."""
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


class Glm53RotaryEmbedding(hf.GlmMoeDsaRotaryEmbedding):
    """Use the MLA rotary width, rather than hidden_size / attention heads."""

    def __init__(self, config: GlmMoeDsaConfig) -> None:
        rotary_config = copy(config)
        rotary_config.head_dim = config.qk_rope_head_dim
        rotary_config.rope_parameters = {**config.rope_parameters, "partial_rotary_factor": 1.0}
        super().__init__(rotary_config)


class Glm53Indexer(hf.GlmMoeDsaIndexer):
    """Retain differentiable Indexer inputs while detaching token selection."""

    def build_index_tensors(
        self,
        hidden_states: torch.Tensor,
        query_residual: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build BSND Indexer queries/keys and scaled per-head weights."""
        batch, sequence, _ = hidden_states.shape
        query = self.wq_b(query_residual).view(batch, sequence, self.n_heads, self.head_dim)
        key = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
        query_rope, query_nope = query.split([self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], -1)
        key_rope, key_nope = key.split([self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], -1)
        query_rope, key_rope = _rotary(query_rope, key_rope, *position_embeddings, dimension=2)
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype))
        weights = weights * self.n_heads**-0.5 * self.softmax_scale
        return torch.cat((query_rope, query_nope), -1), torch.cat((key_rope, key_nope), -1), weights


class Glm53Attention(hf.GlmMoeDsaAttention):
    """Expanded selected-support FA with detached MLA teacher and trainable Indexer KL."""

    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int, indexer_backward_scale: float = 1.0) -> None:
        super().__init__(config, layer_idx)
        self.q_a_layernorm = Glm53RMSNorm(config.q_lora_rank)
        self.kv_a_layernorm = Glm53RMSNorm(config.kv_lora_rank)
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.scaling = self.qk_head_dim**-0.5
        if config.rope_parameters.get("rope_type", "default") != "default":
            multiplier = config.rope_parameters.get("mscale_all_dim", 0)
            factor = config.rope_parameters["factor"]
            if multiplier and factor > 1:
                self.scaling *= (0.1 * multiplier * math.log(factor) + 1.0) ** 2
        self.indexer = None if self.skip_topk else Glm53Indexer(config, layer_idx)
        self.indexer_backward_scale = indexer_backward_scale
        self.indexer_loss = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        position_ids: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        """Execute equal-length, unpadded training without a generation KV cache."""
        if past_key_values is not None:
            raise ValueError("Glm53Attention native training does not support a generation KV cache.")
        batch, sequence, _ = hidden_states.shape
        query_residual = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = (
            self.q_b_proj(query_residual).view(batch, sequence, self.num_heads, self.qk_head_dim).transpose(1, 2)
        )
        query_nope, query_rope = query_states.split([self.qk_nope_head_dim, self.qk_rope_head_dim], -1)
        latent, key_rope = self.kv_a_proj_with_mqa(hidden_states).split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        latent = self.kv_a_layernorm(latent)
        key_rope = key_rope.view(batch, 1, sequence, self.qk_rope_head_dim)
        query_rope, key_rope = _rotary(query_rope, key_rope, *position_embeddings, dimension=1, interleave=True)
        self.indexer_loss = None
        coefficient = getattr(self.config, "indexer_loss_coeff", 0.0)
        train_indexer = self.training and torch.is_grad_enabled() and coefficient > 0
        if self.indexer is not None:
            query_index, key_index, weights = self.indexer.build_index_tensors(
                hidden_states.detach(),
                query_residual.detach(),
                position_embeddings,
            )
            with torch.no_grad():
                indices, _ = lightning_indexer(query_index, key_index, weights, self.indexer.index_topk)
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared Indexer layers require indices from a preceding full layer.")
            indices = prev_topk_indices
        projected = self.kv_b_proj(latent).view(
            batch, sequence, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        key_nope, values = projected.split([self.qk_nope_head_dim, self.v_head_dim], -1)
        query = torch.cat((query_nope, query_rope), -1).transpose(1, 2).contiguous()
        key = torch.cat((key_nope, key_rope.transpose(1, 2).expand(-1, -1, self.num_heads, -1)), -1).contiguous()
        output, maximum, total = selected_attention(query, key, values, indices, self.scaling)
        if train_indexer and self.indexer is not None:
            with torch.no_grad():
                projection_weight = self.kv_b_proj.weight
                if hasattr(projection_weight, "to_local"):
                    projection_weight = projection_weight.to_local()
                projection_weight = projection_weight.view(
                    self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
                )
                teacher_query = torch.einsum(
                    "bhsq,hqr->bhsr", query_nope, projection_weight[:, : self.qk_nope_head_dim]
                )
                teacher_query = teacher_query.transpose(1, 2).contiguous()
                teacher_key = latent.unsqueeze(2).contiguous()
            loss = (
                indexer_kl_loss(
                    teacher_query,
                    teacher_key,
                    query_index,
                    key_index,
                    weights,
                    indices,
                    maximum,
                    total,
                    self.scaling,
                    query_rope.transpose(1, 2).contiguous(),
                    key_rope.transpose(1, 2).contiguous(),
                )
                * coefficient
            )
            self.indexer_loss = loss.detach()
            output = attach_indexer_loss(output, loss, self.indexer_backward_scale)
        output = self.o_proj(output.reshape(batch, sequence, -1).contiguous())
        return output, None, indices


class Glm53Experts(_HFExperts):
    """Keep HF checkpoint layout and expose the dimensions needed by MXFP8 conversion."""

    def __init__(self, config: GlmMoeDsaConfig) -> None:
        super().__init__(config)
        self.intermediate_size = config.moe_intermediate_size

    def ep_forward(self, hidden_states: torch.Tensor, split_list: torch.Tensor) -> torch.Tensor:
        """BF16 fallback; the native quantization converter supplies MXFP8 ep_forward."""
        gate_up = self.gate_up_proj.to_local() if hasattr(self.gate_up_proj, "to_local") else self.gate_up_proj
        down = self.down_proj.to_local() if hasattr(self.down_proj, "to_local") else self.down_proj
        intermediate = grouped_matmul(hidden_states, split_list, gate_up, use_eager=hidden_states.device.type == "cpu")
        gate, up = intermediate.chunk(2, -1)
        return grouped_matmul(self.act_fn(gate) * up, split_list, down, use_eager=hidden_states.device.type == "cpu")


class Glm53Router(hf.GlmMoeDsaTopkRouter):
    """Keep the same group routing across Transformers router interfaces."""

    def __init__(self, config: GlmMoeDsaConfig) -> None:
        super().__init__(config)
        self.config = config

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exclude unselected groups with -inf, preserving the LLM routing objective."""
        config = self.config
        inputs = hidden_states.view(-1, config.hidden_size).float()
        probabilities = F.linear(inputs, self.weight.float()).sigmoid()
        choice = probabilities + self.e_score_correction_bias
        experts_per_group = config.n_routed_experts // config.n_group
        group_scores = choice.view(-1, config.n_group, experts_per_group).topk(2, -1)[0].sum(-1)
        selected_groups = group_scores.topk(config.topk_group, dim=-1, sorted=False)[1]
        groups = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, selected_groups, True)
        allowed = groups.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(-1, config.n_routed_experts)
        indices = choice.masked_fill(~allowed, float("-inf")).topk(config.num_experts_per_tok, -1, sorted=False)[1]
        weights = probabilities.gather(1, indices)
        if config.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return indices, weights * config.routed_scaling_factor


class Glm53MoE(hf.GlmMoeDsaMoE):
    """Assemble the optimized router, routed experts and shared experts."""

    def __init__(self, config: GlmMoeDsaConfig) -> None:
        super().__init__(config)
        self.gate = Glm53Router(config)
        self.experts = Glm53Experts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Call the routed expert module with indices before weights."""
        indices, weights = self.gate(hidden_states)
        flattened = hidden_states.view(-1, hidden_states.shape[-1])
        routed = self.experts(flattened, indices, weights).view_as(hidden_states)
        return routed + self.shared_experts(hidden_states)


class Glm53DecoderLayer(hf.GlmMoeDsaDecoderLayer):
    """Assemble optimized attention/norm/MoE classes without replacing parent classes."""

    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int, indexer_backward_scale: float = 1.0) -> None:
        super().__init__(config, layer_idx)
        self.self_attn = Glm53Attention(config, layer_idx, indexer_backward_scale)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Glm53MoE(config)
        self.input_layernorm = Glm53RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Glm53RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate shared Indexer selections along with the residual stream."""
        attention, _, indices = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prev_topk_indices=prev_topk_indices,
            **kwargs,
        )
        hidden_states = hidden_states + attention
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)), indices


class Glm53Model(hf.GlmMoeDsaModel):
    """GLM53 backbone with shared top-k support propagated between decoder layers."""

    def __init__(self, config: GlmMoeDsaConfig, indexer_backward_scale: float = 1.0) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Glm53DecoderLayer(config, index, indexer_backward_scale) for index in range(config.num_hidden_layers)]
        )
        self.norm = Glm53RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Glm53RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> BaseModelOutputWithPast:
        """Execute unpadded training; causal support is enforced by the Indexer."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")
        if attention_mask is not None:
            raise ValueError("The unpadded GLM53 training adapter expects attention_mask=None.")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache", False):
            raise ValueError("The GLM53 training adapter does not support a generation KV cache.")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        embeddings = self.rotary_emb(hidden, position_ids)
        previous_indices = None
        for layer in self.layers:
            hidden, previous_indices = layer(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=embeddings,
                prev_topk_indices=previous_indices,
            )
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden))


class _LMHead(nn.Linear):
    def forward(
        self, hidden: torch.Tensor, loss_ctx: LossContext | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run the head through its FSDP pre-forward hook before evaluating chunked loss."""
        if loss_ctx is not None:
            return None, loss_ctx(hidden, self.weight, self.bias)
        return super().forward(hidden), None


@dataclass
class Glm53CausalLMOutput(CausalLMOutputWithPast):
    """Include the detached token-mean Indexer metric without adding it to LM loss."""

    aux_loss: torch.Tensor | None = None


class Glm53ForCausalLM(hf.GlmMoeDsaForCausalLM):
    """Native FSDPTurbo training model with optional externally supplied ChunkLoss."""

    def __init__(self, config: GlmMoeDsaConfig, indexer_backward_scale: float = 1.0) -> None:
        if config.attention_dropout != 0 or config.q_lora_rank is None:
            raise ValueError("The GLM53 optimized training recipe requires zero attention dropout and query LoRA.")
        super().__init__(config)
        self.model = Glm53Model(config, indexer_backward_scale)
        self.lm_head = _LMHead(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        loss_ctx: LossContext | None = None,
        **kwargs: object,
    ) -> Glm53CausalLMOutput:
        """Compute LM loss, while each attention output carries its scaled KL gradient."""
        output = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, position_ids=position_ids, **kwargs)
        logits, loss = self.lm_head(output.last_hidden_state, loss_ctx=loss_ctx)
        if loss_ctx is None and labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        losses = [
            layer.self_attn.indexer_loss for layer in self.model.layers if layer.self_attn.indexer_loss is not None
        ]
        auxiliary = torch.stack(losses).sum() if losses else None
        # Pylint cannot infer fields inherited from the Transformers dataclass.
        # pylint: disable-next=unexpected-keyword-arg
        return Glm53CausalLMOutput(loss=loss, logits=logits, aux_loss=auxiliary)

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Kimi-K31 variant: Lightning Indexer, causal top-k attention and sparse KL.

Scoring follows DeepSeek-V3.2-Exp (MIT license); Kimi uses a no-RoPE indexer.
Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
Main attention uses an explicit sparse mask with dense FlashAttention kernels;
the optional AscendC path uses absorbed, indexed shared-KV MLA.
"""

import torch
from torch import nn
from torch.nn import functional as F

from fsdp_turbo.ops.flash_attn import flash_attention_forward
from fsdp_turbo.ops.lightning_indexer import lightning_indexer_topk, lightning_indexer_kl_loss
from fsdp_turbo.ops.sparse_attention import sparse_flash_attention
from .configuration_kimi_k31 import KimiLinearConfig
from .modeling_kimi_k31 import KimiK31ForConditionalGeneration as KimiK31BaseForConditionalGeneration
from .modeling_kimi_linear import (
    KimiMLAAttention,
    KimiDecoderLayer,
    KimiLinearModel,
    KimiLinearForCausalLM,
)
from .moe import FSDPTurboKimiSparseMoeBlock


class LightningIndexer(nn.Module):
    """Score causal keys with weighted ReLU dot products shared by all MLA heads."""

    def __init__(self, config: KimiLinearConfig) -> None:
        super().__init__()
        index_config = config.dsa_config
        self.num_heads = index_config["index_num_heads"]
        self.head_dim = index_config["index_head_dim"]
        self.topk = index_config["index_topk"]
        self.implementation = index_config.get("implementation", "eager")
        if self.implementation not in {"eager", "ascendc"}:
            raise ValueError("DSA implementation must be eager or ascendc")
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)

    def project(
        self, hidden_states: torch.Tensor, query_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project detached main-model features to indexer Q/K and unscaled weights."""
        batch, sequence, _ = hidden_states.shape
        hidden_states, query_latent = hidden_states.detach(), query_latent.detach()
        query = self.wq_b(query_latent).view(batch, sequence, self.num_heads, self.head_dim)
        key = self.k_norm(self.wk(hidden_states))
        return query, key, self.weights_proj(hidden_states)

    def forward(self, hidden_states: torch.Tensor, query_latent: torch.Tensor) -> torch.Tensor:
        """Return differentiable FP32 scores [batch, query, key]."""
        query, key, weights = self.project(hidden_states, query_latent)
        weights = weights.float() * self.num_heads**-0.5
        scores = torch.einsum("bshd,btd->bsht", query.float(), key.float()) * self.head_dim**-0.5
        return torch.einsum("bsh,bsht->bst", weights, F.relu(scores))


class KimiDSAAttention(KimiMLAAttention):
    """Preserve MLA projections/output gate and restrict causal attention to learned top-k."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        if not config.mla_use_nope or config.dsa_config.get("index_rope_dim", 0) != 0:
            raise NotImplementedError("This Kimi DSA variant requires the source no-RoPE configuration")
        self.indexer = LightningIndexer(config)
        self.indexer_loss = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Full-sequence CP=1 training, optionally isolated by cu_seqlens.

        Packed inputs have one batch row and a separate binary padding mask;
        cache/generation is unsupported.
        """
        if past_key_values is not None:
            raise NotImplementedError("Kimi DSA training requires use_cache=False")
        batch, sequence, _ = hidden_states.shape
        query_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(query_latent).view(batch, sequence, self.num_heads, self.q_head_dim)
        compressed = self.kv_a_proj_with_mqa(hidden_states)
        latent, key_rotary = compressed.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent = self.kv_a_layernorm(latent)
        if self.indexer.implementation == "ascendc":
            return self._forward_sparse_mla(
                hidden_states, query_latent, query, latent, key_rotary, attention_mask, cu_seqlens
            )
        expanded = self.kv_b_proj(latent).view(batch, sequence, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        key_plain, value = expanded.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        key = torch.cat((key_plain, key_rotary.unsqueeze(2).expand(-1, -1, self.num_heads, -1)), dim=-1)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        positions = torch.arange(sequence, device=hidden_states.device)
        visible = (positions[:, None] >= positions[None, :])[None, :, :].expand(batch, -1, -1)
        if cu_seqlens is not None:
            document_ids = torch.repeat_interleave(
                torch.arange(cu_seqlens.numel() - 1, device=hidden_states.device),
                torch.diff(cu_seqlens.flatten()),
                output_size=sequence,
            )
            visible = visible & (document_ids[:, None] == document_ids[None, :])
        if attention_mask is not None:
            # Combine causal visibility with padding or the caller's packed mask.
            # Boolean masks mark visible keys; additive masks use zero for them.
            if attention_mask.ndim == 2:
                visible = visible & attention_mask[:, None, :].bool()
            else:
                mask = attention_mask[:, 0, :, :]
                visible = visible & (mask if mask.dtype == torch.bool else mask == 0)
        valid_queries = visible.any(-1)
        if attention_mask is not None and attention_mask.ndim == 2:
            valid_queries = valid_queries & attention_mask.bool()
        # Keep one finite entry for padding-only queries, whose labels are ignored.
        visible = visible | (
            ~visible.any(-1, keepdim=True) & torch.eye(sequence, device=hidden_states.device, dtype=torch.bool)
        )
        scores = self.indexer(hidden_states, query_latent)
        scores = scores.masked_fill(~visible, -torch.inf)
        # Stable sorting defines ties by key position, including early causal rows.
        indices = torch.argsort(scores.detach(), dim=-1, descending=True, stable=True)[
            ..., : min(self.indexer.topk, sequence)
        ]
        selected = torch.zeros_like(visible).scatter(-1, indices, True) & visible
        sparse_mask = torch.zeros_like(scores).masked_fill(~selected, -torch.inf).unsqueeze(1)
        if self.training:
            # Distill the mean MLA-head distribution on the selected keys. The
            # teacher and discrete selection are detached, leaving KL gradients
            # only on the indexer's scores for those same keys.
            with torch.no_grad():
                teacher_scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * self.scaling
                teacher_scores.masked_fill_(~selected.unsqueeze(1), -torch.inf)
                teacher = teacher_scores.softmax(-1).mean(1)
            student_log = scores.masked_fill(~selected, -torch.inf).log_softmax(-1)
            # Avoid 0 * -inf at unselected positions.
            divergence = teacher * (teacher.clamp_min(1e-30).log() - student_log.masked_fill(~selected, 0))
            self.indexer_loss = (divergence.sum(-1) * valid_queries).sum() / valid_queries.sum().clamp_min(1)
        if self.config._attn_implementation == "eager":
            probabilities = (torch.matmul(query, key.transpose(-1, -2)) * self.scaling + sparse_mask).softmax(
                -1, dtype=torch.float32
            )
            output = torch.matmul(probabilities.to(value.dtype), value).transpose(1, 2)
        else:
            # The shared flash-attention interface expects matching Q/K/V head
            # widths. Zero padding preserves MLA values; discard it afterward.
            padded_value = F.pad(value, (0, self.q_head_dim - self.v_head_dim))
            output, _ = flash_attention_forward(
                self,
                query,
                key,
                padded_value,
                sparse_mask,
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
            output = output[..., : self.v_head_dim]
        output = output.reshape(batch, sequence, -1).contiguous() * valid_queries.unsqueeze(-1)
        if self.use_output_gate:
            output = output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(output)

    def _forward_sparse_mla(
        self,
        hidden_states: torch.Tensor,
        query_latent: torch.Tensor,
        query: torch.Tensor,
        latent: torch.Tensor,
        key_rotary: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply absorbed MLA with the same parameter tensors and explicit source scale."""
        if self.training and self.attention_dropout != 0:
            raise NotImplementedError("Fused DSA training requires zero attention dropout")
        batch, sequence = hidden_states.shape[:2]
        positions = torch.arange(sequence, device=hidden_states.device)
        if attention_mask is None:
            lengths = [sequence] * batch
        else:
            if attention_mask.ndim == 2:
                valid = attention_mask.bool()
            else:
                causal = positions[:, None] >= positions[None, :]
                mask = attention_mask[:, 0]
                visible = causal[None] & (mask if mask.dtype == torch.bool else mask == 0)
                valid = visible.any(-1)
            lengths = valid.sum(-1).tolist()
            for item, length in enumerate(lengths):
                if not torch.equal(valid[item], positions < length) or (
                    attention_mask.ndim != 2
                    and not torch.equal(visible[item, :length, :length], causal[:length, :length])
                ):
                    raise NotImplementedError(
                        "Fused DSA supports causal sequences with optional right padding; packed masks require the eager impl"
                    )
        projection = self.kv_b_proj.weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        key_projection, value_projection = projection.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        # Absorb the key projection into Q and move the value projection after
        # attention, preserving the original projection parameters and scale.
        query_plain, query_rope = query.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        query_absorbed = torch.einsum("bshd,hdr->bshr", query_plain, key_projection)
        index_query, index_key, index_weights = self.indexer.project(hidden_states, query_latent)
        # For Hidx=32,Didx=128 the combined multiplier is exactly 1/64,
        # representable in BF16 without an additional rounding operation.
        index_weights = index_weights * (self.indexer.num_heads * self.indexer.head_dim) ** -0.5
        from dsa_trace import training_stage
        training_stage(self, query_latent=query_latent, query=query, latent=latent,
            key_rotary=key_rotary, query_absorbed=query_absorbed,
            index_query=index_query, index_key=index_key, index_weights=index_weights,
            key_projection=key_projection, value_projection=value_projection)
        if cu_seqlens is not None:
            # Remove global right padding, including documents entirely in it.
            # All valid documents share one TND call and document-local indices.
            boundaries = cu_seqlens.flatten().clamp_max(lengths[0])
            cu_seqlens = boundaries[torch.cat((boundaries.new_ones(1, dtype=torch.bool), boundaries.diff() > 0))]
        outputs, losses = [], []
        for item, length in enumerate(lengths):
            if length == 0:
                outputs.append(hidden_states.new_zeros((1, sequence, self.num_heads, self.v_head_dim)))
                continue
            parts = [
                tensor[item, :length]
                for tensor in (
                    index_query,
                    index_key,
                    index_weights,
                    query_absorbed,
                    latent.unsqueeze(2),
                    query_rope,
                    key_rotary.unsqueeze(2),
                )
            ]
            if cu_seqlens is None:
                # Rebuild the canonical singleton batch stride for BSND.
                parts = [tensor.unsqueeze(0) for tensor in parts]
            index_query_part, index_key_part, weights_part, query_part, key_part, query_rope_part, key_rope_part = parts
            indices = lightning_indexer_topk(
                index_query_part, index_key_part, weights_part, self.indexer.topk, cu_seqlens
            )
            output, maximum, denominator = sparse_flash_attention(
                query_part,
                key_part,
                key_part,
                indices.squeeze(-2),
                self.scaling,
                rope_head_dim=self.qk_rope_head_dim,
                query_rope=query_rope_part,
                key_rope=key_rope_part,
                cu_seqlens=cu_seqlens,
                return_softmax_stats=True,
            )
            training_stage(self, indices=indices, attention_core=output,
                maximum=maximum, denominator=denominator,
                query_part=query_part, key_part=key_part,
                query_rope=query_rope_part, key_rope=key_rope_part)
            if self.training:
                losses.append(
                    lightning_indexer_kl_loss(
                        index_query_part,
                        index_key_part,
                        weights_part,
                        query_part.detach(),
                        key_part.detach(),
                        query_rope_part.detach(),
                        key_rope_part.detach(),
                        indices,
                        maximum,
                        denominator,
                        self.scaling,
                        cu_seqlens,
                    )
                    * length
                )
            output = torch.einsum("...hr,hdr->...hd", output, value_projection).reshape(
                1, length, self.num_heads, self.v_head_dim
            )
            training_stage(self, value_output=output)
            outputs.append(F.pad(output, (0, 0, 0, 0, 0, sequence - length)))
        if self.training:
            self.indexer_loss = sum(losses) / max(sum(lengths), 1) if losses else index_weights.float().sum() * 0
        output = torch.cat(outputs, dim=0).reshape(batch, sequence, -1).contiguous()
        if self.use_output_gate:
            output = output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(output)


class KimiTurboDecoderLayer(KimiDecoderLayer):
    """Construct the EP-compatible MoE subclass without changing reference classes."""

    sparse_moe_class = FSDPTurboKimiSparseMoeBlock


class KimiTurboModel(KimiLinearModel):
    """Kimi reference attention with packed, EP-capable expert parameters."""

    decoder_layer_class = KimiTurboDecoderLayer


class KimiTurboForCausalLM(KimiLinearForCausalLM):
    """Preserve standard Kimi behavior with constructor-selected MoE adaptation."""

    model_class = KimiTurboModel


class KimiK31TurboForConditionalGeneration(KimiK31BaseForConditionalGeneration):
    """Default FSDPTurbo Kimi-K3 adapter, retaining MLA full attention."""

    language_model_class = KimiTurboForCausalLM


class KimiDSADecoderLayer(KimiTurboDecoderLayer):
    """KDA is retained; every full-attention layer is a DSA instance."""

    full_attention_class = KimiDSAAttention


class KimiDSAModel(KimiTurboModel):
    """Kimi decoder with DSA in all configured full-attention layers."""

    decoder_layer_class = KimiDSADecoderLayer


class KimiDSAForCausalLM(KimiTurboForCausalLM):
    """Kimi causal LM with DSA decoder assembly."""

    model_class = KimiDSAModel


class KimiK31ForConditionalGeneration(KimiK31TurboForConditionalGeneration):
    """Multimodal Kimi-K3 with DSA in every former MLA full-attention layer."""

    language_model_class = KimiDSAForCausalLM

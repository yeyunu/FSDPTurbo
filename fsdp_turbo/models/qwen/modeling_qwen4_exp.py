# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pylint: disable=too-many-lines

"""FSDPTurbo modeling for Qwen3.8-Flash-Next (qwen4_exp).

All acceleration is implemented by subclassing the stock ``transformers``
qwen4_exp classes and overriding ``__init__``/``forward``; no monkey-patching
at runtime.  Fused NPU kernels are reached exclusively through the
``fsdp_turbo.ops`` dispatch registry (this module never imports ``torch_npu``
directly).  Local subclasses keep the stock names (the imported stock classes
are aliased as ``HFxxx``); build the model directly from
``Qwen4ExpForConditionalGeneration`` (its ``from_pretrained`` accepts the
same arguments as ``AutoModelForImageTextToText``); note that
``AutoModel*.register`` cannot remap a native Transformers config to a custom
class, so the auto-mapping keeps resolving to the stock model.

Overrides, by concern:
  - Qwen4ExpTextRMSNorm / Qwen4ExpTextRMSNormGated: fused RMSNorm via the
    fsdp_turbo.ops rms_norm operator (npu_rms_norm)
  - Qwen4ExpTextExperts: fsdp_turbo.ops MoE pipeline (permute /
    grouped_matmul / swiglu / unpermute) for the fused grouped-MatMul path
  - Qwen4ExpTextAttention: fused q/k norms, fused QSA indexer and
    ops-dispatched attention RoPE
  - Qwen4ExpTextQSAIndexer: token selection via the backend-dispatched
    ``qsa_selected_token_mask`` operator (fsdp_turbo.ops.qsa)
  - Qwen4ExpTextGatedDeltaNet: causal_conv1d + GDN kernels dispatched to
    the fsdp_turbo.ops registry, steered by causal_conv1d_implementation /
    gdn_implementation / skip_gdn_recompute config knobs
  - Qwen4ExpTextNGramEmbeddingCpu: PLE n-gram table CPU offload
  - Qwen4ExpModel: post-fusion Ulysses CP forward (image tokens expand
    the sequence, so sharding happens *after* multimodal fusion)
  - Qwen4ExpForConditionalGeneration: MTP block and MTP loss wiring
"""

import logging
import os
from collections.abc import Callable

import torch
from torch import nn

from transformers import AutoModel
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpCausalLMOutputWithPast,
    Qwen4ExpForConditionalGeneration as HFQwen4ExpForConditionalGeneration,
    Qwen4ExpModel as HFQwen4ExpModel,
    Qwen4ExpModelOutputWithPast,
    Qwen4ExpPreTrainedModel,
    Qwen4ExpTextAttention as HFQwen4ExpTextAttention,
    Qwen4ExpTextDecoderLayer as HFQwen4ExpTextDecoderLayer,
    Qwen4ExpTextExperts as HFQwen4ExpTextExperts,
    Qwen4ExpTextGatedDeltaNet as HFQwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextMLP,
    Qwen4ExpTextModel as HFQwen4ExpTextModel,
    Qwen4ExpTextNGramEmbedding,
    Qwen4ExpTextPLELayer as HFQwen4ExpTextPLELayer,
    Qwen4ExpTextQSAIndexer as HFQwen4ExpTextQSAIndexer,
    Qwen4ExpTextRMSNorm as HFQwen4ExpTextRMSNorm,
    Qwen4ExpTextRMSNormGated as HFQwen4ExpTextRMSNormGated,
    Qwen4ExpTextRotaryEmbedding,
    Qwen4ExpTextSparseMoeBlock as HFQwen4ExpTextSparseMoeBlock,
    Qwen4ExpTextTopKRouter,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb as _stock_apply_rotary_pos_emb,
    causal_conv1d_fn as _stock_causal_conv1d_fn,
    causal_conv1d_update as _stock_causal_conv1d_update,
    eager_attention_forward as _stock_eager_attention_forward,
)

from fsdp_turbo.distributed.context_parallel.batch import (
    pad_tensor_to_group,
    shard_fused_sequence_inputs,
    split_tensor_to_group,
)
from fsdp_turbo.distributed.context_parallel.ulysses.utils import get_ulysses_context
from fsdp_turbo.distributed.parallel_state import get_parallel_state
from fsdp_turbo.ops.moe import grouped_matmul, permute, unpermute
from fsdp_turbo.ops.quantize import swiglu
from fsdp_turbo.ops.qsa import qsa_selected_token_mask
from fsdp_turbo.ops.rms_norm import rms_norm
from fsdp_turbo.ops.rope import rope
from fsdp_turbo.utils.device import get_accelerator_name
from fsdp_turbo.utils.log import print_rank

from fsdp_turbo.modules.mtp import MultiTokenPredictionBlock

logger = logging.getLogger(__name__)

_IS_NPU = get_accelerator_name() == "npu"


def _apply_rotary_pos_emb(q, k=None, cos=None, sin=None, unsqueeze_dim=1):
    """Attention RoPE via the ``fsdp_turbo.ops.rope`` operator on NPU, stock math otherwise."""
    if q.device.type != "npu":
        return _stock_apply_rotary_pos_emb(q, k, cos=cos, sin=sin, unsqueeze_dim=unsqueeze_dim)
    # The op rotates [B, S, H, D] inputs with unique D/2-wide coefficients,
    # while the HF cos/sin are full-width duplicated halves; feed the unique
    # half and keep any passthrough dims outside the op.
    rotary_dim = cos.shape[-1]
    half_cos = cos[:, :, None, : rotary_dim // 2]
    half_sin = sin[:, :, None, : rotary_dim // 2]

    def _rot(x):
        rotated = rope(x[..., :rotary_dim].transpose(1, 2), half_cos, half_sin).transpose(1, 2)
        return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)

    q_embed = _rot(q)
    if k is None:
        return q_embed
    return q_embed, _rot(k)


def _apply_indexer_query_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to indexer queries in plain torch.

    Fused rope variants (``npu_rotary_mul``, Triton FMA) round differently from
    the eager mul/add path; the difference can flip a TopK tie inside the
    routing operator, so the query rotation must stay bit-exact against the
    reference instead of going through the accelerated ``_apply_rotary_pos_emb``.
    """
    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2
    rotary, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated_half = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
    rotated = rotary * cos.unsqueeze(2) + rotated_half * sin.unsqueeze(2)
    return torch.cat((rotated, passthrough), dim=-1)


class Qwen4ExpTextRMSNorm(HFQwen4ExpTextRMSNorm):
    """RMSNorm on the ``fsdp_turbo.ops`` fused kernel (``npu_rms_norm``).

    Grouped RMSNorm (``group_size`` set) keeps the stock eager math since
    ``npu_rms_norm`` cannot replicate the per-group normalization.
    """

    def forward(self, x):
        if self.group_size is not None or x.device.type != "npu":
            return super().forward(x)
        return rms_norm(x, 1.0 + self.weight, self.eps).type_as(x)


class Qwen4ExpTextRMSNormGated(HFQwen4ExpTextRMSNormGated):
    """Gated RMSNorm on the ``fsdp_turbo.ops`` fused kernel (``npu_rms_norm``)."""

    def forward(self, hidden_states, gate=None):
        if hidden_states.device.type != "npu":
            return super().forward(hidden_states, gate)
        input_dtype = hidden_states.dtype
        hidden_states = rms_norm(hidden_states, self.weight, self.variance_epsilon)
        from transformers.activations import ACT2FN

        hidden_states = hidden_states * ACT2FN[self.activation](gate)
        return hidden_states.to(input_dtype)


class Qwen4ExpTextExperts(HFQwen4ExpTextExperts):
    """MoE experts on the fused NPU grouped-MatMul pipeline.

    Replaces the stock per-expert Python loop with the ``fsdp_turbo.ops``
    MoE pipeline (``permute`` → ``grouped_matmul`` → ``swiglu`` →
    ``grouped_matmul`` → ``unpermute``) when ``use_grouped_expert_matmul``
    is set on the config.
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_grouped_expert_matmul = bool(getattr(config, "use_grouped_expert_matmul", False))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_grouped_expert_matmul or hidden_states.device.type != "npu":
            return super().forward(hidden_states, top_k_index, top_k_weights)

        permuted_hidden, row_ids_map = permute(hidden_states, top_k_index.to(torch.int32))
        tokens_per_expert = torch.histc(top_k_index, bins=self.num_experts, min=0, max=self.num_experts)
        intermediate = grouped_matmul(permuted_hidden, tokens_per_expert, self.gate_up_proj)
        intermediate = swiglu(intermediate, dim=-1)
        output = grouped_matmul(intermediate, tokens_per_expert, self.down_proj)
        return unpermute(output, row_ids_map, probs=top_k_weights)


class Qwen4ExpTextQSAIndexer(HFQwen4ExpTextQSAIndexer):
    """QSA indexer on the backend-dispatched ``qsa_selected_token_mask`` operator.

    With ``qsa_implementation='triton'`` the fused operator replaces the stock
    per-query Python loop; any other value keeps the stock forward.
    """

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.qsa_implementation = str(getattr(config, "qsa_implementation", "eager")).lower().strip()
        self.q_layernorm = Qwen4ExpTextRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpTextRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        if self.qsa_implementation != "triton":
            return super().forward(hidden_states, position_embeddings, attention_mask, past_key_values)

        batch_size, sequence_length, _ = hidden_states.shape
        full_cos, full_sin = position_embeddings
        current_cos = full_cos[:, -sequence_length:, :]
        current_sin = full_sin[:, -sequence_length:, :]
        projected = self.index_qk_proj(hidden_states)
        query_width = self.index_n_heads * self.index_head_dim
        query, raw_keys = projected.split([query_width, self.index_kv_heads * self.index_head_dim], dim=-1)
        query = query.view(batch_size, sequence_length, self.index_n_heads, self.index_head_dim)
        raw_keys = raw_keys.view(batch_size, sequence_length, self.index_kv_heads, self.index_head_dim)
        if self.index_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA Triton fast path requires indexer_kv_heads == 1")
        raw_keys = raw_keys.squeeze(2)
        query = _apply_indexer_query_rope(self.q_layernorm(query), current_cos, current_sin)
        if past_key_values is not None:
            raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)
        return qsa_selected_token_mask(
            query,
            raw_keys,
            full_cos,
            full_sin,
            attention_mask,
            self.k_layernorm.weight,
            rms_norm_eps=self.k_layernorm.eps,
            token_budget=self.token_budget,
            compress_ratio=self.compress_ratio,
            query_chunk_size=getattr(self, "qsa_triton_query_chunk_size", None),
            validate_prefix_mask=getattr(self, "qsa_triton_validate_prefix_mask", True),
        )


class Qwen4ExpTextAttention(HFQwen4ExpTextAttention):
    """Full-attention head with fused q/k norms, the fused QSA indexer and NPU RoPE."""

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = Qwen4ExpTextQSAIndexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        selected_token_mask = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values)
        # Combine both masks (they are never None, and are always 4D with either bool for sdpa, or float for eager)
        if attention_mask.is_floating_point():
            attention_mask = attention_mask + selected_token_mask
        else:
            attention_mask = attention_mask & selected_token_mask

        # The cos/sin are the full positions here due to the indexer, so we need to slice to get current positions
        position_embeddings = (x[:, -hidden_states.shape[1] :, :] for x in position_embeddings)
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
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, _stock_eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen4ExpTextGatedDeltaNet(HFQwen4ExpTextGatedDeltaNet):
    """GatedDeltaNet dispatching causal-conv1d and GDN kernels to fsdp_turbo.ops.

    ``causal_conv1d_implementation`` / ``gdn_implementation`` /
    ``skip_gdn_recompute`` (set by the training entry point) select the
    accelerated path; 'eager' keeps the stock computation.
    """

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if not _IS_NPU:
            return
        self.causal_conv1d_implementation = (
            str(getattr(config, "causal_conv1d_implementation", "eager")).lower().strip()
        )
        self.gdn_implementation = str(getattr(config, "gdn_implementation", "eager")).lower().strip()
        self.skip_gdn_recompute = bool(getattr(config, "skip_gdn_recompute", False))
        # Same cross-impl downgrade rule as MindSpeed-MM: the ascendc+triton combo
        # is not supported here (no ascendc gdn op in this repo), so fall back.
        if self.gdn_implementation in ("ascendc", "ascendc_legacy") and self.causal_conv1d_implementation == "triton":
            self.causal_conv1d_implementation = "triton_with_transpose"
        # Unsupported ascendc variants degrade to eager (no ascendc gdn op here).
        if self.gdn_implementation in ("ascendc", "ascendc_legacy"):
            self.gdn_implementation = "eager"
        if self.causal_conv1d_implementation in ("ascendc",):
            self.causal_conv1d_implementation = "eager"
        if self.gdn_implementation == "eager" and self.skip_gdn_recompute:
            raise NotImplementedError(
                "gdn_implementation='eager' does not support skip_gdn_recompute; "
                "use gdn_implementation='triton' or disable skip_gdn_recompute."
            )
        print_rank(
            logger.info,
            f"Qwen4ExpTextGatedDeltaNet(layer={layer_idx}) "
            f"conv1d={self.causal_conv1d_implementation} gdn={self.gdn_implementation} "
            f"skip_recompute={self.skip_gdn_recompute}",
        )
        self.norm = Qwen4ExpTextRMSNormGated(
            self.head_v_dim, eps=self.layer_norm_epsilon, activation=config.output_gate_type or config.hidden_act
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if hidden_states.device.type != "npu":
            return super().forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
        from fsdp_turbo.ops.causal_conv1d import causal_conv1d as _ft_causal_conv1d
        from fsdp_turbo.ops.causal_conv1d import causal_conv1d_update as _ft_causal_conv1d_update
        from fsdp_turbo.ops.fla import chunk_gated_delta_rule as _ft_chunk_gdr
        from fsdp_turbo.ops.fla import fused_recurrent_gated_delta_rule as _ft_recurrent_gdr

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx, state_idx=0
        )

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        cu_seqlens = None
        if kwargs.get("cu_seq_lens_q") is not None:
            cu_seqlens = kwargs.get("cu_seq_lens_q").to(torch.int64)

        conv_impl = self.causal_conv1d_implementation
        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            if conv_impl == "triton":
                mixed_qkv = _ft_causal_conv1d_update(
                    mixed_qkv.transpose(1, 2),
                    conv_state,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            else:
                mixed_qkv = _stock_causal_conv1d_update(
                    mixed_qkv.transpose(1, 2),
                    conv_state,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.activation,
                ).transpose(1, 2)
        else:
            if cache_params is not None:
                mixed_qkv_t = mixed_qkv.transpose(1, 2)
                mixed_qkv_t = cache_params.update_conv_state(
                    mixed_qkv_t, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )
                mixed_qkv = mixed_qkv_t.transpose(1, 2)

            if conv_impl == "triton":
                mixed_qkv, _ = _ft_causal_conv1d(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1).transpose(-1, -2).contiguous(),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    cu_seqlens=cu_seqlens,
                )
            elif conv_impl == "triton_with_transpose":
                # Triton-with-transpose path (no contiguous transpose of weight);
                # the registered kernel handles the [b, s, H*d] layout directly.
                mixed_qkv, _ = _ft_causal_conv1d(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    cu_seqlens=cu_seqlens,
                )
            else:
                mixed_qkv = mixed_qkv.transpose(1, 2)
                mixed_qkv = _stock_causal_conv1d_fn(
                    mixed_qkv,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :seq_len, :] if mixed_qkv.dim() == 3 else mixed_qkv[..., :seq_len]

        if conv_impl in ("triton_with_transpose", "ascendc"):
            query, key, value = torch.split(
                mixed_qkv,
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                dim=1,
            )
            query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
            key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
            value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [self.key_dim, self.key_dim, self.value_dim],
                dim=-1,
            )
            query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
            value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            if conv_impl in ("triton_with_transpose", "ascendc"):
                query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
                key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
            else:
                query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
                key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = _ft_recurrent_gdr(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        elif self.gdn_implementation == "triton":
            core_attn_out, last_recurrent_state = _ft_chunk_gdr(
                query,
                key,
                value,
                g=g,
                beta=beta,
                cu_seqlens=cu_seqlens,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                skip_recompute=self.skip_gdn_recompute,
            )
        else:
            # eager fallback: same call as triton but without skip_recompute semantics
            core_attn_out, last_recurrent_state = _ft_chunk_gdr(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


class Qwen4ExpTextSparseMoeBlock(HFQwen4ExpTextSparseMoeBlock):
    """Sparse MoE block whose experts run on the fused grouped-MatMul pipeline."""

    def __init__(self, config):
        # Replicates the stock constructor with the fused experts class; calling
        # the stock parent __init__ would transiently allocate the expert
        # weights twice.
        nn.Module.__init__(self)
        self.gate = Qwen4ExpTextTopKRouter(config)
        self.experts = Qwen4ExpTextExperts(config)
        self.shared_expert = Qwen4ExpTextMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)


def _save_ngram_dims(module, config=None):
    """Capture vocab/dim before the embedding is replaced with size-0."""
    import math as _math

    if config is not None:
        ngram_vocab_divisor = getattr(config, "make_ngram_vocab_size_divisible_by", 128)
    else:
        ngram_vocab_divisor = 128
    head_dim_per_ngram = (
        module.ngram_head_dim_per_ngram
        if hasattr(module, "ngram_head_dim_per_ngram")
        else module.ngram_embedding.embedding_dim
    )
    total_vocab = getattr(module, "total_vocab_size", 0)
    padded = _math.ceil(total_vocab / ngram_vocab_divisor) * ngram_vocab_divisor if total_vocab else 0
    module.ngram_padded_vocab_size = padded
    module.ngram_head_dim_per_ngram = head_dim_per_ngram


class Qwen4ExpTextNGramEmbeddingCpu(Qwen4ExpTextNGramEmbedding):
    """PLE n-gram embedding that can stay resident on host memory.

    With ``config.ple_embedding_on_cpu`` (and vocab parallelism disabled) the
    table is created with zero rows here and materialized on CPU afterwards by
    ``materialize_ngram_embedding_on_cpu``, so the ~51B-parameter table never
    occupies accelerator memory.  Under vocab parallelism the flag is ignored
    and the stock device-resident table is kept.
    """

    def __init__(self, config: Qwen4ExpTextConfig, embedding_dim: int, layer_idx: int, ple_layer_index: int = 0):
        super().__init__(config, embedding_dim, layer_idx, ple_layer_index)
        ple_on_cpu = getattr(config, "ple_embedding_on_cpu", False)
        if ple_on_cpu and get_parallel_state().is_group_enable("vp"):
            ple_on_cpu = False
        self.ple_embedding_on_cpu = ple_on_cpu
        self._ngram_hf_dir = getattr(config, "_ngram_hf_dir", None)
        if ple_on_cpu:
            _save_ngram_dims(self, config)
            self.ngram_embedding = nn.Embedding(0, self.ngram_head_dim_per_ngram)


class Qwen4ExpTextPLELayer(HFQwen4ExpTextPLELayer):
    """PLE layer whose n-gram embedding supports the CPU-offload variant."""

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int, ple_layer_index: int):
        super().__init__(config, layer_idx, ple_layer_index)
        self.ple_embedding = Qwen4ExpTextNGramEmbeddingCpu(config, config.ple_embed_dim, layer_idx, ple_layer_index)


class Qwen4ExpTextDecoderLayer(HFQwen4ExpTextDecoderLayer):
    """Decoder layer assembling the fused attention/GDN, MoE and PLE modules."""

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        # Replicates the stock constructor with the fused submodule classes;
        # the stock parent __init__ would build the full stock stack first.
        nn.Module.__init__(self)
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpTextGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen4ExpTextAttention(config, layer_idx)
        self.mlp = Qwen4ExpTextSparseMoeBlock(config)
        ple_layer_index = config.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in config.ple_layer_ids else None
        self.ple = Qwen4ExpTextPLELayer(config, layer_idx, ple_layer_index) if ple_layer_index is not None else None
        self.attn_hyper_connection = Qwen4ExpTextGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config)


class Qwen4ExpTextModel(HFQwen4ExpTextModel):
    """Language backbone assembling fused decoder layers."""

    def __init__(self, config: Qwen4ExpTextConfig):
        # Replicates the stock constructor with the fused decoder layer;
        # the stock parent __init__ would build the full stock layer stack.
        Qwen4ExpPreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen4ExpTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Qwen4ExpTextRotaryEmbedding(config=config)
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen4ExpModel(HFQwen4ExpModel):
    """Multimodal wrapper with the post-fusion Ulysses forward and fused language model.

    Image tokens expand the sequence, so Ulysses CP sharding happens *after*
    multimodal fusion (mirroring the Qwen3.5 and Kimi-K3 patterns).
    """

    def __init__(self, config):
        # Replicates the stock constructor with the fused language model;
        # the stock parent __init__ would build the full stock backbone.
        Qwen4ExpPreTrainedModel.__init__(self, config)
        self.visual = AutoModel.from_config(config.vision_config)
        self.language_model = Qwen4ExpTextModel(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        """Fuse full multimodal inputs, then shard only the language-model sequence."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        ulysses_group, _, ulysses_size = get_ulysses_context()
        if ulysses_size > 1 and (past_key_values is not None or cache_position is not None):
            raise ValueError("Post-fusion Ulysses CP for Qwen3.8-Flash-Next does not support cache inputs.")
        if ulysses_size > 1 and kwargs.get("output_router_logits", False):
            raise ValueError(
                "Post-fusion Ulysses CP does not yet support output_router_logits; "
                "the router auxiliary-loss mask remains full sequence."
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        if ulysses_size > 1:
            inputs_embeds, position_ids, attention_mask = shard_fused_sequence_inputs(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                group=ulysses_group,
            )

        ple_input_ids = None
        if input_ids is not None:
            ple_input_ids = input_ids
            if ulysses_size > 1:
                ple_input_ids = split_tensor_to_group(
                    pad_tensor_to_group(ple_input_ids, ulysses_group, dim=1, pad_value=0),
                    ulysses_group,
                    dim=1,
                )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            ple_input_ids=ple_input_ids,
            **kwargs,
        )

        return Qwen4ExpModelOutputWithPast(**outputs, rope_deltas=self.rope_deltas)


class Qwen4ExpForConditionalGeneration(HFQwen4ExpForConditionalGeneration):
    """Qwen4ExpForConditionalGeneration with fused modules and MTP support.

    Stock transformers has no MTP for qwen4_exp; the MTP block,
    ``_compute_mtp_loss`` and the forward wiring mirror the Qwen3.5-MoE
    reference (modeling_qwen3_5_moe.py:3079) adapted to this layout
    (``self.model.language_model`` holds ``embed_tokens`` / ``rotary_emb``).
    """

    def __init__(self, config):
        # Replicates the stock constructor with the fused multimodal wrapper;
        # the stock parent __init__ would build the full stock model.
        Qwen4ExpPreTrainedModel.__init__(self, config)
        self.model = Qwen4ExpModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

        text_config = getattr(config, "text_config", config)
        mtp_num_layers = getattr(text_config, "mtp_num_layers", 0)
        self.enable_mtp = bool(mtp_num_layers)
        self.mtp = (
            MultiTokenPredictionBlock(
                text_config,
                Qwen4ExpTextDecoderLayer,
                Qwen4ExpTextRMSNorm,
                hyper_connection_mixer_cls=Qwen4ExpTextGatedResidual,
            )
            if self.enable_mtp
            else None
        )
        if self.enable_mtp:
            print_rank(logger.info, f"MTP enabled: mtp_num_layers={mtp_num_layers}")

    def _compute_mtp_loss(self, hidden_states, **kwargs):
        if not self.enable_mtp:
            return None
        # Inject the parallel context: MultiTokenPredictionBlock (fsdp_turbo.modules.mtp)
        # must stay free of fsdp_turbo.distributed imports, so the Ulysses and
        # CP group contexts are resolved here and passed in via forward kwargs.
        parallel_state = get_parallel_state()
        mtp_loss = self.mtp(
            hidden_states,
            embed_tokens=self.model.language_model.embed_tokens,
            rotary_emb=self.model.language_model.rotary_emb,
            output_layer=self.lm_head,
            loss_function=getattr(self, "mtp_loss_function", None) or self.loss_function,
            ulysses_context=get_ulysses_context(),
            cp_context=(
                (parallel_state.get_cp_group(), parallel_state.get_cp_group_size())
                if parallel_state.is_group_enable("cp")
                else None
            ),
            **kwargs,
        )
        return mtp_loss

    def forward(self, *args, **kwargs):
        # Force hidden_states output so _compute_mtp_loss can access the last
        # layer's hidden state. Stock forward only returns hidden_states when
        # output_hidden_states=True; without this the MTP early-returns None.
        kwargs["output_hidden_states"] = True

        # Fallback: capture last_hidden_state via a forward hook on the
        # language_model. The transformers 5.x @capture_outputs mechanism
        # silently returns None under FSDP2 + recompute (the _can_record_outputs
        # registry / decoder-layer hooks fail to fire), so output.hidden_states
        # ends up None and MTP never runs. Capturing last_hidden_state directly
        # bypasses capture_outputs entirely.
        captured = {"last_hidden_state": None}

        def _capture_last_hs(module, inputs, output):
            if output is not None:
                try:
                    captured["last_hidden_state"] = output[0]
                except (IndexError, KeyError, TypeError):
                    captured["last_hidden_state"] = None

        language_model = self.model.language_model
        handle = language_model.register_forward_hook(_capture_last_hs)
        try:
            result = super().forward(*args, **kwargs)
        finally:
            handle.remove()

        # can_return_tuple may return a plain tuple or the dataclass; normalize.
        if isinstance(result, Qwen4ExpCausalLMOutputWithPast):
            output = result
        elif isinstance(result, tuple):
            output = Qwen4ExpCausalLMOutputWithPast(*result)
        else:
            return result

        if not self.enable_mtp:
            return result

        # Prefer output.hidden_states[-1] (full all-layer tuple); fall back to
        # the hook-captured last_hidden_state when capture_outputs failed.
        hidden_states = output.hidden_states[-1] if output.hidden_states else None
        if hidden_states is None:
            hidden_states = captured["last_hidden_state"]
        if hidden_states is None:
            print_rank(
                logger.warning,
                "[MTP] hidden_states is None after forward; skipping MTP loss. "
                "Check capture_outputs / output_hidden_states propagation.",
            )
            return result

        mtp_loss = self._compute_mtp_loss(
            hidden_states,
            input_ids=kwargs.get("input_ids"),
            labels=kwargs.get("labels"),
            attention_mask=kwargs.get("attention_mask"),
            position_ids=kwargs.get("position_ids"),
            past_key_values=output.past_key_values,
            cache_position=kwargs.get("cache_position"),
            logits_to_keep=kwargs.get("logits_to_keep", 0),
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in (
                    "input_ids",
                    "labels",
                    "attention_mask",
                    "position_ids",
                    "cache_position",
                    "logits_to_keep",
                )
            },
        )
        output.mtp_loss = mtp_loss
        # Also stash on the module instance: FSDP2's _post_forward hook
        # rebuilds the output dataclass via dataclasses.replace() when
        # output_dtype is set, dropping non-field attributes like mtp_loss.
        # train_step reads self._last_mtp_loss to recover it.
        self._last_mtp_loss = mtp_loss
        return output

    @torch.no_grad()
    def _init_weights(self, module):
        """Stock _init_weights with the initializer_range fix for multimodal configs.

        Stock Qwen4Exp._init_weights uses ``self.config.initializer_range`` for
        experts/gate init, but the top-level ``Qwen4ExpConfig`` (multimodal) does
        not carry that attribute — it lives on ``self.config.text_config``. This
        only surfaces when MTP layers are present: their experts are MISSING from
        the checkpoint and must be initialized, while the language model's experts
        are loaded from the checkpoint and skip _init_weights entirely.

        Mirrors MindSpeed-MM Qwen3.5-MoE's getattr(self.config, "text_config", ...)
        compatibility pattern (modeling_qwen3_5_moe.py:1558-1563).
        """
        text_config = getattr(self.config, "text_config", self.config)
        initializer_range = getattr(text_config, "initializer_range", 0.02)
        # Temporarily expose initializer_range on the top-level config so the
        # stock _init_weights can find it; text_config retains the original value.
        self.config.initializer_range = initializer_range
        try:
            super()._init_weights(module)
        finally:
            # Restore: delete the temporary attribute so the top-level config
            # stays clean after init (text_config still has the real value).
            if not hasattr(type(self.config), "initializer_range"):
                try:
                    del self.config.initializer_range
                except AttributeError:
                    pass


def _locate_ngram_shards(hf_dir: str, layer_idx: int) -> dict | None:
    """Locate the real ngram embedding shards of *layer_idx* under *hf_dir*.

    Returns ``{filename: [shard_key, ...]}`` when a complete, materialized
    (non-LFS-pointer) shard set exists, otherwise ``None``.
    """
    import json

    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
    except (json.JSONDecodeError, KeyError):
        return None

    prefix = f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding.shard_"
    shard_keys = [key for key in weight_map if key.startswith(prefix)]
    if not shard_keys:
        return None

    by_file = {}
    for key in shard_keys:
        fname = weight_map[key]
        fpath = os.path.join(hf_dir, fname)
        try:
            if os.path.getsize(fpath) < 1024:
                return None
        except OSError:
            return None
        by_file.setdefault(fname, []).append(key)
    return by_file


_node_group_cache = None


def _get_node_local_group():
    """Get or create a process group containing only ranks on the same node.

    ``/dev/shm`` is node-local, so ngram table sharing must be scoped to the
    current node.  ``dist.new_group`` is a collective on the parent (WORLD)
    group: **every** rank must call it in the **same order** with the **same**
    sequence of rank lists, so we iterate over all nodes and create a group
    for each.  Ranks not in a given list participate in the barrier but do
    not become members of that group.
    """
    global _node_group_cache
    if _node_group_cache is not None:
        return _node_group_cache

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    if local_world_size <= 1:
        return None

    world_size = dist.get_world_size()
    num_nodes = world_size // local_world_size
    rank = dist.get_rank()

    my_group = None
    for node_id in range(num_nodes):
        node_ranks = list(range(node_id * local_world_size, (node_id + 1) * local_world_size))
        grp = dist.new_group(node_ranks)
        if rank in node_ranks:
            my_group = grp

    _node_group_cache = my_group
    return my_group


def _get_group_src_rank(group=None) -> int:
    """Return the global rank of group-rank 0 for ``dist.broadcast``.

    ``dist.broadcast`` expects ``src`` as a **global** rank, not a group rank.
    For the global group (``group=None``) this is 0; for a sub-group it is the
    first global rank in that group.
    """
    import torch.distributed as dist

    if group is None:
        return 0
    try:
        return dist.get_global_rank(group, 0)
    except (AttributeError, RuntimeError):
        rank = dist.get_rank()
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        node_id = rank // local_world_size if local_world_size > 1 else 0
        return node_id * local_world_size


def _broadcast_ngram_table(module, group=None) -> None:
    """Broadcast the source rank's ngram table to all ranks in *group* (in-place, chunked).

    HCCL cannot broadcast CPU tensors, so each chunk is staged through the
    accelerator: cpu -> accelerator -> broadcast -> cpu.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return

    src = _get_group_src_rank(group)
    rows = module.ngram_padded_vocab_size
    cols = module.ngram_head_dim_per_ngram
    chunk_rows = max(1, (1 << 21) // cols)
    weight = module.ngram_embedding.weight.data
    stage_device = torch.accelerator.current_accelerator() if _IS_NPU else None
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        if stage_device is not None:
            stage = weight[start:end].to(stage_device)
            dist.broadcast(stage, src=src, group=group)
            weight[start:end] = stage.cpu()
            del stage
        else:
            dist.broadcast(weight[start:end], src=src, group=group)
    dist.barrier(group=group)


def _share_ngram_table(module, group=None) -> None:
    """Share the ngram table within the same node via /dev/shm mmap.

    Each node's local-rank-0 saves its already-materialized table to
    ``/dev/shm``; other ranks on the **same node** load it with
    ``torch.load(mmap=True)``.  The OS page cache guarantees all ranks
    on a node map the same physical pages, so a 102 GB table consumes
    102 GB of host RAM per node regardless of local world size.

    Because ``/dev/shm`` is node-local tmpfs, this works transparently
    across multi-node setups — each node independently loads from its
    own local-rank-0 and shares within the node.  No cross-node table
    transfer is needed.

    Falls back to ``_broadcast_ngram_table`` (staged through the
    accelerator) when ``/dev/shm`` is too small or mmap loading fails.
    """
    import shutil

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return

    node_group = _get_node_local_group()
    if node_group is None:
        return

    node_world_size = dist.get_world_size(group=node_group)
    if node_world_size == 1:
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    src = _get_group_src_rank(node_group)
    rows = module.ngram_padded_vocab_size
    cols = module.ngram_head_dim_per_ngram
    weight = module.ngram_embedding.weight.data
    nbytes = rows * cols * weight.element_size()

    shm_dir = "/dev/shm"  # nosec B108
    shm_path = f"{shm_dir}/fsdp_turbo_ngram_layer{module.layer_idx}_{os.getppid()}.pt"

    # Local rank 0 attempts to save the table to /dev/shm; communicates
    # the path (or None on failure) to other ranks on the same node so
    # all ranks agree on whether to use shm or fall back to broadcast.
    shm_path_list = [None]
    if local_rank == 0:
        try:
            if not os.path.isdir(shm_dir):
                raise OSError(f"{shm_dir} not found")
            _, _, free = shutil.disk_usage(shm_dir)
            if free < nbytes * 1.05:
                raise OSError(f"{shm_dir} free {free / 1e9:.0f}GB < table {nbytes / 1e9:.0f}GB")
            print_rank(
                logger.info,
                f"Sharing ngram table ({nbytes / 1e9:.0f}GB) via {shm_path} (local_rank=0, global_rank={dist.get_rank()})",
            )
            torch.save(weight.contiguous().cpu(), shm_path)
            shm_path_list = [shm_path]
        except (OSError, RuntimeError, ValueError) as e:
            print_rank(logger.warning, f"_share_ngram_table /dev/shm failed ({e})")
            shm_path_list = [None]

    dist.broadcast_object_list(shm_path_list, src=src, group=node_group)
    actual_shm_path = shm_path_list[0]

    if actual_shm_path is not None:
        if local_rank != 0:
            shared_weight = torch.load(actual_shm_path, mmap=True, weights_only=True)
            if shared_weight.dtype != weight.dtype:
                shared_weight = shared_weight.to(weight.dtype)
            module.ngram_embedding = nn.Embedding(
                rows,
                cols,
                dtype=weight.dtype,
                device="cpu",
                _weight=shared_weight,
            )

        dist.barrier(group=node_group)

        if local_rank == 0:
            try:
                os.unlink(actual_shm_path)
            except OSError:
                pass
    else:
        print_rank(logger.warning, "Falling back to node-local broadcast for ngram table")
        _broadcast_ngram_table(module, group=node_group)


def materialize_ngram_embedding_on_cpu(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    hf_dir: str | None = None,
    strict_weights: bool = False,
) -> None:
    """Materialize all CPU-resident ngram embedding tables of a Qwen4Exp model.

    Must be called after meta-device materialization so the full table is only
    allocated on host memory and never occupies accelerator memory.  Real
    shards are streamed from rank 0 and shared via /dev/shm mmap (page-cache
    shared, so world_size copies cost the same as one); otherwise all ranks
    generate identical random tables on rank 0 and share the same way.
    """
    import torch.distributed as dist

    if isinstance(model, Qwen4ExpTextNGramEmbedding):
        modules = [model]
    else:
        modules = [m for m in model.modules() if isinstance(m, Qwen4ExpTextNGramEmbedding)]

    group = None
    is_dist = dist.is_available() and dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_dist else 0

    for module in modules:
        if not getattr(module, "ple_embedding_on_cpu", False):
            continue
        hf_dir_eff = getattr(module, "_ngram_hf_dir", None) or hf_dir
        by_file = _locate_ngram_shards(hf_dir_eff, module.layer_idx) if hf_dir_eff else None
        use_real = by_file is not None
        if not use_real and strict_weights:
            raise FileNotFoundError(
                f"[ngram] no readable ngram embedding shards found for layer {module.layer_idx} "
                f"under '{hf_dir_eff}'. Refusing to fall back to random init."
            )

        size_gb = module.ngram_padded_vocab_size * module.ngram_head_dim_per_ngram * 2 / 1e9
        if use_real:
            if is_dist and local_rank != 0:
                _share_ngram_table(module, group)
                module.ngram_embedding.weight.requires_grad_(False)
                continue
            print_rank(
                logger.info,
                f"Materializing ngram embedding (layer {module.layer_idx}, "
                f"{module.ngram_padded_vocab_size} rows, ~{size_gb:.0f}GB) on CPU from checkpoint shards",
            )
            _materialize_on_cpu(module, dtype=dtype, by_file=by_file, hf_dir=hf_dir_eff)
            if is_dist:
                _share_ngram_table(module, group)
        else:
            print_rank(
                logger.info,
                f"Materializing ngram embedding (layer {module.layer_idx}, "
                f"{module.ngram_padded_vocab_size} rows, ~{size_gb:.0f}GB) on CPU with random init",
            )
            _materialize_on_cpu(module, dtype=dtype)
            if is_dist:
                _share_ngram_table(module, group)
        module.ngram_embedding.weight.requires_grad_(False)


def _materialize_on_cpu(module, dtype=torch.bfloat16, by_file=None, hf_dir=None):
    """Allocate the full CPU table and fill it from shards or random init."""
    if module.ngram_embedding.weight.shape[0] != 0:
        return
    if by_file is not None:
        weight = _load_shards_by_file(module, by_file, dtype, hf_dir)
    else:
        weight = _fast_random_init(module, dtype)
    module.ngram_embedding = nn.Embedding(
        module.ngram_padded_vocab_size,
        module.ngram_head_dim_per_ngram,
        dtype=dtype,
        device="cpu",
        _weight=weight,
    )


def _load_shards_by_file(module, by_file, dtype, hf_dir):
    """Stream real ngram weights from safetensors shards on rank 0."""
    from safetensors import safe_open

    rows, cols = module.ngram_padded_vocab_size, module.ngram_head_dim_per_ngram
    weight = torch.empty(rows, cols, dtype=dtype, device="cpu")
    total_shards = sum(len(keys) for keys in by_file.values())
    rows_per_shard = (rows + total_shards - 1) // total_shards

    for fname, keys in by_file.items():
        fpath = os.path.join(hf_dir, fname)
        with safe_open(fpath, framework="pt") as f:
            for key in keys:
                shard_id = int(key.rsplit(".", 2)[-2].split("_")[-1])
                tensor = f.get_tensor(key)
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype)
                dst_start = shard_id * rows_per_shard
                dst_end = (shard_id + 1) * rows_per_shard
                if dst_start >= rows:
                    continue
                weight[dst_start:dst_end] = tensor[: rows - dst_start]
    return weight


def _fast_random_init(module, dtype=torch.bfloat16):
    """Deterministic coarse random init, fine for functional tests only."""
    rows, cols = module.ngram_padded_vocab_size, module.ngram_head_dim_per_ngram
    weight = torch.empty(rows, cols, dtype=dtype, device="cpu")
    device = torch.accelerator.current_accelerator() if _IS_NPU else "cpu"
    chunk_rows = max(1, (1 << 21) // cols * 16)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        torch.manual_seed(module.layer_idx * 100003 + start)
        block = torch.randn(end - start, cols, dtype=torch.float32, device=device)
        weight[start:end] = block.to(dtype).cpu()
        del block
    return weight


_NGRAM_INT_BUFFERS = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")


def restore_ngram_integer_buffers(model: nn.Module) -> None:
    """Restore ``torch.long`` dtype on PLE n-gram embedding integer buffers.

    ``from_pretrained(torch_dtype=...)`` and ``convert_model_dtype`` cast *all*
    parameters and buffers to the target float dtype.  The PLE hashing buffers
    (``layer_multipliers``, ``ngram_heads_vocab_sizes``, ``ngram_heads_offsets``)
    must remain ``torch.long`` because they participate in ``torch.bitwise_xor``
    and modular arithmetic.  On NPU, ``aclnnBitwiseXorTensor`` only accepts
    integer dtypes, so leaving them cast to bfloat16 raises a RuntimeError at
    training time.

    Call after ``convert_model_dtype`` (or ``from_pretrained`` with
    ``torch_dtype``) to undo the unwanted cast on these specific buffers.
    """
    for module in model.modules():
        if not isinstance(module, Qwen4ExpTextNGramEmbedding):
            continue
        for name in _NGRAM_INT_BUFFERS:
            buf = getattr(module, name, None)
            if buf is not None and buf.dtype != torch.long:
                buf.data = buf.data.to(torch.long)


__all__ = [
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpModel",
    "Qwen4ExpTextModel",
    "Qwen4ExpTextDecoderLayer",
    "Qwen4ExpTextAttention",
    "Qwen4ExpTextQSAIndexer",
    "Qwen4ExpTextGatedDeltaNet",
    "Qwen4ExpTextSparseMoeBlock",
    "Qwen4ExpTextExperts",
    "Qwen4ExpTextRMSNorm",
    "Qwen4ExpTextRMSNormGated",
    "Qwen4ExpTextNGramEmbeddingCpu",
    "Qwen4ExpTextPLELayer",
    "materialize_ngram_embedding_on_cpu",
    "restore_ngram_integer_buffers",
]

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2026 tencent inc. all rights reserved.
#
# This file is adapted from the DeepSeek-V4 reference model implementation
# (DeepSeek-AI official release) for FSDPTurbo training: dense-attention and
# FSDPTurbo adaptations (expert-parallel dispatch, reduced config, hash-MoE
# buffers).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at LICENSE (Apache-2.0) in the repo root.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeek-V4 adaptation for transformers >= 5.8.1.

Overrides expert execution and NPU attention via ``indexed_sparse_attention``.
Compressor subclasses expose sparse indices while preserving the stock mask/cache
interface; other model components retain the transformers implementation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.distributed._functional_collectives import all_to_all_single_autograd
from torch.distributed.tensor import DTensor

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention as HFDeepseekV4Attention,
    DeepseekV4CSACompressor as HFDeepseekV4CSACompressor,
    DeepseekV4HCACompressor as HFDeepseekV4HCACompressor,
    DeepseekV4DecoderLayer as HFDeepseekV4DecoderLayer,
    DeepseekV4Experts as HFDeepseekV4Experts,
    DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM,
    DeepseekV4Model as HFDeepseekV4Model,
    DeepseekV4SparseMoeBlock as HFDeepseekV4SparseMoeBlock,
    apply_rotary_pos_emb,
)

from fsdp_turbo.distributed.expert_parallel.dispatcher import (
    alltoall_combine,
    alltoall_dispatch,
    dispatch_preprocess,
)
from fsdp_turbo.distributed.expert_parallel.utils import (
    fixed_router_for_debug,
    normalize_expert_args,
)
from fsdp_turbo.ops.moe import grouped_matmul
from fsdp_turbo.ops.sparse_attention import indexed_sparse_attention
from fsdp_turbo.utils.env import sparse_flash_mla_validation_enabled
from .context_parallel import deepseek_v4_cp_attention_forward


def _sliding_window_indices(
    batch_size: int,
    sequence_length: int,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(sequence_length, device=device).view(-1, 1)
    offsets = torch.arange(window_size, device=device).view(1, -1)
    indices = query_positions - window_size + 1 + offsets
    indices = indices.masked_fill(indices < 0, -1)
    return indices.unsqueeze(0).expand(batch_size, -1, -1).to(torch.int32)


def _compress_training_key_value(
    compressor: HFDeepseekV4HCACompressor | HFDeepseekV4CSACompressor,
    hidden_states: torch.Tensor,
    *,
    overlap: bool,
) -> torch.Tensor:
    """Stock no-cache pooling and RoPE, without allocating a compressed attention mask."""
    batch_size, sequence_length, _ = hidden_states.shape
    ratio = compressor.compress_rate
    compressed_length = sequence_length // ratio
    usable = compressed_length * ratio
    key_value = compressor.kv_proj(hidden_states)[:, :usable]
    gate = compressor.gate_proj(hidden_states)[:, :usable]
    if compressed_length == 0:
        return key_value.new_zeros((batch_size, 1, 0, compressor.head_dim))
    key_value = key_value.view(batch_size, compressed_length, ratio, -1)
    gate = gate.view(batch_size, compressed_length, ratio, -1) + compressor.position_bias.to(gate.dtype)
    if overlap:
        # CSA pools the current Cb half with the preceding window's Ca half.
        overlap_shape = (batch_size, compressed_length, 2 * ratio, compressor.head_dim)
        overlap_key_value = key_value.new_zeros(overlap_shape)
        overlap_gate = gate.new_full(overlap_shape, float("-inf"))
        overlap_key_value[:, :, ratio:] = key_value[..., compressor.head_dim :]
        overlap_gate[:, :, ratio:] = gate[..., compressor.head_dim :]
        overlap_key_value[:, 1:, :ratio] = key_value[:, :-1, :, : compressor.head_dim]
        overlap_gate[:, 1:, :ratio] = gate[:, :-1, :, : compressor.head_dim]
        key_value, gate = overlap_key_value, overlap_gate
    compressed = compressor.kv_norm(
        (key_value * gate.softmax(dim=2, dtype=torch.float32).to(key_value.dtype)).sum(dim=2)
    )
    positions = (torch.arange(compressed_length, device=compressed.device) * ratio).unsqueeze(0)
    cos, sin = compressor.rotary_emb(
        compressed, position_ids=positions.expand(batch_size, -1), layer_type=compressor.rope_layer_type
    )
    return apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin)


class DeepseekV4HCACompressor(HFDeepseekV4HCACompressor):
    """Stock HCA with a mask-free full-sequence training output for sparse attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: object | None,
        layer_idx: int,
        *,
        return_indices: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return stock KV/mask, or KV/None for implicit HCA causality without a cache."""
        if not return_indices:
            return super().forward(hidden_states, q_residual, position_ids, past_key_values, layer_idx)
        if past_key_values is not None:
            raise ValueError("Sparse compressor indices require full-sequence training without KV cache.")
        return _compress_training_key_value(self, hidden_states, overlap=False), None


class DeepseekV4CSACompressor(HFDeepseekV4CSACompressor):
    """Stock CSA with direct indexer output for full-sequence sparse attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: object | None,
        layer_idx: int,
        *,
        return_indices: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stock KV/mask, or KV/int32 indices with -1 marking invisible entries."""
        if not return_indices:
            return super().forward(hidden_states, q_residual, position_ids, past_key_values, layer_idx)
        if past_key_values is not None:
            raise ValueError("Sparse compressor indices require full-sequence training without KV cache.")
        compressed = _compress_training_key_value(self, hidden_states, overlap=True)
        indices = self.indexer(hidden_states, q_residual, position_ids, None, layer_idx)
        return compressed, indices.to(torch.int32)


def _deepseek_v4_sparse_attention(
    module: nn.Module,
    query: torch.Tensor,
    key_value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    *,
    compressed_key_value: torch.Tensor | None = None,
    compressed_topk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, None]:
    """Pass separate raw/compressed KV and indexer output to SparseFlashMla."""
    if dropout:
        raise NotImplementedError("DeepSeek-V4 sparse attention requires attention_dropout=0.")
    batch_size, _, sequence_length, head_dim = query.shape
    if key_value.shape != (batch_size, 1, sequence_length, head_dim):
        raise ValueError("DeepSeek-V4 sparse attention requires full-sequence shared KV with shape [B,1,S,D].")
    if attention_mask is None or not attention_mask.is_floating_point():
        raise ValueError("DeepSeek-V4 sparse attention requires an additive sliding-window mask.")
    if (
        attention_mask.ndim != 4
        or attention_mask.shape[0] not in (1, batch_size)
        or attention_mask.shape[1:] != (1, sequence_length, sequence_length)
    ):
        raise ValueError("DeepSeek-V4 sparse attention requires a head-independent [B,1,S,S] sliding-window mask.")
    # Full-sequence, unpadded/unpacked input is a caller contract. Content checks
    # allocate quadratic tensors and synchronize the device; enable only for debugging.
    if sparse_flash_mla_validation_enabled():
        mask = attention_mask.expand(batch_size, -1, -1, -1).squeeze(1)
        allowed = mask == 0
        blocked = torch.isneginf(mask) | (mask == torch.finfo(mask.dtype).min)
        if not bool((allowed | blocked).all()):
            raise NotImplementedError("DeepSeek-V4 sparse attention does not support additive sliding-window bias.")
        positions = torch.arange(sequence_length, device=query.device)
        expected_original = (positions[None, :] <= positions[:, None]) & (
            positions[None, :] > positions[:, None] - module.sliding_window
        )
        if not torch.equal(allowed, expected_original.unsqueeze(0).expand(batch_size, -1, -1)):
            raise NotImplementedError("DeepSeek-V4 sparse attention requires unpadded, unpacked microbatches.")

    compressor = module.compressor
    compress_ratio = compressor.compress_rate if compressor is not None else 1
    if module.layer_type == "compressed_sparse_attention":
        if compress_ratio != 4:
            raise NotImplementedError("CANN CSA sparse attention requires compress_rate=4.")
        if compressed_topk_indices is None or compressed_topk_indices.shape[-1] > 1024:
            raise ValueError("CSA requires compressed indices within the CANN top-k limit.")
    elif compressed_topk_indices is not None:
        raise ValueError("HCA and sliding attention require compressed_topk_indices=None.")
    if compressed_key_value is not None:
        compressed_key_value = (
            compressed_key_value[:, 0].to(dtype=query.dtype).contiguous() if compressed_key_value.shape[2] else None
        )
        if compressed_key_value is None:
            compressed_topk_indices = None
    output = indexed_sparse_attention(
        query.transpose(1, 2).contiguous(),
        key_value[:, 0].to(dtype=query.dtype).contiguous(),
        module.sinks.float().contiguous(),
        _sliding_window_indices(batch_size, sequence_length, module.sliding_window, query.device),
        scaling,
        use_sparse_flash_attn=True,
        compressed_key_value=compressed_key_value,
        compressed_topk_indices=compressed_topk_indices,
        compress_ratio=compress_ratio if compressed_key_value is not None else 1,
        window_size=module.sliding_window,
    )
    return output, None


class DeepseekV4Attention(HFDeepseekV4Attention):
    """Stock DeepSeek-V4 attention with the shared NPU SparseFlashMla backend."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        if self.layer_type == "compressed_sparse_attention":
            self.compressor = DeepseekV4CSACompressor(config)
        elif self.layer_type == "heavily_compressed_attention":
            self.compressor = DeepseekV4HCACompressor(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: object,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: object | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden_states.device.type != "npu" or not getattr(self.config, "use_sparse_flash_attn", False):
            return super().forward(
                hidden_states,
                position_embeddings,
                position_ids,
                attention_mask,
                past_key_values,
                **kwargs,
            )
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            raise ValueError("DeepSeek-V4 sparse-attention training does not support non-empty KV cache inputs.")

        expected_positions = torch.arange(hidden_states.shape[1], device=position_ids.device)
        if not torch.equal(position_ids, expected_positions.unsqueeze(0).expand_as(position_ids)):
            raise NotImplementedError("DeepSeek-V4 sparse attention requires contiguous positions starting at zero.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]
        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        query = apply_rotary_pos_emb(self.q_b_norm(query), cos, sin)
        key_value = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        key_value = apply_rotary_pos_emb(key_value, cos, sin)

        compressed_key_value = compressed_indices = None
        if self.compressor is not None:
            compressed_key_value, compressed_indices = self.compressor(
                hidden_states, q_residual, position_ids, None, self.layer_idx, return_indices=True
            )

        attention_output, attention_weights = _deepseek_v4_sparse_attention(
            self,
            query,
            key_value,
            attention_mask,
            self.scaling,
            0.0 if not self.training else self.attention_dropout,
            compressed_key_value=compressed_key_value,
            compressed_topk_indices=compressed_indices,
        )
        attention_output = apply_rotary_pos_emb(attention_output.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = attention_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped), attention_weights


class DeepseekV4Experts(HFDeepseekV4Experts):
    """Routed experts with the FSDPTurbo expert-parallel execution path.

    The stock ``forward`` keeps the pure-PyTorch fallback (per-expert ``F.linear``
    loop) and is used whenever no dispatcher replaces this module's
    ``forward``. Under FSDPTurbo, ``expert_parallelize_modules`` re-points
    ``forward`` at the configured dispatcher, which calls
    :meth:`native_eager_forward` with the FSDPTurbo EP group. The expert
    weights keep the stock layout (``gate_up_proj`` ``[E, 2*I, H]``,
    ``down_proj`` ``[E, H, I]``), which is exactly what the EP sharding plan
    (``Shard(0)``) and ``grouped_matmul`` expect.
    """

    def native_eager_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *,
        ep_group,
        fixed_router: bool = False,
    ) -> torch.Tensor:
        """Run EP with the released DeepSeek expert arithmetic order.

        Token routing and combination retain the native eager ordering (only
        the *which-experts* selection is static for hash-MoE layers), while the
        two expert projections use the accelerator's grouped GEMM. The final
        routed-expert accumulation runs in FP32, matching the reference.
        """
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        top_k_index, top_k_weights = normalize_expert_args(top_k_index, top_k_weights)
        if fixed_router:
            top_k_index, top_k_weights = fixed_router_for_debug(
                top_k_index,
                top_k_weights,
                self.num_global_experts,
            )

        indices, split_sizes = dispatch_preprocess(
            ep_group,
            top_k_index,
            self.num_global_experts,
            self.expert_ids_per_ep_rank,
        )
        local_counts, _ = indices
        input_split, output_splits = split_sizes

        hidden_states, unpermute_indices = alltoall_dispatch(
            ep_group,
            hidden_states,
            top_k_index,
            indices,
            split_sizes,
            fused=False,
        )
        source_order, expert_order = unpermute_indices
        router_weights = top_k_weights.reshape(-1).index_select(0, source_order.long())
        router_weights = all_to_all_single_autograd(
            router_weights,
            output_splits.tolist(),
            input_split.tolist(),
            ep_group,
        )
        router_weights = router_weights.index_select(0, expert_order.long())
        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj

        gate_up = grouped_matmul(
            hidden_states,
            local_counts,
            gate_up_proj,
            use_eager=False,
        ).float()
        gate, up = gate_up.chunk(2, dim=-1)
        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        intermediate = self.act_fn(gate) * up
        intermediate *= router_weights.float().unsqueeze(-1)
        expert_output = grouped_matmul(
            intermediate.to(hidden_states.dtype),
            local_counts,
            down_proj,
            use_eager=False,
        )

        # The probabilities are already applied. Ones preserve the original
        # [tokens, top-k] combine shape. Converting before the shared combine
        # makes both its return AllToAll and final expert sum operate in FP32.
        combine_weights = torch.ones_like(top_k_weights, dtype=torch.float32)
        output = alltoall_combine(
            ep_group,
            expert_output.float(),
            combine_weights,
            unpermute_indices,
            split_sizes,
            fused=False,
        )
        return output.view(*original_shape)


class DeepseekV4SparseMoeBlock(HFDeepseekV4SparseMoeBlock):
    """MoE block whose routed experts use the FSDPTurbo subclass above.

    The gate (top-k or hash) and the shared expert MLP are kept from the stock
    implementation untouched; only the routed-expert container is swapped.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.experts = DeepseekV4Experts(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        # Stock flow (routing + shared-expert mixing), but the routed-expert
        # accumulate runs in FP32 (reference semantics) and can therefore come
        # out wider than the block input. The stock DecoderLayer expects the
        # MoE output in the input dtype (it mixes with ``post.to(dtype)``), so
        # cast back at the block boundary.
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        return (routed + self.shared_experts(residual)).to(hidden_states.dtype)


# routed call hits ``self.experts`` -> the FSDPTurbo subclass.


class DeepseekV4DecoderLayer(HFDeepseekV4DecoderLayer):
    """Decoder block with FSDPTurbo attention and routed experts."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)

    # ``forward`` is inherited (flow through attn_hc / self_attn / ffn_hc / mlp).


class DeepseekV4Model(HFDeepseekV4Model):
    """Stock model with the FSDPTurbo decoder-layer chain.

    ``context_parallel_attention_class`` is consumed by the generic CP
    installer; the model-specific attention implementation remains in
    ``context_parallel.py``.

    The parent ``__init__`` builds its own layer stack (which already ran
    ``post_init``), so after replacing the stack we re-run ``post_init`` to
    apply the official initialization to the local layer chain. Re-running
    ``post_init`` is idempotent for initialized modules.
    """

    context_parallel_attention_class = DeepseekV4Attention

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    def context_parallel_attention_forward(
        self,
        attention_module: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: object,
        query_indices: torch.Tensor,
        cp_state: dict[str, object],
        group: object,
        gather_tensor: Callable[[torch.Tensor, object, int], torch.Tensor],
        kwargs: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model-specific attention path using injected CP communication."""
        return deepseek_v4_cp_attention_forward(
            attention_module,
            hidden_states,
            position_embeddings,
            query_indices,
            cp_state,
            group,
            gather_tensor,
            kwargs,
        )


class DeepseekV4ForCausalLM(HFDeepseekV4ForCausalLM):
    """Causal-LM head over the FSDPTurbo model chain.

    Same construction as the stock class; the replaced model chain (see
    :class:`DeepseekV4Model`) is a drop-in for the parent's stack, so the
    inherited ``forward``/loss path (``ForCausalLMLoss``, router aux loss)
    applies unchanged.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.model = DeepseekV4Model(config)


__all__ = [
    "DeepseekV4Attention",
    "DeepseekV4Config",
    "DeepseekV4DecoderLayer",
    "DeepseekV4Experts",
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
    "DeepseekV4SparseMoeBlock",
]

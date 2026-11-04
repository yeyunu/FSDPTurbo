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
"""FSDPTurbo DeepSeek-V4 model: inherit-and-override of the stock transformers
classes (transformers >= 5.8.1), mirroring ``fsdp_turbo/models/qwen/modeling_qwen4_exp.py``.

Why subclasses at all?
    transformers 5.8.1 ships the full DeepSeek-V4 architecture
    (``transformers.models.deepseek_v4``): attention (shared-KV MQA, partial
    RoPE, per-head sink, grouped output projection), the HCA/CSA compressors,
    the Lightning Indexer, hash-MoE (``DeepseekV4HashRouter`` +
    ``tid2eid``), the HyperConnection/HyperHead, and the eager attention
    kernel. FSDPTurbo only needs to touch *one* piece that stock code cannot
    express: routed-expert execution through the FSDPTurbo expert-parallel
    dispatcher (all-to-all + grouped GEMM on the accelerator). So we subclass
    ``DeepseekV4Experts`` and swap it into the block graph by subclassing the
    containers that instantiate it (SparseMoeBlock -> DecoderLayer -> Model ->
    ForCausalLM), with the local subclasses kept under the stock class names.
    There is no runtime monkey-patching: every container explicitly builds its
    children with the same ``__init__``/``forward`` contract as the parent.

Classes kept from the stock module without override: attention, compressors,
indexer, routers, hyper-connection, RMSNorm, RotaryEmbedding, caches and the
top-level ``DeepseekV4Model``/``DeepseekV4ForCausalLM`` forward paths — i.e.
everything except expert execution.
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
    DeepseekV4DecoderLayer as HFDeepseekV4DecoderLayer,
    DeepseekV4Experts as HFDeepseekV4Experts,
    DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM,
    DeepseekV4Model as HFDeepseekV4Model,
    DeepseekV4SparseMoeBlock as HFDeepseekV4SparseMoeBlock,
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
from .context_parallel import deepseek_v4_cp_attention_forward


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
    """Decoder block with the FSDPTurbo MoE block; all other children are stock."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Only the routed-experts container is swapped: attention, norms and
        # the hyper-connections were already built stock by the parent.
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

    context_parallel_attention_class = HFDeepseekV4Attention

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
    "DeepseekV4Config",
    "DeepseekV4DecoderLayer",
    "DeepseekV4Experts",
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
    "DeepseekV4SparseMoeBlock",
]

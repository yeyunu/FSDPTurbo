# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
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
"""Trainable causal-LM adapter over the stock transformers DeepSeek-V4 model.

The model itself is ``transformers >= 5.8.1``'s
``transformers.models.deepseek_v4`` implementation (see ``modeling.py`` for the
single inherit-and-override point: routed experts through the FSDPTurbo
expert-parallel dispatcher). This module only wires the stock model into the
``BaseTrainer`` contract and carries the reduced 4-layer / 16-expert training
profile of DeepSeek-V4-Flash used for GPU/NPU precision comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from .modeling import DeepseekV4ForCausalLM

# Reduced 4-layer / 16-expert training profile of DeepSeek-V4-Flash.
# Inlined here (no external json) so the training entry point stays self
# contained; scaling back up to the official 43-layer / 256-expert release is a
# matter of restoring ``num_hidden_layers`` / ``n_routed_experts`` /
# ``compress_ratios`` / ``mlp_layer_types``.
_REFERENCE_MODEL_KWARGS = {
    # Architecture (aligned with the official DeepSeek-V4-Flash release).
    "vocab_size": 129280,
    "hidden_size": 4096,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 4,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,  # -> partial_rotary_factor = 0.125
    "num_experts_per_tok": 6,
    "n_routed_experts": 16,
    "n_shared_experts": 1,
    "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 1.5,
    "swiglu_limit": 10.0,
    "sliding_window": 128,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 128,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1.0e-6,
    "compress_rope_theta": 160000.0,
    # Attention schedule: 4 layers -> sliding / sliding / CSA / HCA.
    "compress_ratios": [0, 0, 4, 128],
    # MoE schedule: layer 0 hash-MoE (frozen tid2eid lookup), rest top-k MoE.
    "num_hash_layers": 1,
    "rope_parameters": {
        "rope_type": "yarn",
        "factor": 16.0,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    },
    # The released checkpoint ships MTP blocks; we do not instantiate them
    # (loading skips ``mtp.*`` keys).
    "num_nextn_predict_layers": 0,
    "max_position_embeddings": 4096,
    "attention_dropout": 0.0,
    "use_sparse_flash_attn": True,
    "tie_word_embeddings": False,
    "bos_token_id": 0,  # nosec B105 - token id, not a secret
    "eos_token_id": 1,  # nosec B105 - token id, not a secret
}


@dataclass
class DeepseekV4CausalLMOutput:
    """Minimal Transformers-like output consumed by :class:`BaseTrainer`."""

    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None


def _validate_training_args(model_args: DeepseekV4Config) -> None:
    if len(model_args.layer_types) < model_args.num_hidden_layers:
        raise ValueError("layer_types must contain at least one entry per backbone layer.")
    if model_args.n_routed_experts <= 0 or model_args.num_experts_per_tok > model_args.n_routed_experts:
        raise ValueError("n_routed_experts / num_experts_per_tok are inconsistent.")
    # The released tid2eid table maps tokens to experts 0..255; training with a
    # reduced expert count relies on ``train.py``'s clip step.
    if model_args.n_routed_experts > 256:
        raise ValueError("n_routed_experts cannot exceed the released tid2eid range (256).")


class DeepseekV4ForCausalLMAdapter(nn.Module):
    """Trainable causal-LM facade over the stock ``DeepseekV4ForCausalLM``."""

    def __init__(self, backbone: nn.Module, model_args: DeepseekV4Config):
        super().__init__()
        self.model = backbone
        self.model_args = model_args

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        **model_inputs,
    ) -> DeepseekV4CausalLMOutput:
        if use_cache:
            raise ValueError("DeepSeek-V4 FSDPTurbo training does not support KV cache inputs.")
        if model_inputs:
            unsupported_inputs = ", ".join(sorted(model_inputs))
            raise ValueError(f"Unsupported model inputs: {unsupported_inputs}.")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )
        # ``outputs.loss`` is computed by the stock ``loss_function``, i.e.
        # ``transformers.loss.loss_utils.ForCausalLMLoss`` — the exact target
        # the context-parallel ``loss_function_patches`` re-wraps, so CP-local
        # loss stays active without any model-side loss code.
        return DeepseekV4CausalLMOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
        )


def build_deepseek_v4_model_args() -> DeepseekV4Config:
    """Build the reduced DeepSeek-V4-Flash config (validated by the stock
    ``DeepseekV4Config.__post_init__``).
    """
    model_args = DeepseekV4Config(**_REFERENCE_MODEL_KWARGS)
    _validate_training_args(model_args)
    return model_args


def _build_deepseek_v4_model(
    model_args: DeepseekV4Config,
    *,
    tokenizer=None,
) -> DeepseekV4ForCausalLMAdapter:
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        backbone = DeepseekV4ForCausalLM(model_args)
    finally:
        torch.set_default_dtype(previous_dtype)
    return DeepseekV4ForCausalLMAdapter(backbone, model_args)


def build_deepseek_v4_model(
    *,
    model_args: DeepseekV4Config | None = None,
    tokenizer=None,
) -> DeepseekV4ForCausalLMAdapter:
    """Build a BF16 model from the reduced config, always validating first.

    ``build_deepseek_v4_model_args()`` validates on construction, so the
    validation step here is a cheap idempotent pass that runs identically for
    default and caller-provided configs (no build/validate branching).
    """
    if model_args is None:
        model_args = build_deepseek_v4_model_args()
    _validate_training_args(model_args)
    return _build_deepseek_v4_model(model_args, tokenizer=tokenizer)


def initialize_deepseek_v4_model(model: nn.Module, standard_deviation: float = 0.02) -> None:
    """No-op: the stock ``DeepseekV4PreTrainedModel._init_weights`` already
    applies the official DeepSeek-V4 initialization at construction
    (router/HC/ape/sinks/embedding/head rules included). Kept as a hook so
    ``train.py``'s random-init branch reads the same at both sites.
    """
    return None


def build_deepseek_v4_model_meta(
    *,
    model_args: DeepseekV4Config | None = None,
    tokenizer=None,
) -> DeepseekV4ForCausalLMAdapter:
    """Build the DeepSeek-V4 skeleton entirely on meta device (zero host RAM).

    Instant construction for the 5L/256E profile (~67 GiB BF16): every
    parameter/buffer is a meta placeholder. FSDPTurbo then wraps it (outer
    FSDP shards + EP expert shards stay meta as well, mirroring the v4.1
    host-offload deferral pattern); the training entry calls ``to_empty`` on
    the wrapped model followed by :func:`materialize_deepseek_v4_model` to
    fill each rank's local shards (a few GiB/rank at 16 ranks).

    Construction runs under the same BF16 default dtype as
    :func:`_build_deepseek_v4_model` (stock transformers applies no dtype of
    its own at ``__init__`` time), so ``to_empty`` materializes BF16 directly
    instead of a 2x FP32 transient at exactly the memory point this path
    exists to protect.
    """
    if model_args is None:
        model_args = build_deepseek_v4_model_args()
    _validate_training_args(model_args)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            backbone = DeepseekV4ForCausalLM(model_args)
    finally:
        torch.set_default_dtype(previous_dtype)
    return DeepseekV4ForCausalLMAdapter(backbone, model_args)


def _local_shard(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor itself, or its local shard when it is a DTensor."""
    from torch.distributed.tensor import DTensor  # meta build path only

    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def materialize_deepseek_v4_model(model: nn.Module, standard_deviation: float | None = None) -> None:
    """Fill ``to_empty``-materialized local shards with the stock official init.

    Exact mirror of ``DeepseekV4PreTrainedModel._init_weights`` (router / HC /
    attention-sinks / compressors / rope / embedding / head rules included),
    applied per rank on the post-FSDP local shard, so no rank ever allocates
    more than its slice and the build stays O(params / world size).

    ``standard_deviation`` defaults to ``config.initializer_range`` (0.02 when
    unset), matching the stock ``_init_weights``. The mirror is enforced: any
    parameter or buffer no rule touches raises instead of silently keeping
    its ``to_empty`` placeholder content.
    """
    from torch.nn import init

    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        ROPE_INIT_FUNCTIONS,
        DeepseekV4Attention,
        DeepseekV4CSACompressor,
        DeepseekV4Experts,
        DeepseekV4HCACompressor,
        DeepseekV4HashRouter,
        DeepseekV4HyperConnection,
        DeepseekV4HyperHead,
        DeepseekV4Indexer,
        DeepseekV4RotaryEmbedding,
        DeepseekV4TopKRouter,
    )

    config = getattr(model, "config", None) or getattr(model, "model_args", None)
    if standard_deviation is None:
        standard_deviation = getattr(config, "initializer_range", 0.02) or 0.02

    # Qualified names of every parameter/buffer the rules below touch, so the
    # final sweep can prove the mirror left nothing uninitialized.
    handled: set[str] = set()

    def mark(module_name: str, *leaf_names: str) -> None:
        prefix = f"{module_name}." if module_name else ""
        handled.update(prefix + leaf for leaf in leaf_names)

    with torch.no_grad():
        for module_name, module in model.named_modules():
            if isinstance(module, DeepseekV4RotaryEmbedding):
                for layer_type in module.layer_types:
                    rope_init_fn = module.compute_default_rope_parameters
                    if module.rope_type[layer_type] != "default":
                        rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                    curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                    for attr in (f"{layer_type}_inv_freq", f"{layer_type}_original_inv_freq"):
                        buf = _local_shard(getattr(module, attr))
                        buf.copy_(curr_inv_freq.to(dtype=buf.dtype, device=buf.device))
                        mark(module_name, attr)
                continue

            if isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
                init.normal_(_local_shard(module.weight), mean=0.0, std=standard_deviation)
                mark(module_name, "weight")
                for buffer_name in ("e_score_correction_bias", "tid2eid"):
                    if hasattr(module, buffer_name):
                        init.zeros_(_local_shard(getattr(module, buffer_name)))
                        mark(module_name, buffer_name)
                continue

            if isinstance(module, DeepseekV4Experts):
                init.normal_(_local_shard(module.gate_up_proj), mean=0.0, std=standard_deviation)
                init.normal_(_local_shard(module.down_proj), mean=0.0, std=standard_deviation)
                mark(module_name, "gate_up_proj", "down_proj")
                continue

            if isinstance(module, DeepseekV4Attention) and hasattr(module, "sinks"):
                init.zeros_(_local_shard(module.sinks))
                mark(module_name, "sinks")
                continue

            if isinstance(module, DeepseekV4HyperConnection):
                init.normal_(_local_shard(module.fn), mean=0.0, std=standard_deviation)
                init.zeros_(_local_shard(module.base))
                init.ones_(_local_shard(module.scale))
                mark(module_name, "fn", "base", "scale")
                continue

            if isinstance(module, DeepseekV4HyperHead):
                init.normal_(_local_shard(module.hc_fn), mean=0.0, std=standard_deviation)
                init.zeros_(_local_shard(module.hc_base))
                init.ones_(_local_shard(module.hc_scale))
                mark(module_name, "hc_fn", "hc_base", "hc_scale")
                continue

            if isinstance(module, (DeepseekV4HCACompressor, DeepseekV4CSACompressor, DeepseekV4Indexer)) and hasattr(
                module, "position_bias"
            ):
                init.zeros_(_local_shard(module.position_bias))
                mark(module_name, "position_bias")
                continue

            # Default branch — mirrors ``PreTrainedModel._init_weights`` for the
            # remaining dense modules (norms ones, linears normal / zeros bias).
            class_name = type(module).__name__.lower()
            if "rmsnorm" in class_name or "layernorm" in class_name:
                for local_name, parameter in module.named_parameters(recurse=False):
                    p = _local_shard(parameter)
                    if local_name == "weight":
                        init.ones_(p)
                    else:
                        init.zeros_(p)
                    mark(module_name, local_name)
                continue

            if isinstance(module, nn.Linear):
                init.normal_(_local_shard(module.weight), mean=0.0, std=standard_deviation)
                if module.bias is not None:
                    init.zeros_(_local_shard(module.bias))
                mark(module_name, "weight", "bias")
                continue

            if isinstance(module, nn.Embedding):
                init.normal_(_local_shard(module.weight), mean=0.0, std=standard_deviation)
                if module.padding_idx is not None:
                    p = _local_shard(module.weight)
                    p[module.padding_idx].zero_()
                mark(module_name, "weight")
                continue

            for local_name, parameter in module.named_parameters(recurse=False):
                p = _local_shard(parameter)
                if p.ndim >= 2:
                    init.normal_(p, mean=0.0, std=standard_deviation)
                else:
                    init.zeros_(p)
                mark(module_name, local_name)
            for local_name, buffer in module.named_buffers(recurse=False):
                init.zeros_(_local_shard(buffer))
                mark(module_name, local_name)

    missing = [name for name, _ in model.named_parameters() if name not in handled]
    missing += [name for name, _ in model.named_buffers() if name not in handled]
    if missing:
        raise ValueError(
            "materialize_deepseek_v4_model found tensors not covered by any "
            f"initialization rule (they would stay uninitialized): {sorted(missing)}"
        )


def prepare_deepseek_v4_model_for_fsdp(
    model: nn.Module, *, device, parameter_dtype: torch.dtype, move_to_device: bool = True
) -> nn.Module:
    """Materialize a uniform parameter dtype before applying native FSDP.

    The stock BF16 model is already uniform, so this is a pure device move on
    the training loop entry; buffers (RoPE frequencies, tid2eid) are only moved
    to the accelerator, never cast, to preserve their integer/float layout.
    """
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype != parameter_dtype:
                parameter.data = parameter.data.to(dtype=parameter_dtype)
    if move_to_device:
        for module in model.modules():
            module._apply(lambda tensor: tensor.to(device=device), recurse=False)
    return model


__all__ = [
    "DeepseekV4CausalLMOutput",
    "DeepseekV4ForCausalLMAdapter",
    "build_deepseek_v4_model",
    "build_deepseek_v4_model_args",
    "build_deepseek_v4_model_meta",
    "materialize_deepseek_v4_model",
    "initialize_deepseek_v4_model",
    "prepare_deepseek_v4_model_for_fsdp",
]

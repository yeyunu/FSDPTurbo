# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Kimi-K3.1 model adapter for shard initialization and the DSA objective."""

from contextlib import AbstractContextManager, nullcontext
from types import TracebackType

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

from fsdp_turbo.utils.model import init_submodule_weights

from .dsa import KimiDSAAttention, KimiDSAForCausalLM, KimiK31ForConditionalGeneration
from .modeling_kimi_k31 import Learnable2DInterpPosEmbDivided_fixed, get_1d_sincos_pos_embed
from .modeling_kimi_linear import KimiDeltaAttention, KimiK_3_MoeRMSNormGated, KimiMoEGate, KimiRMSNorm


class _KimiDSALossReplayContext:
    """Restore live KL references when a decoder checkpoint finishes replaying."""

    def __init__(self, attentions: tuple[KimiDSAAttention, ...]) -> None:
        self._attentions = attentions
        self._saved_losses: list[tuple[torch.Tensor | None, ...]] = []

    def __enter__(self) -> "_KimiDSALossReplayContext":
        # Read at replay entry, after the outer training forward cleared KL.
        # A stack also supports repeated/nested entry on retained graphs.
        self._saved_losses.append(tuple(attention.indexer_loss for attention in self._attentions))
        return self

    def __exit__(
        self, _exc_type: type[BaseException] | None, _exc: BaseException | None, _traceback: TracebackType | None
    ) -> bool:
        for attention, loss in zip(self._attentions, self._saved_losses.pop()):
            attention.indexer_loss = loss
        return False


def kimi_dsa_loss_replay_context(module: torch.nn.Module | None) -> AbstractContextManager:
    """Isolate Kimi-K3.1 KL state from non-reentrant module replay."""
    attentions = (
        tuple(child for child in module.modules() if isinstance(child, KimiDSAAttention)) if module is not None else ()
    )
    return _KimiDSALossReplayContext(attentions) if attentions else nullcontext()


@torch.no_grad()
def _initialize_embedding(module: torch.nn.Embedding, std: float) -> None:
    """Clear only the locally owned padding row without gathering a DTensor."""
    torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.padding_idx is None:
        return
    weight = module.weight
    row_offset = 0
    if isinstance(weight, DTensor):
        _, offsets = compute_local_shape_and_global_offset(weight.shape, weight.device_mesh, weight.placements)
        row_offset = offsets[0]
        weight = weight.to_local()
    local_padding = module.padding_idx - row_offset
    if 0 <= local_padding < weight.shape[0]:
        weight[local_padding].zero_()


class KimiK31TrainingModel(KimiK31ForConditionalGeneration):
    """Kimi-K3.1 model with a supervised DSA objective and shard initialization."""

    class _KimiK31ForCausalLM(KimiDSAForCausalLM):
        """Use Transformers 5's tied-parameter mapping without patching the Turbo base."""

        _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    language_model_class = _KimiK31ForCausalLM

    # Preserve the Training adapter's positional argument order.
    def forward(  # pylint: disable=arguments-differ
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | list[torch.Tensor] | None = None,
        grid_thws: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> LlavaCausalLMOutputWithPast:
        """Return CE plus mean DSA KL, leaving gradient accumulation to the caller.

        Detached scalar ``loss_metrics`` are for generic engine logging. Layer
        KL references are cleared after composition; the returned loss owns the
        graph until backward. Evaluation reports CE only, as indexers do not
        construct a KL target in eval mode. Module checkpoint replay restores
        live KL attributes through the registered replay context.
        """
        if use_cache or return_dict is False:
            raise ValueError("Kimi-K31 training requires use_cache=False and return_dict=True.")
        layers = [
            layer.self_attn
            for layer in self.language_model.model.layers
            if isinstance(layer.self_attn, KimiDSAAttention)
        ]
        has_targets = False
        if labels is not None:
            labels = labels.clone()
            if attention_mask is not None:
                labels.masked_fill_(~attention_mask.bool(), self.config.ignore_index)
            if input_ids is not None:
                labels.masked_fill_(input_ids == self.config.media_placeholder_token_id, self.config.ignore_index)
            has_targets = labels[..., 1:].ne(self.config.ignore_index).any().item()
        try:
            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                labels=labels if has_targets else None,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
                **kwargs,
            )
            if labels is not None:
                # Mean CE is undefined when every target is ignored. Preserve
                # a differentiable zero without hiding a non-finite model output.
                ce_loss = output.loss.float() if has_targets else output.logits.sum() * 0
                output["loss"] = ce_loss
                output["loss_metrics"] = {"lm_loss": ce_loss.detach()}
                if self.training:
                    dsa_loss = torch.stack([layer.indexer_loss.float() for layer in layers]).mean()
                    output["loss"] = ce_loss + dsa_loss
                    output["loss_metrics"]["dsa_loss"] = dsa_loss.detach()
            return output
        finally:
            for layer in layers:
                layer.indexer_loss = None

    @torch.no_grad()
    def init_sharded_weights(self) -> None:
        """Initialize materialized FSDP/EP parameters in place after the caller seeds DTensor RNG.

        Matches the K31 random-init reference, including non-Linear parameters
        and deterministic vision time buffers. No full embedding is allocated
        and no Parameter object is replaced, preserving optimizer/FSDP aliases.
        """
        for module in self.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset) and not isinstance(module, torch.nn.Embedding):
                reset()

        def initialize_language_module(module: torch.nn.Module) -> None:
            if isinstance(module, torch.nn.Embedding):
                _initialize_embedding(module, self.config.text_config.initializer_range)
            elif module is not self.language_model.lm_head or module.weight is not self.get_input_embeddings().weight:
                # A tied head shares the embedding that already received its
                # initialization and local padding zero; do not overwrite it.
                self.language_model._init_weights(module)

        self.language_model.apply(initialize_language_module)
        for submodel in (self.vision_tower, self.mm_projector):
            init_submodule_weights(submodel)
        for module in self.modules():
            if isinstance(module, (KimiRMSNorm, KimiK_3_MoeRMSNormGated)):
                torch.nn.init.ones_(module.weight)
            elif isinstance(module, KimiDeltaAttention):
                torch.nn.init.uniform_(module.A_log, 1, 16)
                module.A_log.log_()
                torch.nn.init.zeros_(module.dt_bias)
            elif isinstance(module, KimiMoEGate):
                torch.nn.init.zeros_(module.e_score_correction_bias)
            elif isinstance(module, Learnable2DInterpPosEmbDivided_fixed):
                # Reconstruct even when include_buffers=True put this fixed
                # NumPy-generated buffer on meta during model construction.
                time_weight = torch.from_numpy(get_1d_sincos_pos_embed(module.dim, module.num_frames)).unsqueeze(1)
                module.time_weight = time_weight.to(device=module.weight.device, dtype=module.weight.dtype)
            if hasattr(module, "_is_hf_initialized"):
                module._is_hf_initialized = True

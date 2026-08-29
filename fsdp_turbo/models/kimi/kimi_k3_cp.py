# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Post-fusion Ulysses forward for the remote-code Kimi K3 model."""

import torch
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

from fsdp_turbo.distributed.context_parallel.batch import shard_fused_sequence_inputs
from fsdp_turbo.distributed.context_parallel.ulysses.utils import get_ulysses_context


def _build_position_ids(inputs_embeds, attention_mask):
    if attention_mask is None:
        return (
            torch.arange(inputs_embeds.size(1), device=inputs_embeds.device)
            .unsqueeze(0)
            .expand(inputs_embeds.size(0), -1)
        )
    return (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)


def _mask_labels(labels, attention_mask, ignore_index):
    if labels is None or attention_mask is None:
        return labels
    if labels.shape != attention_mask.shape:
        raise ValueError(
            "Kimi K3 labels and attention_mask must have the same complete fused shape before CP sharding; "
            f"got labels={tuple(labels.shape)}, attention_mask={tuple(attention_mask.shape)}."
        )
    return labels.masked_fill(attention_mask.to(device=labels.device) == 0, ignore_index)


def kimi_k3_model_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | list[torch.FloatTensor] | None = None,
    grid_thws: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    """Fuse Kimi multimodal inputs completely, then shard the language sequence."""
    if input_ids is None and inputs_embeds is None:
        raise ValueError("You must specify input_ids or inputs_embeds.")

    ulysses_group, _, ulysses_size = get_ulysses_context()
    if ulysses_size > 1:
        if past_key_values is not None or cache_position is not None or use_cache is True:
            raise ValueError("Post-fusion Ulysses CP for Kimi K3 does not support cache inputs.")
        if getattr(self, "enable_chunk_loss", False):
            raise ValueError("Kimi K3 chunk loss is not supported together with Ulysses CP.")
        use_cache = False

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and len(pixel_values) > 0 and input_ids.shape[1] != 1:
            image_features = self._extract_image_features(pixel_values, grid_thws)
            if self.mm_projector:
                image_features = self.mm_projector(image_features)
            inputs_embeds = inputs_embeds.to(image_features[0].dtype)
            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image_features,
                inputs_embeds,
                input_ids,
                attention_mask,
                labels,
            )
        elif past_key_values is not None:
            raise ValueError("Kimi K3 CP training does not support generation-style cache inputs.")
    elif pixel_values is not None and len(pixel_values) > 0:
        raise ValueError("pixel_values cannot be merged when inputs_embeds are supplied directly.")

    if attention_mask is None:
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    if position_ids is None:
        position_ids = _build_position_ids(inputs_embeds, attention_mask)

    complete_attention_mask = attention_mask
    complete_labels = _mask_labels(labels, complete_attention_mask, self.config.ignore_index)
    num_items_in_batch = kwargs.pop("num_items_in_batch", None)
    if ulysses_size > 1:
        inputs_embeds, position_ids, attention_mask = shard_fused_sequence_inputs(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            group=ulysses_group,
        )

    outputs = self.language_model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = outputs.hidden_states
    logits = self.language_model.lm_head(hidden_states).float()

    loss = None
    if complete_labels is not None:
        loss = self.loss_function(
            logits=logits,
            labels=complete_labels,
            vocab_size=self.config.text_config.vocab_size,
            num_items_in_batch=num_items_in_batch,
            ignore_index=self.config.ignore_index,
        )

    if not return_dict:
        output = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
        return (loss,) + output if loss is not None else output

    return LlavaCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

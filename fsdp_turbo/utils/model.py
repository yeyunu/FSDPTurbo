# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Shared model-construction helpers for examples and system tests."""

import logging

import torch
from torch import nn

from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)

_MODEL_CLASSES = {}


def register_model(model_name):
    """Class decorator that registers a model class under ``model_name``.

    The class registers itself as soon as its modeling module is imported.
    """

    def decorator(cls):
        _MODEL_CLASSES[model_name] = cls
        return cls

    return decorator


def get_model_by_name(name):
    """Return the model class registered under ``name``, or ``None``."""
    return _MODEL_CLASSES.get(name)


def set_attn_implementation(model_config, attn_implementation: str) -> None:
    """Stamp ``attn_implementation`` onto a (possibly nested) HF config.

    Args:
        model_config: HuggingFace ``PretrainedConfig`` (possibly multimodal with
            nested ``text_config`` / ``vision_config``).
        attn_implementation: Attention backend name (e.g. ``"flash_attention_2"``).
    """
    for config in (
        model_config,
        getattr(model_config, "text_config", None),
        getattr(model_config, "vision_config", None),
    ):
        if config is not None:
            config._attn_implementation = attn_implementation


def set_fused_overrides(model_config, config) -> None:
    """Propagate fused-operator switches from ``FSDPTurboConfig`` to HF ``model_config``.

    Args:
        model_config: HuggingFace ``PretrainedConfig`` (possibly multimodal with
            nested ``text_config`` / ``vision_config``) to stamp the switches on.
        config: The :class:`~fsdp_turbo.fsdp_turbo_config.FSDPTurboConfig` whose
            ``model`` and ``distributed.cp_plan`` fields are read.
    """
    import dataclasses
    from fsdp_turbo.ops.flash_attn import register_flash_attention

    register_flash_attention()

    sub_configs = [
        model_config,
        getattr(model_config, "text_config", None),
        getattr(model_config, "vision_config", None),
    ]
    if dataclasses.is_dataclass(config.model):
        fused_fields = [f.name for f in dataclasses.fields(config.model) if f.name.startswith("use_fused_")]
    else:
        fused_fields = [k for k in dir(config.model) if k.startswith("use_fused_")]
    for attr in fused_fields:
        val = getattr(config.model, attr, False)
        if val:
            for sub_cfg in sub_configs:
                if sub_cfg is not None:
                    setattr(sub_cfg, attr, val)


def set_num_hidden_layers(model_config, num_hidden_layers):
    """Reduce the language-model layer count on a (possibly nested) config.

    Walks the top-level config plus ``text_config``/``language_config`` (nested
    multimodal configs) and lowers ``num_hidden_layers`` on the first candidate
    that exposes it. No-op when ``num_hidden_layers`` is ``None``.
    """
    if num_hidden_layers is None:
        return False
    if num_hidden_layers <= 0:
        raise ValueError(f"model.num_hidden_layers must be positive, got {num_hidden_layers}.")

    config_candidates = [
        model_config,
        getattr(model_config, "text_config", None),
        getattr(model_config, "language_config", None),
    ]
    for candidate in config_candidates:
        if candidate is not None and hasattr(candidate, "num_hidden_layers"):
            old_num_layers = candidate.num_hidden_layers
            if num_hidden_layers > old_num_layers:
                raise ValueError(
                    f"model.num_hidden_layers={num_hidden_layers} exceeds "
                    f"checkpoint config num_hidden_layers={old_num_layers}."
                )
            candidate.num_hidden_layers = num_hidden_layers
            print_rank(
                logger.info,
                f"> Reduced language layers: {old_num_layers} -> {num_hidden_layers}",
            )
            return True

    raise ValueError("Cannot find num_hidden_layers in model config.")


def convert_model_dtype(model, dtype):
    """Cast all model parameters and buffers to ``dtype`` in place."""
    for param in model.parameters():
        param.data = param.data.to(dtype)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(dtype)


def init_submodule_weights(submodule, name="submodule"):
    """Randomly initialize the parameters of a materialized submodule."""
    init_fn = getattr(submodule, "_init_weights", None)
    if init_fn is not None:
        submodule.apply(init_fn)
        print_rank(
            logger.info,
            f"> Initialized {name} via {type(submodule).__name__}._init_weights.",
        )
        return

    print_rank(
        logger.warning,
        f"> {type(submodule).__name__} has no _init_weights; falling back to generic init for {name}.",
    )
    std = getattr(getattr(submodule, "config", None), "initializer_range", None) or 0.02
    for module in submodule.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

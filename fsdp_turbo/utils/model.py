# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Shared model-construction helpers for examples and system tests."""

import logging

import torch
from torch import nn

from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


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

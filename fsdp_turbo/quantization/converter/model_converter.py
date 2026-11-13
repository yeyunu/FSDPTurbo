# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib import import_module
from typing import Dict, List, Protocol, Union
from torch import nn

from fsdp_turbo.fsdp_turbo_config import QuantizeConfig


class ModelConverter(Protocol):
    """General model converter interface.
    A model converter is applying a modification to PyTorch model.
    Typical use cases are:
        - Quantization: using QAT, FP8, ... specialized linear layers;
        - Fused optimized layers (e.g. flash-attention, norms, ...)
    """

    def __init__(self, config: QuantizeConfig) -> None: ...

    def convert(self, model: nn.Module) -> None:
        """Inplace conversion of the model."""

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        """Post-optimizer (optional) hook (e.g. compute weights statistics)."""


_registry_model_converter_cls: Dict[str, type[ModelConverter]] = {}
"""Registry of model converter classes.
"""

_BUILTIN_MODEL_CONVERTER_MODULES = {
    "quantize.linear.mx": "fsdp_turbo.quantization.converter.mx_linear_converter",
    "quantize.moe.mx": "fsdp_turbo.quantization.converter.mx_moe_converter",
    "quantize.linear.w8a16": "fsdp_turbo.quantization.qat.w8a16.w8a16_converter",
    "quantize.moe.w8a16": "fsdp_turbo.quantization.qat.w8a16.w8a16_converter",
    "quantize.linear.w4a16": "fsdp_turbo.quantization.qat.w4a16.w4a16_converter",
    "quantize.moe.w4a16": "fsdp_turbo.quantization.qat.w4a16.w4a16_converter",
}


def register_model_converter(converter_cls: type[ModelConverter], name: str):
    """Register a model converter class.

    A registered model converter can be applied on any model
    using the `model.converters` config parameter.
    """

    if name in _registry_model_converter_cls:
        raise ValueError(f"A model converter '{name}' is already registered.")

    _registry_model_converter_cls[name] = converter_cls


class ModelConvertersContainer(ModelConverter):
    """Model converters sequential container.
    The class build the sequence of model converters defined in `model.converters`
    job config, and apply them to the model sequentially.
    """

    def __init__(self, config: QuantizeConfig) -> None:
        converter_names = config.converters or []
        for name in converter_names:
            module_name = _BUILTIN_MODEL_CONVERTER_MODULES.get(name)
            if name not in _registry_model_converter_cls and module_name is not None:
                import_module(module_name)
        converter_classes = [_registry_model_converter_cls[name] for name in converter_names]

        self.converters = [mh_cls(config) for mh_cls in converter_classes]

    def convert(self, model: nn.Module):
        for mh in self.converters:
            mh.convert(model)


def build_model_converter(config: QuantizeConfig) -> ModelConvertersContainer:
    """Build the collection of model converters to apply to the model."""
    return ModelConvertersContainer(config)

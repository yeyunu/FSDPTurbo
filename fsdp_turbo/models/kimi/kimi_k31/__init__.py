# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");


def register_kimi_k31() -> None:
    """Register Kimi-K31 DSA without changing Kimi-K3 registry mappings.

    Imports stay lazy so accelerator setup can precede modeling. The composite
    config constructs its local text config directly; registering the shared
    ``kimi_linear`` key would overwrite the original model's mapping.
    """
    from transformers import AutoConfig, AutoModel

    from .configuration_kimi_k31 import KimiK31Config
    from .dsa import KimiK31ForConditionalGeneration

    AutoConfig.register(KimiK31Config.model_type, KimiK31Config, exist_ok=True)
    AutoModel.register(KimiK31Config, KimiK31ForConditionalGeneration, exist_ok=True)


__all__ = ['register_kimi_k31']

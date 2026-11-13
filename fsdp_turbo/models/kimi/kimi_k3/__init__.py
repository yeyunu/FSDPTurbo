# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");


def register_kimi_k3():
    """Register repository-local Kimi configs and modeling with Transformers.

    Call before loading the config or model with ``trust_remote_code=False``.
    Imports stay lazy so accelerator setup can run before modeling is imported.
    """
    from transformers import AutoConfig, AutoModel

    from .configuration_kimi_k3 import KimiK3Config, KimiLinearConfig
    from .modeling_kimi_k3 import KimiK3ForConditionalGeneration

    AutoConfig.register('kimi_linear', KimiLinearConfig, exist_ok=True)
    AutoConfig.register('kimi_k3', KimiK3Config, exist_ok=True)
    AutoModel.register(KimiK3Config, KimiK3ForConditionalGeneration, exist_ok=True)


__all__ = ['register_kimi_k3']

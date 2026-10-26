# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");


def register_glm5_next():
    """Register repository-local GLM5Next configs and modeling with Transformers.

    Call before loading the config or model with ``trust_remote_code=False``.
    Imports stay lazy so accelerator setup can run before modeling is imported.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText

    from .configuration_glm5_next import Glm5NextConfig, Glm5NextTextConfig, Glm5NextVisionConfig
    from .modeling_glm5_next import Glm5NextForConditionalGeneration

    AutoConfig.register('glm5_next_text', Glm5NextTextConfig, exist_ok=True)
    AutoConfig.register('glm5_next_vision', Glm5NextVisionConfig, exist_ok=True)
    AutoConfig.register('glm5_next', Glm5NextConfig, exist_ok=True)
    AutoModelForImageTextToText.register(Glm5NextConfig, Glm5NextForConditionalGeneration, exist_ok=True)


__all__ = ['register_glm5_next']

# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import TYPE_CHECKING

# The classes resolve lazily (PEP 562): ``from fsdp_turbo.models import X``
# still works, but importing one model's submodule no longer pulls the other
# modeling stacks and their device-only dependencies (triton/fla/...). The
# name -> module map is shared with get_model_by_name, which imports a model on
# its first lookup for the same reason.
from .utils import _BUILTIN_MODELS

if TYPE_CHECKING:
    from .kimi.kimi_k3.modeling_kimi_k3 import KimiK3ForConditionalGeneration  # noqa: F401
    from .qwen.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration  # noqa: F401
    from .qwen.qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: F401
    from .qwen.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration  # noqa: F401


def __getattr__(name: str) -> type:
    if name in _BUILTIN_MODELS:
        import importlib

        module = importlib.import_module(_BUILTIN_MODELS[name][0], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_BUILTIN_MODELS)

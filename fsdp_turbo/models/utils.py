# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Model class registration shared by the built-in adapters.

Modeling modules import this module for the ``register_model`` decorator, and
``models/__init__.py`` resolves the package attributes lazily, so a class
registers once its own modeling module is imported. This module must not import
model implementations itself, so decorators can use it while the models package
is still initializing.
"""

from collections.abc import Callable
from importlib import import_module
from typing import TypeVar

_ModelClass = TypeVar('_ModelClass', bound=type)
_MODEL_CLASSES: dict[str, type] = {}

# Class name -> (relative module path, registered model name). Shared with
# models/__init__.py, which resolves a class lazily on attribute access; the
# model name lets get_model_by_name import the matching module on first lookup.
_BUILTIN_MODELS: dict[str, tuple[str, str]] = {
    "KimiK3ForConditionalGeneration": (".kimi.modeling_kimi_k3", "kimi_k3"),
    "Qwen4ExpForConditionalGeneration": (".qwen.modeling_qwen4_exp", "qwen4_exp"),
    "Qwen3_5ForConditionalGeneration": (".qwen.qwen3_5", "qwen3_5"),
    "Qwen3_5MoeForConditionalGeneration": (".qwen.qwen3_5_moe", "qwen3_5_moe"),
}


def register_model(model_name: str) -> Callable[[_ModelClass], _ModelClass]:
    """Class decorator that registers a model class under ``model_name``.

    The class registers itself as soon as its modeling module is imported.
    """

    def decorator(cls: _ModelClass) -> _ModelClass:
        _MODEL_CLASSES[model_name] = cls
        return cls

    return decorator


def get_model_by_name(name: str) -> type | None:
    """Return the class registered under the exact name, or None if unknown.

    The built-in modeling modules are imported lazily (see models/__init__.py),
    so the first lookup of a built-in name imports its module to run the
    ``@register_model`` decorator. Unknown names return None without importing.
    """
    if name not in _MODEL_CLASSES:
        for module_path, model_name in _BUILTIN_MODELS.values():
            if model_name == name:
                import_module(module_path, __package__)
                break
    return _MODEL_CLASSES.get(name)

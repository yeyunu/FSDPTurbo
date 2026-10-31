# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Model class registration shared by the built-in adapters.

Importing this module first runs models/__init__.py, whose model imports register
their classes here. This module must not import model implementations itself,
so decorators can use it while the models package is still initializing.
"""

from collections.abc import Callable
from typing import TypeVar

_ModelClass = TypeVar('_ModelClass', bound=type)
_MODEL_CLASSES: dict[str, type] = {}


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

    Built-in classes register during the parent package import; callers do not
    need to import individual modeling modules before querying this registry.
    """
    return _MODEL_CLASSES.get(name)

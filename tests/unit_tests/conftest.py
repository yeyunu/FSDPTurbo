# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Make ``fsdp_turbo.models.deepseek`` importable without triton/fla.

``models/__init__`` imports the kimi/qwen adapters; if that fails, install a
namespace module with the same ``__path__``. Where the real init succeeds
nothing is stubbed.
"""

import sys
import types
from pathlib import Path

if "fsdp_turbo.models" not in sys.modules:
    try:
        import fsdp_turbo.models  # noqa: F401
    except ImportError:
        _models_module = types.ModuleType("fsdp_turbo.models")
        _models_module.__path__ = [str(Path(__file__).resolve().parents[2] / "fsdp_turbo" / "models")]
        sys.modules["fsdp_turbo.models"] = _models_module

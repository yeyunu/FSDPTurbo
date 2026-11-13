# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class KimiK3RegistrationTests(unittest.TestCase):
    def test_registration_is_lazy_and_selects_local_classes_on_repeated_calls(self):
        package_name = '_test_local_kimi'
        source = Path(__file__).resolve().parents[2] / 'fsdp_turbo/models/kimi/kimi_k3/__init__.py'
        specification = importlib.util.spec_from_file_location(package_name, source)
        package = importlib.util.module_from_spec(specification)
        # Importing the package must not import Transformers or accelerator modeling.
        with patch.dict(sys.modules, {'transformers': None}):
            specification.loader.exec_module(package)

        config_module = ModuleType(f'{package_name}.configuration_kimi_k3')
        local_config = type('KimiK3Config', (), {})
        local_linear_config = type('KimiLinearConfig', (), {})
        config_module.KimiK3Config = local_config
        config_module.KimiLinearConfig = local_linear_config
        model_module = ModuleType(f'{package_name}.modeling_kimi_k3')
        local_model = type('KimiK3ForConditionalGeneration', (), {})
        model_module.KimiK3ForConditionalGeneration = local_model
        config_registry = {}
        model_registry = {}

        class AutoConfig:
            @staticmethod
            def register(model_type, config_class, exist_ok=False):
                if model_type in config_registry and not exist_ok:
                    raise ValueError('Duplicate config')
                config_registry[model_type] = config_class

        class AutoModel:
            @staticmethod
            def register(config_class, model_class, exist_ok=False):
                if config_class in model_registry and not exist_ok:
                    raise ValueError('Duplicate model')
                model_registry[config_class] = model_class

        transformers = ModuleType('transformers')
        transformers.AutoConfig = AutoConfig
        transformers.AutoModel = AutoModel
        with patch.dict(
            sys.modules,
            {
                package_name: package,
                f'{package_name}.configuration_kimi_k3': config_module,
                f'{package_name}.modeling_kimi_k3': model_module,
                'transformers': transformers,
            },
        ):
            package.register_kimi_k3()
            package.register_kimi_k3()

        self.assertIs(config_registry['kimi_linear'], local_linear_config)
        self.assertIs(config_registry['kimi_k3'], local_config)
        self.assertIs(model_registry[local_config], local_model)


if __name__ == '__main__':
    unittest.main()

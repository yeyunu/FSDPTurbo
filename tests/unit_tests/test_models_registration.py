# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

import ast
import importlib.util
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path


class RegistryLoader(SourceFileLoader):
    """Load registration functions without importing model implementations."""

    def get_code(self, fullname):
        source = self.get_filename(fullname)
        syntax_tree = ast.parse(Path(source).read_text(encoding='utf-8'), filename=source)
        syntax_tree.body = [
            node
            for node in syntax_tree.body
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == '_MODEL_CLASSES' for target in node.targets)
            )
            or (isinstance(node, ast.FunctionDef) and node.name in {'register_model', 'get_model_by_name'})
        ]
        return compile(syntax_tree, source, 'exec')


class ModelRegistrationTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[2] / 'fsdp_turbo/utils/model.py'
        loader = RegistryLoader('_test_model_registry', str(source))
        specification = importlib.util.spec_from_loader(loader.name, loader)
        registry = importlib.util.module_from_spec(specification)
        loader.exec_module(registry)
        self.register_model = registry.register_model
        self.get_model_by_name = registry.get_model_by_name

    def test_register_and_lookup_use_exact_model_names(self):
        @self.register_model('ut_qwen4_exp')
        class RegisteredQwen:
            pass

        @self.register_model('ut_kimi_k3')
        class RegisteredKimi:
            pass

        self.assertIs(self.get_model_by_name('ut_qwen4_exp'), RegisteredQwen)
        self.assertIs(self.get_model_by_name('ut_kimi_k3'), RegisteredKimi)
        self.assertIsNone(self.get_model_by_name('ut_deepseek_v41'))
        self.assertIsNone(self.get_model_by_name('ut_qwen4_exp '))

    def test_registration_returns_the_original_class(self):
        class RegisteredModel:
            pass

        self.assertIs(self.register_model('ut_model')(RegisteredModel), RegisteredModel)
        self.assertIs(self.get_model_by_name('ut_model'), RegisteredModel)

    def test_registration_replaces_a_class_with_the_same_name(self):
        @self.register_model('ut_model')
        class OriginalModel:
            pass

        @self.register_model('ut_model')
        class ReplacementModel:
            pass

        self.assertIs(self.get_model_by_name('ut_model'), ReplacementModel)


if __name__ == '__main__':
    unittest.main()

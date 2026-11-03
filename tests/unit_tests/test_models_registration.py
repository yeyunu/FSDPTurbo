# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

# models/__init__.py resolves classes lazily, so importing the registry does
# not import any model implementation.
from fsdp_turbo.models.utils import get_model_by_name, register_model


class ModelRegistrationTests(unittest.TestCase):
    def test_register_and_lookup_semantics(self):
        @register_model('ut_qwen4_exp')
        class RegisteredQwen:
            pass

        @register_model('ut_qwen4_exp')
        class ReplacementModel:
            pass

        # The decorator returns the original class; lookups use the exact name.
        self.assertIs(register_model('ut_model')(RegisteredQwen), RegisteredQwen)
        self.assertIs(get_model_by_name('ut_qwen4_exp'), ReplacementModel)
        self.assertIs(get_model_by_name('ut_model'), RegisteredQwen)
        # Unknown names and names with trailing whitespace do not match.
        self.assertIsNone(get_model_by_name('ut_deepseek_v41'))
        self.assertIsNone(get_model_by_name('ut_qwen4_exp '))


if __name__ == '__main__':
    unittest.main()

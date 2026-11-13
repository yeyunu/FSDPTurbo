import subprocess
import sys
import textwrap
from unittest import mock

import pytest


def test_non_quantized_recompute_does_not_import_quantization_kernels():
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("fsdp_turbo.quantization.mx_formats",
                                "fsdp_turbo.quantization.mxfp8_config")):
                raise AssertionError(f"eager quantization import: {name}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        import fsdp_turbo.quantization
        import fsdp_turbo.quantization.core.recompute_phase
        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        assert FSDPTurbo.__name__ == "FSDPTurbo"
        """
    )

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def test_builtin_converter_is_imported_only_when_requested():
    from fsdp_turbo.fsdp_turbo_config import QuantizeConfig
    from fsdp_turbo.quantization.converter import model_converter

    registry = model_converter._registry_model_converter_cls
    original_registry = registry.copy()
    registry.clear()

    class DummyConverter:
        def __init__(self, config):
            self.config = config

        def convert(self, model):
            return model

    def import_and_register(module_name):
        model_converter.register_model_converter(DummyConverter, "quantize.linear.mx")
        return mock.Mock(__name__=module_name)

    try:
        config = QuantizeConfig(converters=["quantize.linear.mx"])
        with mock.patch.object(model_converter, "import_module", side_effect=import_and_register) as lazy_import:
            container = model_converter.build_model_converter(config)
        lazy_import.assert_called_once_with("fsdp_turbo.quantization.converter.mx_linear_converter")
        assert isinstance(container.converters[0], DummyConverter)
    finally:
        registry.clear()
        registry.update(original_registry)


@pytest.mark.parametrize(
    'name',
    [
        'quantize.linear.mx',
        'quantize.moe.mx',
        'quantize.linear.w8a16',
        'quantize.moe.w8a16',
        'quantize.linear.w4a16',
        'quantize.moe.w4a16',
    ],
)
def test_each_builtin_converter_is_available_without_eager_package_imports(name):
    script = textwrap.dedent(f'''
        from fsdp_turbo.fsdp_turbo_config import QuantizeConfig
        from fsdp_turbo.quantization.converter.model_converter import build_model_converter
        config = QuantizeConfig(converters=[{name!r}])
        container = build_model_converter(config)
        assert len(container.converters) == 1
    ''')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unquantized_tp_and_expert_arithmetic_do_not_load_quantized_implementations():
    script = textwrap.dedent('''
        import importlib.abc
        import sys
        from types import SimpleNamespace
        import torch
        import torch.nn.functional as F

        class BlockQuantizedImplementations(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(("fsdp_turbo.quantization.mx_formats", "fsdp_turbo.quantization.qat",
                                        "fsdp_turbo.quantization.mxfp8_config", "fsdp_turbo.quantization.utils")):
                    raise AssertionError(f"unexpected quantization dependency: {fullname}")
                return None

        sys.meta_path.insert(0, BlockQuantizedImplementations())
        from fsdp_turbo.fsdp_turbo import FSDPTurbo
        from fsdp_turbo.fsdp_turbo_config import QuantizeConfig
        from fsdp_turbo.quantization.converter.model_converter import build_model_converter
        from fsdp_turbo.distributed.expert_parallel.dispatcher import experts_computation
        assert build_model_converter(QuantizeConfig(converters=None)).converters == []
        owner = SimpleNamespace(config=SimpleNamespace(
            distributed=SimpleNamespace(tensor_parallel_size=2),
            quantization=SimpleNamespace(quantization_plan=QuantizeConfig()),
        ))
        FSDPTurbo._capture_tp_info_on_mx_linear(owner)
        torch.manual_seed(5)
        inputs = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        gate_up = torch.randn(2, 8, 4, dtype=torch.float64, requires_grad=True)
        down = torch.randn(2, 4, 4, dtype=torch.float64, requires_grad=True)
        actual = experts_computation(inputs, torch.tensor([2, 1]), gate_up, down, fused=False)
        parts = []
        for index, chunk in enumerate(inputs.split([2, 1])):
            gate, up = F.linear(chunk, gate_up[index]).chunk(2, dim=-1)
            parts.append(F.linear(F.silu(gate) * up, down[index]))
        expected = torch.cat(parts)
        torch.testing.assert_close(actual, expected)
        grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (inputs, gate_up, down), grad, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (inputs, gate_up, down), grad)
        for left, right in zip(actual_grads, expected_grads):
            torch.testing.assert_close(left, right)
    ''')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    'converter_name,handler_module,converted_class',
    [
        ('quantize.linear.mx', 'fsdp_turbo.quantization.mx_formats.mx_linear', 'MXLinear'),
        ('quantize.moe.mx', 'fsdp_turbo.quantization.mx_formats.mx_gmm', 'MXFP8GMM'),
        ('quantize.linear.w8a16', 'fsdp_turbo.quantization.qat.w8a16.w8a16_linear', 'W8A16Linear'),
        ('quantize.moe.w8a16', 'fsdp_turbo.quantization.qat.w8a16.w8a16_gmm', 'W8A16GMM'),
        ('quantize.linear.w4a16', 'fsdp_turbo.quantization.qat.w4a16.w4a16_linear', 'W4A16Linear'),
        ('quantize.moe.w4a16', 'fsdp_turbo.quantization.qat.w4a16.w4a16_gmm', 'W4A16GMM'),
    ],
)
def test_fresh_converter_executes_real_handler(converter_name, handler_module, converted_class):
    script = textwrap.dedent(f'''
        import sys
        import torch
        from fsdp_turbo.fsdp_turbo_config import QuantizeConfig
        from fsdp_turbo.quantization.converter.model_converter import build_model_converter

        assert {handler_module!r} not in sys.modules
        is_moe = '.moe.' in {converter_name!r}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if is_moe:
                    experts = torch.nn.Module()
                    experts.num_experts = 2
                    experts.hidden_dim = 64
                    experts.intermediate_size = 64
                    experts.act_fn = torch.nn.functional.silu
                    experts.gate_up_proj = torch.nn.Parameter(torch.randn(2, 128, 64))
                    experts.down_proj = torch.nn.Parameter(torch.randn(2, 64, 64))
                    layer = torch.nn.Module()
                    layer.add_module('experts', experts)
                else:
                    layer = torch.nn.Linear(64, 64)
                self.layers = torch.nn.ModuleList([layer])

            @property
            def device(self):
                return next(self.parameters()).device

        model = Model()
        original = model.layers[0].experts if is_moe else model.layers[0]
        parameters = dict(original.named_parameters())
        expected = {{name: parameter.detach().clone() for name, parameter in parameters.items()}}
        container = build_model_converter(QuantizeConfig(
            converters=[{converter_name!r}],
            quant_format='E4M3',
            quant_apply_modules=['layers.0'],
            enable_fsdp_low_precision_all_gather=False,
        ))
        container.convert(model)

        # Resolve the class only after conversion, so the test cannot register
        # the missing handler by importing its implementation beforehand.
        assert {handler_module!r} in sys.modules
        expected_class = getattr(sys.modules[{handler_module!r}], {converted_class!r})
        converted = model.layers[0].experts if is_moe else model.layers[0]
        assert type(converted) is expected_class
        assert converted is original
        applied_config = getattr(converted, '_quant_config', None) or getattr(converted, 'config', None)
        assert applied_config is container.converters[0].config
        converted_parameters = dict(converted.named_parameters())
        assert converted_parameters.keys() == parameters.keys()
        for name, parameter in converted_parameters.items():
            assert parameter is parameters[name]
            assert parameter.device.type == 'cpu'
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
    ''')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

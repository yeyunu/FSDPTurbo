import ast
from pathlib import Path
from types import CodeType, FunctionType, MethodType

import pytest
import torch

from fsdp_turbo.modules.rms_norm import RMSNorm
from fsdp_turbo.ops.rms_norm import rms_norm

_MODULE_PATH = Path(__file__).parents[3] / "examples" / "deepseekv4" / "modeling_deepseek_v4.py"


def _load_rms_norm_forward(rms_norm_impl=rms_norm):
    """Load the actual adapter method without importing Transformers backend modules."""
    module = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    adapter = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "FSDPTurboDeepSeekV4RMSNorm"
    )
    forward = next(node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    compiled = compile(ast.Module(body=[forward], type_ignores=[]), str(_MODULE_PATH), "exec")
    forward_code = next(
        constant for constant in compiled.co_consts if isinstance(constant, CodeType) and constant.co_name == "forward"
    )
    return FunctionType(forward_code, {"rms_norm": rms_norm_impl})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_rms_norm_preserves_fp32_parameter_and_matches_output_and_gradients(monkeypatch, dtype, eps):
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cpu"))
    torch.manual_seed(123)
    module = RMSNorm(8, eps=eps)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(0.17, 1.83, 8))
    original_parameter = module.weight
    calls = []

    def checked_rms_norm(activations, weight, epsilon):
        assert weight.dtype == activations.dtype
        assert epsilon == eps
        calls.append(weight.dtype)
        return rms_norm(activations, weight, epsilon)

    forward = _load_rms_norm_forward(checked_rms_norm)
    module.forward = MethodType(forward, module)
    inputs = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weight = module.weight.detach().clone().requires_grad_()
    normalized_inputs = reference_inputs.float()
    variance = normalized_inputs.pow(2).mean(dim=-1, keepdim=True)
    normalized_inputs = normalized_inputs * torch.rsqrt(variance + eps)
    expected = (reference_weight.to(dtype=dtype) * normalized_inputs).to(dtype=dtype)

    output = module(inputs)
    assert output.dtype == dtype
    assert calls == [dtype]
    assert module.weight is original_parameter
    assert module.weight.dtype == torch.float32
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    upstream = torch.randn_like(output)
    output.backward(upstream)
    expected.backward(upstream)
    assert module.weight.grad.dtype == torch.float32
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    torch.testing.assert_close(module.weight.grad, reference_weight.grad, rtol=0, atol=0)

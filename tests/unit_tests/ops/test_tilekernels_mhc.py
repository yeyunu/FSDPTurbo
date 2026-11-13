# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU tests for mHC post dispatch and the TileKernels autograd adapter.

Raw TileKernels kernels are replaced with torch stand-ins, as in
test_mhc_pre_gradient.py. These tests do not validate Ascend kernel numerics.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fsdp_turbo.ops import registry
from fsdp_turbo.ops.cpu.tilekernels_mhc import tilekernels_mhc_post_cpu
from fsdp_turbo.ops.tilekernels import tilekernels_mhc_post


def _reference_post(x, residual, post, comb):
    # comb 的两个轴依次为输入流、输出流，用非对称矩阵检查混合方向。
    mixed = torch.einsum("bsij,bsid->bsjd", comb.float(), residual.float())
    expanded = torch.einsum("bsj,bsd->bsjd", post.float(), x.float())
    return (mixed + expanded).to(x.dtype)


def _make_inputs(batch=2, seqlength=3, dtype=torch.float32, mix_dtype=torch.float32, noncontiguous=False):
    generator = torch.Generator().manual_seed(17)
    shapes = (
        (batch, seqlength, 8),
        (batch, seqlength, 4, 8),
        (batch, seqlength, 4),
        (batch, seqlength, 4, 4),
    )
    inputs = []
    for shape, tensor_dtype in zip(shapes, (dtype, dtype, mix_dtype, mix_dtype)):
        storage_shape = (*shape[:-1], shape[-1] * 2) if noncontiguous else shape
        tensor = torch.randn(storage_shape, generator=generator).to(tensor_dtype)
        if noncontiguous:
            tensor = tensor[..., ::2]
        inputs.append(tensor.detach().requires_grad_(True))
    return inputs[0], inputs[1], inputs[2], inputs[3]


def _assert_forward_and_gradients(operator, inputs):
    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    expected = _reference_post(*reference_inputs)
    actual = operator(*inputs)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # 非连续且非全 1 的上游梯度可捕获轴序、梯度顺序和 contiguous 适配错误。
    generator = torch.Generator().manual_seed(29)
    grad_output = torch.randn((*expected.shape[:-1], expected.shape[-1] * 2), generator=generator).to(expected.dtype)[
        ..., ::2
    ]
    assert not grad_output.is_contiguous()
    actual_grads = torch.autograd.grad(actual, inputs, grad_output)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    tolerance = 2e-2 if actual.dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for actual_grad, expected_grad, tensor in zip(actual_grads, expected_grads, inputs):
        assert actual_grad.shape == tensor.shape
        assert actual_grad.dtype == tensor.dtype
        torch.testing.assert_close(actual_grad, expected_grad, rtol=tolerance, atol=tolerance)


def test_tilekernels_mhc_post_dispatches_using_input_device(monkeypatch):
    inputs = _make_inputs()
    expected = object()
    backend = Mock(return_value=expected)
    monkeypatch.setitem(registry._OPERATOR_REGISTRY["tilekernels_mhc_post"], "cpu", backend)
    # 不应根据默认加速器把 CPU 输入送往 NPU。
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("meta"))

    x, residual, post, comb = inputs
    assert tilekernels_mhc_post(x, residual, post, comb) is expected
    backend.assert_called_once_with(*inputs)


def test_tilekernels_mhc_post_has_cpu_reference():
    assert registry.get_op("tilekernels_mhc_post", "cpu") is tilekernels_mhc_post_cpu


@pytest.mark.parametrize("batch,seqlength", [(1, 1), (2, 3), (1, 7)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tilekernels_mhc_post_cpu_forward_and_all_gradients(batch, seqlength, dtype):
    _assert_forward_and_gradients(tilekernels_mhc_post, _make_inputs(batch, seqlength, dtype, noncontiguous=True))


@pytest.fixture(name="npu_adapter")
def fixture_npu_adapter(monkeypatch):
    # 直接加载 adapter，避免初始化其他 NPU 算子的可选依赖。
    # 注册项和伪第三方模块均由 monkeypatch 恢复，不能污染其他 UT。
    monkeypatch.setitem(
        registry._OPERATOR_REGISTRY,
        "tilekernels_mhc_post",
        dict(registry._OPERATOR_REGISTRY["tilekernels_mhc_post"]),
    )
    path = Path(__file__).parents[3] / "fsdp_turbo/ops/npu/tilekernels_mhc.py"
    spec = importlib.util.spec_from_file_location("test_tilekernels_mhc_adapter", path)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    monkeypatch.setattr(torch.accelerator, "get_device_name", lambda device: "Ascend950", raising=False)

    def forward(x, residual, post, comb):
        assert all(tensor.is_contiguous() for tensor in (x, residual, post, comb))
        assert post.shape == (*x.shape[:2], 4, 1)
        assert post.dtype == comb.dtype == torch.float32
        return _reference_post(x, residual, post.squeeze(-1), comb)

    def backward(x, residual, post, comb, grad_output):
        assert grad_output.is_contiguous()
        grad = grad_output.float()
        # 独立解析梯度，期望值则由 torch autograd 从参考公式求得。
        grad_x = (grad * post).sum(dim=-2)
        grad_residual = torch.einsum("bsij,bsjd->bsid", comb, grad)
        grad_post = (grad * x.float().unsqueeze(-2)).sum(dim=-1, keepdim=True)
        grad_comb = torch.einsum("bsid,bsjd->bsij", residual.float(), grad)
        return grad_x.to(x.dtype), grad_residual.to(residual.dtype), grad_post, grad_comb

    kernels = ModuleType("tile_kernels.mhc")
    kernels.mhc_post_fwd = Mock(side_effect=forward)
    kernels.mhc_post_bwd = Mock(side_effect=backward)
    package = ModuleType("tile_kernels")
    package.__path__ = []
    package.mhc = kernels
    monkeypatch.setitem(sys.modules, "tile_kernels", package)
    monkeypatch.setitem(sys.modules, "tile_kernels.mhc", kernels)
    return adapter, kernels


@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize(
    "dtype,mix_dtype",
    [(torch.float32, torch.float32), (torch.bfloat16, torch.float32), (torch.bfloat16, torch.bfloat16)],
)
def test_tilekernels_mhc_post_adapter_forward_and_all_gradients(npu_adapter, dtype, mix_dtype, noncontiguous):
    adapter, kernels = npu_adapter
    inputs = _make_inputs(dtype=dtype, mix_dtype=mix_dtype, noncontiguous=noncontiguous)
    _assert_forward_and_gradients(adapter.tilekernels_mhc_post_npu, inputs)
    kernels.mhc_post_fwd.assert_called_once()
    kernels.mhc_post_bwd.assert_called_once()


def test_tilekernels_mhc_post_rejects_non_950_device(npu_adapter, monkeypatch):
    adapter, kernels = npu_adapter
    monkeypatch.setattr(torch.accelerator, "get_device_name", lambda device: "Ascend910B")
    with pytest.raises(ValueError, match="requires Ascend 950"):
        adapter.tilekernels_mhc_post_npu(*_make_inputs())
    kernels.mhc_post_fwd.assert_not_called()


def test_tilekernels_mhc_post_missing_dependency_reports_config_path(npu_adapter, monkeypatch):
    adapter, kernels = npu_adapter
    monkeypatch.setitem(sys.modules, "tile_kernels.mhc", None)
    with pytest.raises(RuntimeError, match=r"model\.tilekernels\.tilekernels_mhc_post") as error:
        adapter.tilekernels_mhc_post_npu(*_make_inputs())
    assert isinstance(error.value.__cause__, ImportError)
    kernels.mhc_post_fwd.assert_not_called()


@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_tilekernels_mhc_post_kernel_errors_propagate(npu_adapter, phase):
    adapter, kernels = npu_adapter
    kernel = kernels.mhc_post_fwd if phase == "forward" else kernels.mhc_post_bwd
    kernel.side_effect = RuntimeError("kernel execution failed")
    with pytest.raises(RuntimeError, match="kernel execution failed"):
        adapter.tilekernels_mhc_post_npu(*_make_inputs()).sum().backward()
    kernel.assert_called_once()


@pytest.mark.parametrize("tilekernels_enabled", [False, True])
@pytest.mark.parametrize("fused_enabled", [False, True])
def test_block_hc_post_selects_backend(monkeypatch, tilekernels_enabled, fused_enabled):
    from fsdp_turbo.models.deepseek.deepseek_v41 import layers, mhc

    # 仅测试 hc_post 分支，避免构造与该方法无关的 Attention/MoE 参数。
    block = SimpleNamespace(
        mhc_post=mhc.MHCPost(
            SimpleNamespace(use_fused_mhc=fused_enabled, tilekernels={"tilekernels_mhc_post": tilekernels_enabled})
        )
    )
    inputs = _make_inputs()
    tilekernels_output, fused_output = object(), object()
    tilekernels_backend = Mock(return_value=tilekernels_output)
    fused_backend = Mock(return_value=fused_output)
    monkeypatch.setattr(mhc, "tilekernels_mhc_post", tilekernels_backend)
    monkeypatch.setattr(mhc, "fused_mhc_post", fused_backend)
    x, residual, post, comb = inputs
    output = layers.Block.hc_post(block, x, residual, post, comb)
    if tilekernels_enabled:
        assert output is tilekernels_output
        tilekernels_backend.assert_called_once_with(*inputs)
        fused_backend.assert_not_called()
    elif fused_enabled:
        assert output is fused_output
        fused_backend.assert_called_once_with(*inputs)
        tilekernels_backend.assert_not_called()
    else:
        torch.testing.assert_close(output, _reference_post(x, residual, post, comb))
        tilekernels_backend.assert_not_called()
        fused_backend.assert_not_called()

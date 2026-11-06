"""Real CUDA forward/adjoint checks for the PyTorch2.9 and public grouped-mm APIs."""

import pytest
import torch


accelerator = torch.accelerator.current_accelerator()
requires_cuda = pytest.mark.skipif(
    accelerator is None or accelerator.type != 'cuda' or not torch.accelerator.is_available(),
    reason='requires an available CUDA accelerator',
)


@pytest.fixture(scope='module', autouse=True)
def initialize_device_runtime():
    # The operator registry imports device properties while registering kernels.
    if torch.accelerator.is_available():
        from fsdp_turbo.utils.device import set_accelerator_compatible

        set_accelerator_compatible()


@requires_cuda
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('counts', [[16, 0, 32], [7, 0, 13, 2], [0, 0]])
@pytest.mark.parametrize('dimensions', [(128, 128), (128, 256), (2304, 5120)])
@pytest.mark.parametrize('weights_layout', ['linear', 'input_major'])
def test_grouped_linear_and_both_adjoints(dtype, counts, dimensions, weights_layout):
    from fsdp_turbo.ops.moe import grouped_matmul
    from fsdp_turbo.ops.registry import get_op

    assert get_op('grouped_matmul', 'cuda').__module__ == 'fsdp_turbo.ops.cuda.grouped_matmul'
    torch.manual_seed(42)
    device = torch.accelerator.current_accelerator()
    input_dim, output_dim = dimensions
    x = (torch.randn(sum(counts), input_dim, device=device, dtype=dtype) * 0.1).requires_grad_()
    weight_shape = (output_dim, input_dim) if weights_layout == 'linear' else (input_dim, output_dim)
    weights = (torch.randn(len(counts), *weight_shape, device=device, dtype=dtype) * 0.1).requires_grad_()
    reference_x = x.detach().double().requires_grad_()
    reference_weights = weights.detach().double().requires_grad_()
    sizes = torch.tensor(counts, device=device, dtype=torch.int64)
    actual = grouped_matmul(x, sizes, weights, weights_layout=weights_layout)
    offset = 0
    chunks = []
    for index, size in enumerate(counts):
        mat_b = reference_weights[index].T if weights_layout == 'linear' else reference_weights[index]
        chunks.append(reference_x[offset : offset + size] @ mat_b)
        offset += size
    expected = torch.cat(chunks)
    # Deliberately noncontiguous upstream gradients exercise the grouped kernel's layout boundary.
    upstream = torch.randn(output_dim, sum(counts), device=device, dtype=dtype).T
    actual_grads = torch.autograd.grad(actual, (x, weights), upstream)
    expected_grads = torch.autograd.grad(expected, (reference_x, reference_weights), upstream.double())
    # FP64 is the oracle: low-precision GEMM tiling need not reproduce a second
    # low-precision GEMM's cancellation points bit-for-bit.
    tolerance = 5e-3 if dtype == torch.bfloat16 else 1e-5
    for left, right in [(actual, expected), *zip(actual_grads, expected_grads)]:
        assert left.dtype == dtype and left.shape == right.shape
        if left.numel():
            relative_l2 = (left.double() - right).norm() / right.norm().clamp_min(1e-12)
            assert relative_l2.item() <= tolerance
        else:
            assert right.numel() == 0


@pytest.mark.parametrize('public_available', [True, False])
def test_grouped_mm_api_selection(monkeypatch, public_available):
    from fsdp_turbo.ops.cuda.grouped_matmul import _grouped_mm

    inputs = torch.ones(1, 2)
    weights = torch.ones(1, 2, 3)
    offsets = torch.tensor([1], dtype=torch.int32)
    expected = torch.ones(1, 3)
    calls = []

    def public(input_tensor, mat_b, *, offs):
        assert input_tensor is inputs and mat_b is weights and offs is offsets
        calls.append('public')
        return expected

    def private(input_tensor, mat_b, *, offs):
        assert input_tensor is inputs and mat_b is weights and offs is offsets
        calls.append('private')
        return expected

    monkeypatch.setattr(torch.nn.functional, 'grouped_mm', public if public_available else None, raising=False)
    monkeypatch.setattr(torch, '_grouped_mm', private, raising=False)
    assert _grouped_mm(inputs, weights, offsets) is expected
    assert calls == ['public' if public_available else 'private']


def test_missing_grouped_mm_apis_report_requirement(monkeypatch):
    from fsdp_turbo.ops.cuda.grouped_matmul import _grouped_mm

    monkeypatch.delattr(torch.nn.functional, 'grouped_mm', raising=False)
    monkeypatch.delattr(torch, '_grouped_mm', raising=False)
    with pytest.raises(RuntimeError, match='requires torch.nn.functional.grouped_mm or torch._grouped_mm'):
        _grouped_mm(torch.ones(1, 2), torch.ones(1, 2, 3), torch.tensor([1], dtype=torch.int32))


@pytest.mark.parametrize('weights_layout', ['linear', 'input_major'])
def test_empty_cuda_wrapper_needs_no_grouped_mm_api(monkeypatch, weights_layout):
    from fsdp_turbo.ops.cuda.grouped_matmul import grouped_matmul_cuda

    monkeypatch.delattr(torch.nn.functional, 'grouped_mm', raising=False)
    monkeypatch.delattr(torch, '_grouped_mm', raising=False)
    inputs = torch.empty(0, 2, requires_grad=True)
    weight_shape = (3, 4, 2) if weights_layout == 'linear' else (3, 2, 4)
    weights = torch.randn(*weight_shape, requires_grad=True)
    result = grouped_matmul_cuda(inputs, torch.zeros(3, dtype=torch.int64), weights, weights_layout=weights_layout)
    assert result.shape == (0, 4)
    input_gradient, weight_gradient = torch.autograd.grad(result.sum(), (inputs, weights))
    torch.testing.assert_close(input_gradient, torch.zeros_like(inputs), rtol=0, atol=0)
    torch.testing.assert_close(weight_gradient, torch.zeros_like(weights), rtol=0, atol=0)

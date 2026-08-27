import importlib.util
import pathlib
from unittest.mock import Mock, patch

import pytest
import torch

from fsdp_turbo.quantization.core.post_quant_weight import PostQuantWeight

# Load the kernel module by file so tests run even where torch_npu/triton
# (eagerly imported by fsdp_turbo.ops.npu) is unavailable.
_spec = importlib.util.spec_from_file_location(
    "mx_grouped_matmul_kernel", pathlib.Path("fsdp_turbo/ops/npu/mx_grouped_matmul.py")
)
kernel_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kernel_mod)


class _FakeNpu:
    """Minimal torch_npu stand-in that keeps shapes consistent so the autograd
    Function's forward/backward can run end-to-end on CPU tensors.
    """

    float8_e8m0fnu = Mock()

    def __init__(self, e, k, n):
        self._e = e
        self._k = k
        self._n = n

    def npu_grouped_matmul(self, inputs, weights, **kwargs):
        if kwargs.get("group_type") == 2:  # dw: [E, K, N], deterministic for value assertions
            return [torch.arange(self._e * self._k * self._n).reshape(self._e, self._k, self._n).to(torch.float32)]
        return [torch.randn(inputs[0].shape[0], weights[0].shape[-1], dtype=torch.float32)]

    @staticmethod
    def npu_dynamic_mx_quant(tensor, axis=-1, dst_type=None):
        return tensor, torch.ones(tensor.shape[0], 1)

    @staticmethod
    def npu_grouped_dynamic_mx_quant(tensor, group_list, round_mode="rint", dst_type=None, blocksize=32):
        return tensor, torch.ones(tensor.shape[0], 1)


def _run_apply(e, k, n, weight_2d_shape, weight):
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    x = torch.randn(8, k, dtype=torch.bfloat16)

    with patch.object(kernel_mod, "torch_npu", _FakeNpu(e, k, n)):
        output = kernel_mod.MxGroupedMatmul.apply(
            x,
            weight,
            x,  # x_fp8
            torch.ones(8, 1),
            torch.randn(e, k, n, dtype=torch.bfloat16),  # weight_fwd
            torch.ones(1, n, dtype=torch.bfloat16),
            torch.randn(e, n, k, dtype=torch.bfloat16),  # weight_bwd
            torch.ones(1, k, dtype=torch.bfloat16),
            weight_2d_shape,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.bfloat16,
            group_list,
            0,
            None,
            None,
        )
        output.sum().backward()

    return weight.grad


@pytest.mark.parametrize("stored_shape", [(4 * 8, 6), (4, 6, 8)])
def test_backward_returns_dw_in_stored_shape(stored_shape):
    """dw must come back in the stored layout: 2D [E*K, N] reshaped directly
    (a transpose would corrupt it), 3D [E, N, K] transposed from [E, K, N].
    """
    e, k, n = 4, 8, 6
    weight = torch.randn(*stored_shape, dtype=torch.bfloat16, requires_grad=True)

    grad = _run_apply(e, k, n, stored_shape, weight)

    dw = torch.arange(e * k * n).reshape(e, k, n).to(torch.float32)
    if len(stored_shape) == 2:
        expected = dw.reshape(e * k, n)
        wrong = dw.transpose(1, 2).reshape(e * k, n)
        assert grad.shape == (e * k, n)
        assert torch.equal(grad, expected)
        assert not torch.equal(grad, wrong)  # n != k: a transpose would reorder values
    else:
        expected = dw.transpose(1, 2).reshape(e, n, k)
        assert grad.shape == (e, n, k)
        assert torch.equal(grad, expected)


def _post_quant_weight(e, k, n):
    """PostQuantWeight with both pairs resident, stored in 3D [E, N, K]."""
    return PostQuantWeight(
        torch.randn(e, n, k, dtype=torch.bfloat16, requires_grad=True),
        torch.ones(e, 1, n, dtype=torch.bfloat16),
        torch.randn(e, n, k, dtype=torch.bfloat16, requires_grad=True),
        torch.ones(e, 1, k, dtype=torch.bfloat16),
        torch.bfloat16,
    )


def _run_apply_post_quant(e, k, n, weight_2d_shape, weight_ref, live_weight, weight_attr):
    """Run forward+backward through the post-quant path: the bwd pair is
    resolved live from the owner module in backward (real
    resolve_live_post_quant_weight, phase switch mocked).
    """
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    x = torch.randn(8, k, dtype=torch.bfloat16, requires_grad=True)
    owner = Mock()
    setattr(owner, weight_attr, live_weight)

    with (
        patch.object(kernel_mod, "torch_npu", _FakeNpu(e, k, n)),
        patch("fsdp_turbo.quantization.core.recompute_phase.ensure_mxfp8_backward_phase"),
    ):
        output = kernel_mod.MxGroupedMatmul.apply(
            x,
            weight_ref,
            x,  # x_fp8
            torch.ones(8, 1),
            torch.randn(e, k, n, dtype=torch.bfloat16),  # weight_fwd
            torch.ones(1, n, dtype=torch.bfloat16),
            None,  # weight_bwd: resolved live in backward
            None,
            weight_2d_shape,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.bfloat16,
            group_list,
            0,
            owner,
            weight_attr,
        )
        output.sum().backward()

    return weight_ref.grad


def test_post_quant_backward_resolves_live_bwd_pair():
    """Backward must resolve the live weight from the owner (not the saved
    snapshot) and come back with dw in the stored 3D shape.
    """
    e, k, n = 4, 8, 6
    live = _post_quant_weight(e, k, n)

    grad = _run_apply_post_quant(e, k, n, (e, n, k), live, live, "down_proj")

    assert grad.shape == (e, n, k)


def test_post_quant_backward_fails_closed_when_bwd_pair_not_resident():
    """A live weight without a resident bwd pair must fail loudly, not reuse
    a stale snapshot.
    """
    e, k, n = 4, 8, 6
    live = PostQuantWeight(
        torch.randn(e, n, k, dtype=torch.bfloat16, requires_grad=True),
        torch.ones(e, 1, n, dtype=torch.bfloat16),
        None,  # no bwd pair resident
        None,
        torch.bfloat16,
    )
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    x = torch.randn(8, k, dtype=torch.bfloat16, requires_grad=True)
    owner = Mock()
    owner.down_proj = live

    with (
        patch.object(kernel_mod, "torch_npu", _FakeNpu(e, k, n)),
        patch("fsdp_turbo.quantization.core.recompute_phase.ensure_mxfp8_backward_phase"),
    ):
        output = kernel_mod.MxGroupedMatmul.apply(
            x,
            live,
            x,
            torch.ones(8, 1),
            torch.randn(e, k, n, dtype=torch.bfloat16),
            torch.ones(1, n, dtype=torch.bfloat16),
            None,
            None,
            (e, n, k),
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.bfloat16,
            group_list,
            0,
            owner,
            "down_proj",
        )
        try:
            output.sum().backward()
        except RuntimeError as error:
            assert "backward weight is not resident" in str(error)
        else:
            raise AssertionError("a live weight without bwd pair must fail loudly")


def test_zero_tokens_backward_returns_zero_dw():
    """total_tokens == 0 must short-circuit dw to zeros (no grouped quant)."""
    e, k, n = 4, 8, 6
    weight = torch.randn(e * k, n, dtype=torch.bfloat16, requires_grad=True)
    group_list = torch.tensor([0], dtype=torch.int64)
    x = torch.randn(0, k, dtype=torch.bfloat16, requires_grad=True)

    with patch.object(kernel_mod, "torch_npu", _FakeNpu(e, k, n)):
        output = kernel_mod.MxGroupedMatmul.apply(
            x,
            weight,
            x,
            torch.ones(0, 1),
            torch.randn(e, k, n, dtype=torch.bfloat16),
            torch.ones(1, n, dtype=torch.bfloat16),
            torch.randn(e, n, k, dtype=torch.bfloat16),
            torch.ones(1, k, dtype=torch.bfloat16),
            (e * k, n),
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.bfloat16,
            group_list,
            0,
            None,
            None,
        )
        output.sum().backward()

    assert weight.grad.shape == (e * k, n)
    assert torch.equal(weight.grad, torch.zeros(e * k, n))


def test_post_quant_dtensor_weight_grad_wrapped_back():
    """EP-sharded weights (DTensor weight_ref) must get dw wrapped back into
    a DTensor with the weight's mesh/placements.
    """
    import torch.distributed as dist
    from torch.distributed.tensor import DeviceMesh, DTensor, Shard

    e, k, n = 4, 8, 6
    dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29500", rank=0, world_size=1)
    try:
        mesh = DeviceMesh("cpu", [0])
        live = _post_quant_weight(e, k, n)
        weight_ref = DTensor.from_local(live, mesh, [Shard(0)])
        weight_ref.retain_grad()

        group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
        x = torch.randn(8, k, dtype=torch.bfloat16, requires_grad=True)
        owner = Mock()
        owner.down_proj = live

        with (
            patch.object(kernel_mod, "torch_npu", _FakeNpu(e, k, n)),
            patch("fsdp_turbo.quantization.core.recompute_phase.ensure_mxfp8_backward_phase"),
        ):
            output = kernel_mod.MxGroupedMatmul.apply(
                x,
                weight_ref,
                x,
                torch.ones(8, 1),
                torch.randn(e, k, n, dtype=torch.bfloat16),
                torch.ones(1, n, dtype=torch.bfloat16),
                None,
                None,
                (e, n, k),
                torch.float8_e4m3fn,
                torch.float8_e4m3fn,
                torch.bfloat16,
                group_list,
                0,
                owner,
                "down_proj",
            )
            output.sum().backward()

        assert isinstance(weight_ref.grad, DTensor)
        assert weight_ref.grad.shape == (e, n, k)
    finally:
        dist.destroy_process_group()

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""End-to-end guards for ``AllToAllGroupedMatmul`` backward gradients on NPU.

The autograd Function runs for real against the environment's torch_npu: tensors
live on ``npu``, the HCCL communicator name comes from a real single-rank process
group, and the weight gradient is produced by the real ``npu_grouped_matmul``
kernel.  Only the two alltoallv communication kernels are substituted — their
``ep_world_size`` argument accepts 8/16/32/64 ranks only (torch_npu op docs),
which a single-process unit test cannot provide.

The stand-ins compute the math those kernels perform under identity routing (each
rank keeps its own tokens), so the op degenerates into grouped linear layers and
the gradients landing on ``inputs`` / ``weights`` / ``shared_inputs`` /
``shared_weight`` must equal the analytic gradients ``dx = dy @ W`` and
``dW = dy.T @ x``.

This pins the backward fix — ``npu_grouped_matmul`` must receive
``([output_grad.T], [permute_output])`` and ``npu_gmm_alltoallv`` must not carry
the stale ``trans_gmm_weight`` flag.  Reverting either changes which tensors get
multiplied; with ``K != N`` the shapes no longer compose, so the regression fails
loudly instead of producing silently wrong gradients.
"""

from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist

torch_npu = pytest.importorskip("torch_npu")

# Import after the skip: fsdp_turbo.ops.npu eagerly imports torch_npu.
import fsdp_turbo.ops.npu.grouped_matmul_mc2 as mc2_module  # noqa: E402

if not torch.npu.is_available():
    pytest.skip("requires an NPU device", allow_module_level=True)

E, K, N = 2, 128, 256  # experts, hidden-in, hidden-out; K != N so wrong operand orders break shapes
COUNTS = [32, 16]  # per-expert token counts, 16-aligned for the real GMM kernel
T = sum(COUNTS)


def _rows(tensor, counts):
    """Split ``tensor`` along dim 0 into chunks sized by ``counts``."""
    return list(torch.split(tensor, counts, dim=0))


def _npu_alltoallv_gmm_identity(
    inputs, weights, hcomm, ep_world_size, send_counts, recv_counts, mm_x=None, mm_weight=None, **kwargs
):
    """Grouped linear under identity routing: ``y_e = x_e @ W_e.T``."""
    output = torch.cat([x @ w.t() for x, w in zip(_rows(inputs, recv_counts), weights)])
    shared_output = None if mm_x is None else mm_x @ mm_weight
    return output, shared_output, inputs  # permute_output: identity permutation


def _npu_gmm_alltoallv_identity(
    output_grad, weights, hcomm, ep_world_size, recv_counts, send_counts, mm_x=None, mm_weight=None, **kwargs
):
    """Reverse of the forward stand-in: ``dx_e = dy_e @ W_e``.

    ``trans_gmm_weight`` is honoured on purpose: the production backward must not
    pass it, and a stale flag transposes ``W_e`` to ``[K, N]`` so the shapes stop
    composing instead of producing silently wrong gradients.
    """
    weights = weights.transpose(-1, -2) if kwargs.get("trans_gmm_weight") else weights
    inputs_grad = torch.cat([dy @ w for dy, w in zip(_rows(output_grad, recv_counts), weights)])
    shared_inputs_grad = None if mm_x is None else mm_x @ mm_weight.t()
    return inputs_grad, shared_inputs_grad


@contextmanager
def _single_rank_hccl_group():
    """Real single-rank HCCL group, so production computes a real communicator name."""
    torch.npu.set_device(0)
    dist.init_process_group(backend="hccl", init_method="tcp://127.0.0.1:29531", rank=0, world_size=1)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


def _patch_comm_kernels(monkeypatch):
    # Substitute only the two kernels that need 8+ ranks; npu_grouped_matmul and
    # everything else on the real torch_npu module stays untouched.
    monkeypatch.setattr(torch_npu, "npu_alltoallv_gmm", _npu_alltoallv_gmm_identity)
    monkeypatch.setattr(torch_npu, "npu_gmm_alltoallv", _npu_gmm_alltoallv_identity)


def test_expert_backward_matches_analytic_gradients(monkeypatch):
    """Without a shared expert the backward must produce exactly the gradients of
    ``x_e @ W_e.T``: ``dx = dy @ W`` and ``dW = dy.T @ x`` from the real kernel.
    """
    torch.manual_seed(0)
    inputs = torch.randn(T, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(E, N, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    output_mask = torch.randn(T, N, device="npu", dtype=torch.bfloat16)

    _patch_comm_kernels(monkeypatch)
    with _single_rank_hccl_group() as group:
        output, shared_output = mc2_module.AllToAllGroupedMatmul.apply(
            inputs, weights, group, COUNTS, COUNTS, None, None
        )

        assert shared_output is None
        (output * output_mask).sum().backward()

    # The real kernel and plain matmul round bf16 differently, so compare in fp32.
    expected_dx = torch.cat([m @ w for m, w in zip(_rows(output_mask, COUNTS), weights)])
    expected_dw = torch.stack([m.t() @ x for m, x in zip(_rows(output_mask, COUNTS), _rows(inputs, COUNTS))])
    assert torch.allclose(inputs.grad.float(), expected_dx.float(), rtol=2e-2, atol=2e-2)
    assert torch.allclose(weights.grad.float(), expected_dw.float(), rtol=2e-2, atol=2e-2)


def test_shared_expert_backward_matches_analytic_gradients(monkeypatch):
    """With a shared expert the shared path must additionally produce the
    gradients of ``shared_inputs @ shared_weight`` on both operands.
    """
    torch.manual_seed(0)
    inputs = torch.randn(T, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(E, N, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    shared_inputs = torch.randn(T, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
    shared_weight = torch.randn(K, N, device="npu", dtype=torch.bfloat16, requires_grad=True)
    output_mask = torch.randn(T, N, device="npu", dtype=torch.bfloat16)
    shared_mask = torch.randn(T, N, device="npu", dtype=torch.bfloat16)

    _patch_comm_kernels(monkeypatch)
    with _single_rank_hccl_group() as group:
        output, shared_output = mc2_module.AllToAllGroupedMatmul.apply(
            inputs, weights, group, COUNTS, COUNTS, shared_inputs, shared_weight
        )

        ((output * output_mask).sum() + (shared_output * shared_mask).sum()).backward()

    expected_dx = torch.cat([m @ w for m, w in zip(_rows(output_mask, COUNTS), weights)])
    expected_dw = torch.stack([m.t() @ x for m, x in zip(_rows(output_mask, COUNTS), _rows(inputs, COUNTS))])
    expected_shared_dx = (shared_mask @ shared_weight.t()).float()
    expected_shared_dw = (shared_inputs.t() @ shared_mask).float()
    assert torch.allclose(inputs.grad.float(), expected_dx.float(), rtol=2e-2, atol=2e-2)
    assert torch.allclose(weights.grad.float(), expected_dw.float(), rtol=2e-2, atol=2e-2)
    assert torch.allclose(shared_inputs.grad.float(), expected_shared_dx, rtol=2e-2, atol=2e-2)
    assert torch.allclose(shared_weight.grad.float(), expected_shared_dw, rtol=2e-2, atol=2e-2)

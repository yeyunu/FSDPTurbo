"""Chunk loss preserves gradient dtype and matches the unchunked loss."""

import pytest
import torch
import torch.nn.functional as F

from fsdp_turbo.memory.chunk_loss.chunk_loss import calculate_lm_loss, chunk_loss


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_chunk_gradients_preserve_weight_dtype(dtype):
    torch.manual_seed(7)
    hidden = torch.randn(2, 17, 4, dtype=dtype, requires_grad=True)
    weight = torch.randn(11, 4, dtype=dtype, requires_grad=True)
    chunk_hidden = hidden.detach().clone().requires_grad_()
    chunk_weight = weight.detach().clone().requires_grad_()
    targets = torch.randint(11, (2, 17))
    expected = F.cross_entropy(F.linear(hidden, weight).float().reshape(-1, 11), targets.flatten())
    arguments = [
        dict(shift_labels=labels, alpha=torch.tensor(34), reduction="sum") for labels in targets.split(4, dim=1)
    ]
    actual = chunk_loss(chunk_hidden, chunk_weight, None, calculate_lm_loss, arguments, 4)
    # Check the live backward buffer, not just the returned parameter gradient.
    saved_weight_gradient = actual.grad_fn.saved_tensors[1]
    assert saved_weight_gradient.dtype == dtype
    assert saved_weight_gradient.nbytes == chunk_weight.nbytes
    (expected / 3).backward()
    (actual / 3).backward()
    torch.testing.assert_close(actual, expected, rtol=0.0001, atol=0.0001)
    torch.testing.assert_close(chunk_weight.grad, weight.grad, rtol=0.03, atol=0.002)
    torch.testing.assert_close(chunk_hidden.grad, hidden.grad, rtol=0.01, atol=0.0001)
    assert chunk_weight.grad.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [3, 5, 64])
@pytest.mark.parametrize("alpha_layout", ["scalar", "sample", "token"])
@pytest.mark.parametrize("split_arguments", [False, True])
def test_merged_tokens_match_dense_loss_and_gradients(dtype, chunk_size, alpha_layout, split_arguments):
    torch.manual_seed(17)
    hidden = torch.randn(3, 4, 7, dtype=dtype).transpose(1, 2).requires_grad_()
    weight = torch.randn(11, 4, dtype=dtype, requires_grad=True)
    targets = torch.randint(11, (3, 7))
    targets[0, 2:5] = -100
    targets[1, -1] = -100
    valid = targets.ne(-100)
    if alpha_layout == "scalar":
        alpha = valid.sum()
        expanded_alpha = alpha
    elif alpha_layout == "sample":
        alpha = valid.sum(1) * targets.shape[0]
        expanded_alpha = alpha[:, None]
    else:
        alpha = torch.arange(1, targets.numel() + 1).reshape_as(targets).float()
        expanded_alpha = alpha
    losses = F.cross_entropy(
        F.linear(hidden, weight).float().reshape(-1, 11), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    expected = (losses / expanded_alpha).sum()
    reduction = "sum" if alpha_layout == "scalar" else "none"
    if split_arguments:
        arguments = [
            dict(
                shift_labels=labels,
                alpha=alpha[:, start : start + chunk_size] if alpha.ndim == 2 else alpha,
                reduction=reduction,
            )
            for start, labels in zip(range(0, targets.shape[1], chunk_size), targets.split(chunk_size, dim=1))
        ]
    else:
        arguments = [dict(shift_labels=targets, alpha=alpha, reduction=reduction)]

    chunk_hidden = hidden.detach().clone().requires_grad_()
    chunk_weight = weight.detach().clone().requires_grad_()
    chunk_lengths = []

    def record_projection(hidden_chunk, head_weight, head_bias, **kwargs):
        assert hidden_chunk.shape[0] == 1
        assert hidden_chunk.shape[1] <= chunk_size
        assert kwargs["shift_labels"].shape == hidden_chunk.shape[:2]
        if alpha_layout != "scalar":
            assert kwargs["alpha"].shape == kwargs["shift_labels"].shape
        chunk_lengths.append(hidden_chunk.shape[1])
        return calculate_lm_loss(hidden_chunk, head_weight, head_bias, **kwargs)

    actual = chunk_loss(chunk_hidden, chunk_weight, None, record_projection, arguments, chunk_size)
    (expected / 3).backward()
    (actual / 3).backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    tolerance = dict(rtol=0.03, atol=0.002) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(chunk_hidden.grad, hidden.grad, **tolerance)
    torch.testing.assert_close(chunk_weight.grad, weight.grad, **tolerance)
    token_count = int(valid.sum()) if alpha_layout == "scalar" else targets.numel()
    assert chunk_lengths == [min(chunk_size, token_count - start) for start in range(0, token_count, chunk_size)]

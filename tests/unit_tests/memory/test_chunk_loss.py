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

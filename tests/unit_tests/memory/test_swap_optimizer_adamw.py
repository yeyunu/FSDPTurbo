# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import pytest
import torch

from fsdp_turbo.memory.swap_optimizer.swap_optimizer import AdamWSwap


pytestmark = pytest.mark.skipif(
    torch.accelerator.current_accelerator() is None,
    reason="swap optimizer requires a usable device (fast fail on CPU-only hosts)",
)


_STEPS = 4


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("chunk_numel", [4 << 20, 32])
def test_adamw_swap_matches_torch_adamw(amsgrad, chunk_numel):
    """Multi-step numerical equivalence between AdamWSwap and torch.optim.AdamW(fused=True)."""
    torch.manual_seed(1234)
    dev = torch.accelerator.current_device_index()
    params = [torch.randn(3, 20, device=dev, requires_grad=True) for _ in range(5)]
    ref_params = [p.detach().clone().requires_grad_(True) for p in params]

    opt = AdamWSwap(
        params,
        lr=1e-3,
        amsgrad=amsgrad,
        chunk_numel=chunk_numel,
    )
    ref_opt = torch.optim.AdamW(ref_params, lr=1e-3, amsgrad=amsgrad, fused=True)

    for _ in range(_STEPS):
        for p, rp in zip(params, ref_params):
            grad = torch.randn_like(p)
            p.grad = grad.clone()
            rp.grad = grad.clone()
        opt.step()
        ref_opt.step()

    for p, rp in zip(params, ref_params):
        torch.testing.assert_close(p, rp)


def test_resident_states_ratio_keeps_small_params_on_device():
    """Resident params keep their states on device and stay numerically equivalent to AdamW(fused=True)."""
    torch.manual_seed(1234)
    dev = torch.accelerator.current_device_index()
    params = [torch.randn(3, 20, device=dev, requires_grad=True) for _ in range(3)]
    params += [torch.randn(50, 80, device=dev, requires_grad=True) for _ in range(2)]
    ref_params = [p.detach().clone().requires_grad_(True) for p in params]

    opt = AdamWSwap(params, lr=1e-3, resident_states_ratio=0.3)
    ref_opt = torch.optim.AdamW(ref_params, lr=1e-3, fused=True)

    smalls, bigs = set(params[:3]), set(params[3:])
    assert opt._resident_params == smalls
    for p in smalls:
        assert all(v is None for v in opt.param_to_cpu_states_map[p].values())

    for _ in range(_STEPS):
        for p, rp in zip(params, ref_params):
            grad = torch.randn_like(p)
            p.grad = grad.clone()
            rp.grad = grad.clone()
        opt.step()
        ref_opt.step()

    for p, rp in zip(params, ref_params):
        torch.testing.assert_close(p, rp)
    # Resident storage stays non-zero; swapped params offloaded back to host
    for p in smalls:
        assert all(s is not None and s.storage().size() != 0 for s in opt.param_to_device_states_map[p].values())
    for p in bigs:
        assert all(s is None or s.storage().size() == 0 for s in opt.param_to_device_states_map[p].values())


def test_step_handles_zero_numel_param():
    """0-numel local shards must neither raise a spurious OOM nor touch memory illegally."""
    dev = torch.accelerator.current_device_index()
    params = [
        torch.randn(0, device=dev, requires_grad=True),
        torch.randn(4, 5, device=dev, requires_grad=True),
    ]
    opt = AdamWSwap(params, lr=1e-3)
    params[0].grad = torch.randn(0, device=dev)
    params[1].grad = torch.randn(4, 5, device=dev)
    opt.step()


def test_step_raises_oom_when_first_chunk_cannot_fit():
    dev = torch.accelerator.current_device_index()
    params = [torch.randn(3, 20, device=dev, requires_grad=True)]
    opt = AdamWSwap(params, lr=1e-3)
    params[0].grad = torch.randn_like(params[0])
    # Simulate device memory exhaustion: zero budget
    opt.total_memory = 1
    with pytest.raises(AssertionError, match="OOM"):
        opt.step()


def test_fused_adamw_update_raises_when_states_not_on_device():
    dev = torch.accelerator.current_device_index()
    params = [torch.randn(3, 20, device=dev, requires_grad=True)]
    opt = AdamWSwap(params, lr=1e-3)
    params[0].grad = torch.randn_like(params[0])
    # States offloaded to host at construction: the update must fail loudly on empty storage
    group = opt.param_to_group_map[params[0]]
    with pytest.raises(AssertionError, match="OOM"):
        opt._prepare_fused_adamw(params[0], group)

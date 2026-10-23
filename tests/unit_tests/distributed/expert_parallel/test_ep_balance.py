"""CPU regressions for balanced EP routing, communication and gradients."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fsdp_turbo.distributed.expert_parallel.dispatcher import all2all_experts_forward_fn
from fsdp_turbo.distributed.expert_parallel.dispatcher_all_gather import all_gather_experts_forward_fn
from fsdp_turbo.distributed.expert_parallel.ep_balance import (
    EPBalanceStrategy,
    EPBalanceWeights,
)
from fsdp_turbo.distributed.expert_parallel.ep_balance.training_context import TrainingContext, TrainingStage


def test_disabled_balance_leaves_weights_unchanged():
    """Disabled balancing performs no tensor copies or communication."""
    gate_up, down = torch.randn(2, 6, 3), torch.randn(2, 3, 3)
    balance = EPBalanceWeights(None, gate_up)
    assert balance.prepare_gate_up(gate_up, down) is gate_up
    assert balance.prepare_down(down) is down


@pytest.mark.parametrize('dispatcher', ['all2all', 'allgather'])
def test_recompute_reuses_balanced_routes_without_communication(dispatcher):
    """Recompute returns cached gathered routes rather than local input IDs."""
    strategy = EPBalanceStrategy.__new__(EPBalanceStrategy)
    strategy.ep_group = Mock()
    strategy.num_experts = 4
    strategy.planner = Mock(num_local_experts=2, max_dup_experts_num=1)
    routes = torch.tensor([[0], [5], [1], [3]])
    cache = {
        'selected_experts_with_dup': routes,
        'num_global_sum_tokens_per_local_expert': torch.tensor([1, 1, 0]),
        'num_global_tokens_per_local_expert': torch.tensor([[1, 0, 0], [0, 1, 0]]),
        'input_splits': torch.tensor([1, 1]),
        'output_splits': torch.tensor([1, 1]),
    }
    strategy.planner.get_plan_item.side_effect = cache.__getitem__
    with (
        patch.object(TrainingContext(), 'get_training_stage', return_value=TrainingStage.BACKWARD),
        patch(
            'fsdp_turbo.distributed.expert_parallel.ep_balance.ep_balance_strategy.dist.get_world_size', return_value=2
        ),
        patch(
            'fsdp_turbo.distributed.expert_parallel.ep_balance.ep_balance_strategy.dist.all_gather_into_tensor'
        ) as counts,
        patch('fsdp_turbo.distributed.expert_parallel.ep_balance.ep_balance_strategy.allgather_tokens_in_ep') as gather,
    ):
        result = getattr(strategy, f'prepare_{dispatcher}')(torch.tensor([[0], [1]]))
    assert result[0] is routes
    counts.assert_not_called()
    gather.assert_not_called()
    strategy.planner.select_dup_experts_and_rearrange_experts.assert_not_called()


def _reference(hidden_states, routes, scores, gate_up, down):
    result = []
    for token, selected, probabilities in zip(hidden_states, routes, scores):
        outputs = []
        for expert, probability in zip(selected, probabilities):
            gate, up = torch.nn.functional.linear(token, gate_up[expert]).chunk(2, dim=-1)
            outputs.append(torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down[expert]) * probability)
        result.append(sum(outputs))
    return torch.stack(result)


@patch('torch.accelerator.current_device_index', return_value=torch.device('cpu'))
def _distributed_balance_worker(rank, rendezvous, _device_index):
    from fsdp_turbo.distributed.expert_parallel import utils as ep_utils

    # Exercise the existing deterministic counter on CPU (histc is accelerator-specific).
    ep_utils.EP_BINCOUNT_DISPATCH = True
    # The original collective helper allocates on the active accelerator.
    # Select CPU for this CPU-only distributed regression.
    torch.accelerator.current_device_index = lambda: torch.device('cpu')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        group = dist.group.WORLD
        TrainingContext().set_training_stage(TrainingStage.FORWARD)
        for dispatcher in (all2all_experts_forward_fn, all_gather_experts_forward_fn):
            for duplicates, empty_destination in ((None, False), (0, False), (1, False), (0, True)):
                strategy = None if duplicates is None else EPBalanceStrategy(group, 4, duplicates)
                # Two iterations catch persistent parameter hooks and stale state.
                gate_up_leaf = (torch.randn(2, 6, 3, dtype=torch.float64) * 0.1).requires_grad_()
                down_leaf = (torch.randn(2, 3, 3, dtype=torch.float64) * 0.1).requires_grad_()
                for iteration in range(2):
                    torch.manual_seed(321 + iteration)
                    hidden = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
                    gate_up = (torch.randn(4, 6, 3, dtype=torch.float64) * 0.1).requires_grad_()
                    down = (torch.randn(4, 3, 3, dtype=torch.float64) * 0.1).requires_grad_()
                    scores = torch.rand(8, 2, dtype=torch.float64, requires_grad=True)
                    routes = torch.tensor([[0, 1], [0, 1], [0, 2], [0, 1]] * 2)
                    if empty_destination:
                        routes.fill_(0)  # Rank one receives no expert tokens.
                    token_slice = slice(rank * 4, (rank + 1) * 4)
                    expert_slice = slice(rank * 2, (rank + 1) * 2)
                    with torch.no_grad():
                        gate_up_leaf.copy_(gate_up[expert_slice])
                        down_leaf.copy_(down[expert_slice])
                    gate_up_leaf.grad = down_leaf.grad = None
                    local_hidden = hidden[token_slice].detach().clone().requires_grad_()
                    local_scores = scores[token_slice].detach().clone().requires_grad_()
                    local_routes = routes[token_slice].clone()
                    output = dispatcher(
                        local_hidden,
                        local_routes,
                        local_scores,
                        (gate_up_leaf, down_leaf),
                        4,
                        torch.tensor([0, 1, 0, 1]),
                        group,
                        fused=False,
                        ep_balance_strategy=strategy,
                    )
                    reference = _reference(hidden, routes, scores, gate_up, down)
                    torch.testing.assert_close(output, reference[token_slice], rtol=1e-9, atol=1e-11)
                    output.square().sum().backward()
                    reference.square().sum().backward()
                    torch.testing.assert_close(local_hidden.grad, hidden.grad[token_slice], rtol=1e-8, atol=1e-11)
                    torch.testing.assert_close(local_scores.grad, scores.grad[token_slice], rtol=1e-8, atol=1e-11)
                    torch.testing.assert_close(gate_up_leaf.grad, gate_up.grad[expert_slice], rtol=1e-8, atol=1e-11)
                    torch.testing.assert_close(down_leaf.grad, down.grad[expert_slice], rtol=1e-8, atol=1e-11)
                    if strategy is not None and duplicates and not empty_destination:
                        assert (strategy.planner.get_plan_item('dup_experts_map') >= 0).any()
                    assert not gate_up_leaf._backward_hooks
                    assert not down_leaf._backward_hooks
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_balanced_dispatchers_match_dense_outputs_and_gradients(tmp_path):
    """Two real CPU ranks compare both dispatchers against an unsharded MoE.

    Covers disabled balance, zero duplicate slots, duplicated hot experts,
    zero-token expert ranks and repeated forward/backward on the same weights.
    """
    pytest.importorskip('numba', reason='The duplicated-expert planner requires Numba')
    mp.start_processes(
        _distributed_balance_worker,
        args=(f'file://{tmp_path / "gloo_init"}',),
        nprocs=2,
        start_method='spawn',
    )

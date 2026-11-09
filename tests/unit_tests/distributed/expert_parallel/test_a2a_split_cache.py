from unittest.mock import patch

import pytest
import torch

from fsdp_turbo.distributed.expert_parallel import utils
from fsdp_turbo.distributed.expert_parallel.utils import (
    DISPATCH_SPLIT_POLICY_ATTR,
    DispatchSplitPolicy,
    MC2_DISPATCH_SPLIT_CACHE_ATTR,
    pop_dispatch_splits,
    push_dispatch_splits,
)


class _DummyExpertModule(torch.nn.Module):
    pass


def _enable_policy(module: torch.nn.Module, cache_cap: int = 8) -> None:
    setattr(module, DISPATCH_SPLIT_POLICY_ATTR, DispatchSplitPolicy(enabled=True, cache_cap=cache_cap))


def _host_group_list(send_counts: list[int], ep_size: int) -> list[int]:
    """Replicate the backward fallback in grouped_matmul_mc2: per-local-expert
    totals summed over source ranks from the flat op-level send_counts.
    """
    num_cols = len(send_counts) // ep_size
    chunk = [send_counts[i * num_cols : (i + 1) * num_cols] for i in range(ep_size)]
    return [sum(col) for col in zip(*chunk)]


@pytest.mark.parametrize("ep_size,e_local", [(2, 3), (4, 2), (8, 14)])
def test_device_totals_match_backward_host_sums(ep_size: int, e_local: int):
    """The device-side recv_totals must equal the backward host sums
    elementwise; a mismatch silently corrupts the wgrad grouping.
    """
    counts = torch.randint(0, 9, (ep_size, e_local), dtype=torch.int64)
    host = torch.tensor(_host_group_list(counts.reshape(-1).tolist(), ep_size), dtype=torch.int64)
    device = counts.reshape(ep_size, -1).sum(0)
    assert torch.equal(host, device)


def test_combine_group_list_uses_recv_side_counts():
    """The combine op's send_counts is the dispatcher recv_counts, so its
    backward group_list equals recv_totals; send_totals (fixed bug) differs
    whenever the routing is unbalanced.
    """
    ep_size = 2
    send_counts = torch.tensor([5, 0, 0, 1], dtype=torch.int64)  # dispatcher send side [E]
    recv_counts = torch.tensor([2, 2, 3, 0], dtype=torch.int64)  # dispatcher recv side [EP*E_local]
    send_totals = send_counts.reshape(ep_size, -1).sum(0)
    recv_totals = recv_counts.reshape(ep_size, -1).sum(0)
    host_group_list = torch.tensor(_host_group_list(recv_counts.tolist(), ep_size), dtype=torch.int64)
    assert torch.equal(host_group_list, recv_totals)
    assert not torch.equal(recv_totals, send_totals)


def test_push_pop_lifo_and_token_total_guard():
    """Replay pops the most recent forward entry; the caller-side guard
    rejects it when the cached token total no longer matches the input.
    """
    module = _DummyExpertModule()
    _enable_policy(module)
    entry_old = ([1, 1], [1], torch.zeros(1, dtype=torch.int64), False)  # total 2
    entry_new = ([3], [3], torch.ones(1, dtype=torch.int64), False)  # total 3
    with patch.object(utils, "is_backward_recompute", return_value=False):
        push_dispatch_splits(module, entry_old, MC2_DISPATCH_SPLIT_CACHE_ATTR)
        push_dispatch_splits(module, entry_new, MC2_DISPATCH_SPLIT_CACHE_ATTR)
    with patch.object(utils, "is_backward_recompute", return_value=True):
        assert pop_dispatch_splits(module, MC2_DISPATCH_SPLIT_CACHE_ATTR) is entry_new
        # Guard expressions used by the dispatchers: a 3-token input matches,
        # a 2-token input rejects the stale entry.
        assert sum(entry_new[0]) == 3
        assert sum(entry_new[0]) != 2
        assert pop_dispatch_splits(module, MC2_DISPATCH_SPLIT_CACHE_ATTR) is entry_old


def test_stale_entry_guard_covers_both_payload_layouts():
    """Lock which tuple element holds the token counts for the eager
    (input_split CPU tensor) and mc2 (send_counts list) cache entries.
    """
    eager_entry = (torch.zeros(4, dtype=torch.int64), torch.tensor([2, 1]), torch.tensor([3]))
    mc2_entry = ([2, 1], [3], torch.ones(2, dtype=torch.int64), False)
    assert int(eager_entry[1].sum()) == 3
    assert sum(mc2_entry[0]) == 3
    # A different batch size must reject the entry.
    assert int(eager_entry[1].sum()) != 4
    assert sum(mc2_entry[0]) != 4


def test_push_pop_noop_when_disabled():
    module = _DummyExpertModule()
    entry = ([1], [1], torch.zeros(1, dtype=torch.int64), False)
    # No policy attached (e.g. a module never tagged by EP init): no-ops.
    push_dispatch_splits(module, entry, MC2_DISPATCH_SPLIT_CACHE_ATTR)
    with patch.object(utils, "is_backward_recompute", return_value=True):
        assert pop_dispatch_splits(module, MC2_DISPATCH_SPLIT_CACHE_ATTR) is None
    assert getattr(module, MC2_DISPATCH_SPLIT_CACHE_ATTR, None) is None
    # An attached-but-disabled policy behaves the same way.
    setattr(module, DISPATCH_SPLIT_POLICY_ATTR, DispatchSplitPolicy(enabled=False))
    push_dispatch_splits(module, entry, MC2_DISPATCH_SPLIT_CACHE_ATTR)
    with patch.object(utils, "is_backward_recompute", return_value=True):
        assert pop_dispatch_splits(module, MC2_DISPATCH_SPLIT_CACHE_ATTR) is None
    assert getattr(module, MC2_DISPATCH_SPLIT_CACHE_ATTR, None) is None


def test_push_clears_cache_at_cap():
    module = _DummyExpertModule()
    _enable_policy(module, cache_cap=2)
    with patch.object(utils, "is_backward_recompute", return_value=False):
        for i in range(3):
            push_dispatch_splits(module, ([i + 1],), MC2_DISPATCH_SPLIT_CACHE_ATTR)
    cache = getattr(module, MC2_DISPATCH_SPLIT_CACHE_ATTR)
    assert len(cache) == 1

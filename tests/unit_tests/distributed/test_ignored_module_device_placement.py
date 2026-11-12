import pytest
import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule

from fsdp_turbo.distributed.fully_shard_parallel.fully_shard_parallel import (
    _move_unmanaged_ignored_modules,
)


class ManagedLinear(FSDPModule, nn.Linear):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)


class HostStorageModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Linear(2, 2)
        self.register_buffer("metadata", torch.ones(2))

    def pre_fsdp_hook(self, parallel_state):
        return None


@pytest.fixture(name="moved_tensor_ids")
def record_moved_tensor_ids(monkeypatch):
    calls = []

    def record_transfer(tensor, device):
        calls.append((tensor.data_ptr(), device))
        return tensor

    monkeypatch.setattr(torch.Tensor, "to", record_transfer)
    return calls


def test_moves_only_unmanaged_parameters_and_buffers(moved_tensor_ids):
    model = nn.Module()
    model.ignored = nn.Linear(2, 2)
    model.ignored.register_buffer("metadata", torch.ones(2))
    model.other = nn.Linear(2, 2)
    device = torch.device("cpu")

    _move_unmanaged_ignored_modules(model, {"ignored"}, device)

    expected = {tensor.data_ptr() for tensor in (*model.ignored.parameters(), *model.ignored.buffers())}
    assert {pointer for pointer, _ in moved_tensor_ids} == expected
    assert all(destination == device for _, destination in moved_tensor_ids)


@pytest.mark.parametrize("nested", [False, True])
def test_preserves_fsdp_subtree_with_overlapping_ignored_paths(moved_tensor_ids, nested):
    model = nn.Module()
    experts = ManagedLinear(2, 2)
    experts.child = nn.Linear(2, 2)
    experts.register_buffer("metadata", torch.ones(2))
    model.ignored = nn.ModuleList([experts, nn.Linear(2, 2)]) if nested else experts
    ignored = {"ignored", "ignored.0", "ignored.0.child", "ignored.child"}

    _move_unmanaged_ignored_modules(model, ignored, torch.device("cpu"))

    expected = {tensor.data_ptr() for tensor in model.ignored[1].parameters()} if nested else set()
    assert {pointer for pointer, _ in moved_tensor_ids} == expected


def test_preserves_host_hook_descendants_and_shared_state(moved_tensor_ids):
    model = nn.Module()
    model.ignored = nn.Module()
    model.ignored.host = HostStorageModule()
    model.ignored.alias = nn.Linear(2, 2)
    model.ignored.alias.weight = model.ignored.host.table.weight
    model.ignored.alias.register_buffer("metadata", model.ignored.host.metadata)

    _move_unmanaged_ignored_modules(model, {"ignored", "ignored.host.table"}, torch.device("cpu"))

    assert [pointer for pointer, _ in moved_tensor_ids] == [model.ignored.alias.bias.data_ptr()]


def test_preserves_fsdp_parameters_shared_with_unmanaged_sibling(moved_tensor_ids):
    model = nn.Module()
    model.ignored = nn.Module()
    model.ignored.alias = nn.Linear(2, 2)
    model.ignored.experts = ManagedLinear(2, 2)
    model.ignored.alias.weight = model.ignored.experts.weight

    _move_unmanaged_ignored_modules(model, {"ignored"}, torch.device("cpu"))

    expected = [model.ignored.alias.bias.data_ptr()]
    assert [pointer for pointer, _ in moved_tensor_ids] == expected


def test_skips_meta_state_and_deduplicates_shared_tensors(moved_tensor_ids):
    model = nn.Module()
    model.ignored = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2, device="meta")])
    model.ignored.alias = model.ignored[0]
    model.ignored.register_parameter("shared_weight", model.ignored[0].weight)

    _move_unmanaged_ignored_modules(model, {"ignored", "ignored.0"}, torch.device("cpu"))

    expected = {tensor.data_ptr() for tensor in model.ignored[0].parameters()}
    assert {pointer for pointer, _ in moved_tensor_ids} == expected
    assert len(moved_tensor_ids) == len(expected)


def test_no_ignored_modules_leaves_model_unchanged(moved_tensor_ids):
    _move_unmanaged_ignored_modules(nn.Linear(2, 2), set(), torch.device("cpu"))
    assert moved_tensor_ids == []

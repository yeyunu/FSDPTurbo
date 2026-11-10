# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Regression checks for local storage allocation before HF pretrained loading."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.multiprocessing import spawn

from fsdp_turbo.models.deepseek.deepseek_v41 import (
    initialize_deepseek_v41_model,
    load_deepseek_v41_from_pretrained,
    prepare_deepseek_v41_model_for_fsdp,
)
from fsdp_turbo.models.deepseek.deepseek_v41.engram.host_offload_engram import HostOffloadEngram


def _shared_model() -> nn.Module:
    model = nn.Module()
    model.left = nn.Linear(4, 3, bias=False)
    model.right = nn.Linear(4, 3, bias=False)
    model.right.weight = model.left.weight
    model.left.weight.requires_grad_(False)
    model.norm = nn.LayerNorm(4, bias=False)
    model.register_buffer("rope", torch.tensor([1 + 2j], dtype=torch.complex64), persistent=False)
    model.register_buffer("hash_ids", torch.tensor([2, 7], dtype=torch.int64), persistent=False)
    return model


def test_materialization_preserves_aliases_buffers_and_loaded_values():
    model = _shared_model()
    rope = model.rope.clone()
    hash_ids = model.hash_ids.clone()
    prepare_deepseek_v41_model_for_fsdp(model, device=torch.device("cpu"), parameter_dtype=torch.bfloat16)
    assert model.left.weight.is_meta
    assert model.left.weight is model.right.weight

    rng_state = torch.get_rng_state()
    initialize_deepseek_v41_model(model, device=torch.device("cpu"), initialize_weights=False)

    assert not model.left.weight.is_meta
    assert model.left.weight.device.type == "cpu"
    assert model.left.weight.dtype == torch.bfloat16
    assert not model.left.weight.requires_grad
    assert model.left.weight is model.right.weight
    torch.testing.assert_close(model.rope, rope)
    torch.testing.assert_close(model.hash_ids, hash_ids)
    assert torch.equal(torch.get_rng_state(), rng_state)

    expected = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    norm = torch.full((4,), 2.0, dtype=torch.bfloat16)
    model.load_state_dict({"left.weight": expected, "right.weight": expected, "norm.weight": norm}, strict=True)
    parameters = dict(model.named_parameters())

    initialize_deepseek_v41_model(model, device=torch.device("cpu"), initialize_weights=False)

    assert all(parameter is parameters[name] for name, parameter in model.named_parameters())
    torch.testing.assert_close(model.right.weight, expected)
    torch.testing.assert_close(model.norm.weight, norm)


def test_materialized_model_accepts_strict_hf_weights(tmp_path, recwarn):
    model = nn.Module()
    model.model = _shared_model()
    model.model_args = SimpleNamespace(n_routed_experts=2, moe_inter_dim=3, vision_patch_size=2)
    expected = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    norm = torch.full((4,), 3.0, dtype=torch.bfloat16)
    save_file(
        {"left.weight": expected, "right.weight": expected.clone(), "norm.weight": norm},
        str(tmp_path / "model.safetensors"),
    )
    prepare_deepseek_v41_model_for_fsdp(model, device=torch.device("cpu"), parameter_dtype=torch.bfloat16)

    initialize_deepseek_v41_model(model, device=torch.device("cpu"), initialize_weights=False)
    load_deepseek_v41_from_pretrained(model, str(tmp_path), strict=True)

    assert not any(parameter.is_meta for parameter in model.parameters())
    assert model.model.left.weight is model.model.right.weight
    torch.testing.assert_close(model.model.left.weight, expected)
    torch.testing.assert_close(model.model.norm.weight, norm)
    assert not any("is a no-op" in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("has_local_storage", [False, True])
def test_materialization_leaves_host_engram_storage_untouched(has_local_storage):
    model = nn.Module()
    model.engram = HostOffloadEngram(2048, 128, layer_id=1, num_max_tokens_per_rank=16)
    if has_local_storage:
        # Represent a table whose EP hook has already allocated a small local shard.
        model.engram.weight = nn.Parameter(torch.full((2, 128), 7.0, dtype=torch.bfloat16))
    weight = model.engram.weight
    model.dense = nn.Linear(4, 3, bias=False, device="meta", dtype=torch.bfloat16)

    initialize_deepseek_v41_model(model, device=torch.device("cpu"), initialize_weights=False)

    assert model.engram.weight is weight
    assert model.engram.weight.is_meta == (not has_local_storage)
    assert not model.dense.weight.is_meta
    if has_local_storage:
        torch.testing.assert_close(weight, torch.full_like(weight, 7.0))


def _materialize_dtensor_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=init_method)
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        local = torch.empty(3, 4, device="meta", dtype=torch.bfloat16)
        model = nn.Module()
        model.weight = nn.Parameter(
            DTensor.from_local(local, mesh, (Shard(0),), shape=torch.Size((6, 4)), stride=(4, 1))
        )
        model.shared_weight = model.weight
        model.norm = nn.LayerNorm(4, bias=False, device="meta", dtype=torch.bfloat16)

        initialize_deepseek_v41_model(model, device=torch.device("cpu"), initialize_weights=False)

        assert isinstance(model.weight, DTensor)
        assert model.weight.device_mesh == mesh
        assert model.weight.placements == (Shard(0),)
        assert model.weight.shape == (6, 4)
        assert model.weight.to_local().shape == (3, 4)
        assert not model.weight.to_local().is_meta
        assert not model.norm.weight.is_meta
        assert model.weight is model.shared_weight
        with torch.no_grad():
            model.weight.to_local().fill_(rank + 1)
        weight = model.weight
        initialize_deepseek_v41_model(model, device=torch.device("cpu"), initialize_weights=False)
        assert model.weight is weight
        expected = torch.cat([torch.full((3, 4), owner + 1, dtype=torch.bfloat16) for owner in range(world_size)])
        torch.testing.assert_close(model.weight.full_tensor(), expected)
    finally:
        dist.destroy_process_group()


def test_materialization_preserves_two_rank_dtensor_shards(tmp_path):
    spawn(
        _materialize_dtensor_worker,
        args=(2, (tmp_path / "rendezvous").as_uri()),
        nprocs=2,
        join=True,
    )


def test_random_initialization_still_materializes_parameters():
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.empty(3, 4, device="meta")))

    initialize_deepseek_v41_model(model, standard_deviation=0.0, device=torch.device("cpu"))

    assert not model.weight.is_meta
    torch.testing.assert_close(model.weight, torch.zeros(3, 4))

# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in the LICENSE
# file in the torchtitan source repository.
"""Load DeepSeek-V4.1 HuggingFace weights into the FSDPTurbo model layout."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

try:
    from torch.distributed.tensor.placement_types import _StridedShard
except ImportError:
    _StridedShard = None

from fsdp_turbo.utils.log import print_rank

from .engram.host_offload_engram import HostOffloadEngram


logger = logging.getLogger(__name__)


_EXPERT_KEY = re.compile(r"^model\.layers\.(\d+)\.ffn\.experts\.(gate_up_proj|down_proj)$")
_HF_EXPERT_KEY = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$")


def _is_strided_shard(placement) -> bool:
    return _StridedShard is not None and isinstance(placement, _StridedShard)


class DeepSeekV41StateDictAdapter:
    """Convert between released HF keys and FSDPTurbo's packed model state.

    The released checkpoint stores one tensor per routed expert. FSDPTurbo
    packs them into ``gate_up_proj`` and ``down_proj``. The DTensor conversion
    below preserves EP/FSDP placements and exposes only owner-local experts to
    DCP, so loading does not all-gather a complete MoE layer.
    """

    def __init__(self, model_args):
        self.model_args = model_args
        self.num_experts = model_args.n_routed_experts
        self.intermediate_size = model_args.moe_inter_dim
        self._expert_metadata: dict[str, tuple[Any, Any, Any, tuple[int, int]]] = {}

    @staticmethod
    def _to_hf_key(key: str) -> str:
        if not key.startswith("model."):
            raise KeyError(f"Unexpected DeepSeek-V4.1 model key: {key}")
        return key.removeprefix("model.")

    @staticmethod
    def _to_local_key(key: str) -> str:
        return f"model.{key}"

    @staticmethod
    def _abstract_expert_key(key: str) -> str:
        return re.sub(r"^model\.layers\.\d+\.", "model.layers.{}.", key, count=1)

    @staticmethod
    def _expert_range(value: DTensor) -> tuple[int, int]:
        dim_size = value.shape[0]
        start, end = 0, dim_size
        placements_on_experts = []
        for mesh_dim, placement in enumerate(value.placements):
            if isinstance(placement, Shard) and placement.dim == 0:
                placements_on_experts.append((mesh_dim, placement))
            elif _is_strided_shard(placement) and placement.dim == 0:
                placements_on_experts.append((mesh_dim, placement))

        if not placements_on_experts:
            return start, end
        if len(placements_on_experts) == 1:
            mesh_dim, placement = placements_on_experts[0]
            if _is_strided_shard(placement):
                raise NotImplementedError("A lone strided expert shard is unsupported.")
            degree = value.device_mesh.size(mesh_dim)
            rank = value.device_mesh.get_local_rank(mesh_dim)
            if dim_size % degree:
                raise ValueError(f"{dim_size} experts cannot be evenly sharded over {degree} ranks.")
            block = dim_size // degree
            return rank * block, (rank + 1) * block
        if len(placements_on_experts) == 2:
            outer_dim, outer = placements_on_experts[0]
            inner_dim, inner = placements_on_experts[1]
            if not _is_strided_shard(outer) or _is_strided_shard(inner):
                raise NotImplementedError("Two-dimensional expert sharding requires _StridedShard(0), Shard(0).")
            outer_degree = value.device_mesh.size(outer_dim)
            inner_degree = value.device_mesh.size(inner_dim)
            if dim_size % (outer_degree * inner_degree):
                raise ValueError(
                    f"{dim_size} experts cannot be evenly sharded over {outer_degree * inner_degree} ranks."
                )
            block = dim_size // (outer_degree * inner_degree)
            index = outer_degree * value.device_mesh.get_local_rank(inner_dim) + value.device_mesh.get_local_rank(
                outer_dim
            )
            return index * block, (index + 1) * block
        raise NotImplementedError(f"Unsupported expert placements: {value.placements}")

    @staticmethod
    def _expert_submesh(value: DTensor):
        names = []
        placements = []
        mesh_names = value.device_mesh.mesh_dim_names
        if mesh_names is None:
            mesh_names = tuple(str(i) for i in range(value.device_mesh.ndim))
        for name, placement in zip(mesh_names, value.placements):
            if (isinstance(placement, Shard) or _is_strided_shard(placement)) and placement.dim == 0:
                continue
            names.append(name)
            if isinstance(placement, Replicate):
                placements.append(Replicate())
            elif isinstance(placement, Shard):
                placements.append(Shard(placement.dim))
            elif _is_strided_shard(placement):
                assert _StridedShard is not None
                placements.append(_StridedShard(placement.dim, split_factor=placement.split_factor))
            else:
                raise ValueError(f"Unsupported DTensor placement: {placement}")
        if not names:
            return None, ()
        if value.device_mesh.mesh_dim_names is None:
            # Current FSDPTurbo meshes are named. Fail clearly instead of
            # constructing a subtly different process group.
            raise ValueError("Expert DTensor loading requires a named DeviceMesh.")
        return value.device_mesh[tuple(names)], tuple(placements)

    def _split_expert_parameter(
        self,
        key: str,
        value: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        match = _EXPERT_KEY.fullmatch(key)
        if match is None:
            raise KeyError(key)
        layer_id, packed_name = match.groups()
        abstract_key = self._abstract_expert_key(key)

        if isinstance(value, DTensor):
            start, end = self._expert_range(value)
            local = value._local_tensor
            if local.shape[0] != end - start:
                raise ValueError(f"Local expert count for {key} is {local.shape[0]}, expected {end - start}.")
            if packed_name == "gate_up_proj" and local.shape[1] != 2 * self.intermediate_size:
                raise NotImplementedError(
                    "Loading packed gate_up_proj with expert-FSDP sharding on its "
                    "intermediate dimension is unsupported; set "
                    "expert_fully_shard_parallel_size=1 for HF initial loading."
                )
            submesh, subplacements = self._expert_submesh(value)
            self._expert_metadata[abstract_key] = (
                value.device_mesh,
                value.placements,
                value.shape,
                (start, end),
            )
        else:
            start, end = 0, value.shape[0]
            local = value
            submesh, subplacements = None, ()

        result = {}
        for expert_id in range(start, end):
            local_expert = local[expert_id - start]
            projections = (
                (
                    ("w1", local_expert[: self.intermediate_size]),
                    ("w3", local_expert[self.intermediate_size :]),
                )
                if packed_name == "gate_up_proj"
                else (("w2", local_expert),)
            )
            for projection, tensor in projections:
                if submesh is not None:
                    tensor = DTensor.from_local(
                        tensor.unsqueeze(0),
                        submesh,
                        subplacements,
                        run_check=False,
                    ).squeeze(0)
                result[f"layers.{layer_id}.ffn.experts.{expert_id}.{projection}.weight"] = tensor
        return result

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in state_dict.items():
            if _EXPERT_KEY.fullmatch(key):
                result.update(self._split_expert_parameter(key, value))
                continue
            hf_key = self._to_hf_key(key)
            if hf_key == "vision.patch_embed.proj.weight":
                patch = self.model_args.vision_patch_size
                value = value.reshape(value.shape[0], 3, patch, patch)
            result[hf_key] = value
        return result

    @staticmethod
    def _stack_loaded_experts(experts: dict[int, torch.Tensor]) -> torch.Tensor:
        values = [experts[index] for index in sorted(experts)]
        stacked = torch.stack(values)
        return stacked._local_tensor if isinstance(stacked, DTensor) else stacked

    def _pack_expert_parameter(
        self,
        abstract_key: str,
        projections: dict[str, dict[int, torch.Tensor]],
    ) -> torch.Tensor | None:
        metadata = self._expert_metadata.get(abstract_key)
        expected = self.num_experts if metadata is None else metadata[3][1] - metadata[3][0]
        required = ("w1", "w3") if abstract_key.endswith(".gate_up_proj") else ("w2",)
        if any(len(projections.get(name, {})) < expected for name in required):
            return None

        if abstract_key.endswith(".gate_up_proj"):
            local = torch.cat(
                (
                    self._stack_loaded_experts(projections["w1"]),
                    self._stack_loaded_experts(projections["w3"]),
                ),
                dim=1,
            )
        else:
            local = self._stack_loaded_experts(projections["w2"])

        if metadata is None:
            return local
        mesh, placements, _, _ = metadata
        return DTensor.from_local(local, mesh, placements, run_check=False)

    def from_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        result = {}
        experts: dict[str, dict[str, dict[int, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
        local_keys: dict[str, str] = {}

        for key, value in state_dict.items():
            match = _HF_EXPERT_KEY.fullmatch(key)
            if match is None:
                local_key = self._to_local_key(key)
                if key == "vision.patch_embed.proj.weight":
                    value = value.flatten(1)
                result[local_key] = value
                continue

            layer_id, expert_id, projection = match.groups()
            packed_name = "gate_up_proj" if projection in ("w1", "w3") else "down_proj"
            local_key = f"model.layers.{layer_id}.ffn.experts.{packed_name}"
            abstract_key = self._abstract_expert_key(local_key)
            experts[abstract_key][projection][int(expert_id)] = value
            local_keys[abstract_key] = local_key

        for abstract_key, projections in experts.items():
            packed = self._pack_expert_parameter(abstract_key, projections)
            if packed is None:
                raise ValueError(f"Incomplete routed expert weights for {local_keys[abstract_key]}.")
            result[local_keys[abstract_key]] = packed
        return result

    def expected_hf_keys(self, state_dict: dict[str, Any]) -> set[str]:
        expected = set()
        for key in state_dict:
            match = _EXPERT_KEY.fullmatch(key)
            if match is None:
                expected.add(self._to_hf_key(key))
                continue
            layer_id, packed_name = match.groups()
            projections = ("w1", "w3") if packed_name == "gate_up_proj" else ("w2",)
            expected.update(
                f"layers.{layer_id}.ffn.experts.{expert_id}.{projection}.weight"
                for expert_id in range(self.num_experts)
                for projection in projections
            )
        return expected


def _state_group(key: str) -> str:
    for pattern in (
        r"^model\.layers\.\d+",
        r"^model\.vision\.blocks\.\d+",
    ):
        match = re.match(pattern, key)
        if match is not None:
            return match.group(0)
    return "root"


def _is_ignored_hf_key(key: str) -> bool:
    # The source FP8 scales may remain beside already-dequantized BF16
    # weights. MTP/DSpark is inference-only and is deliberately absent from
    # the training model.
    return key.endswith(".scale") or key.startswith("mtp.")


def _checkpoint_tensor_size(metadata: dict[str, Any], key: str) -> tuple[int, ...] | None:
    stored = metadata.get(key)
    size = getattr(stored, "size", None)
    return None if size is None else tuple(size)


def _match_checkpoint_layout(
    key: str,
    value: torch.Tensor,
    saved_size: tuple[int, ...] | None,
) -> torch.Tensor:
    if key != "vision.patch_embed.proj.weight" or saved_size is None or len(saved_size) != 2 or value.ndim != 4:
        return value
    if value.shape[0] == saved_size[0] and value.numel() == math.prod(saved_size):
        return value.flatten(1)
    return value


def _checkpoint_weight_map(path: Path) -> tuple[dict[str, str], set[str]]:
    indexes = sorted(path.glob("*.safetensors.index.json"))
    if indexes:
        if len(indexes) != 1:
            raise ValueError(f"Expected one safetensors index in {path}, found {len(indexes)}.")
        with indexes[0].open("r", encoding="utf-8") as index_file:
            weight_map = json.load(index_file).get("weight_map", {})
        return weight_map, set(weight_map)

    from safetensors import safe_open

    weight_map = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in weight_map:
                    raise ValueError(f"Duplicate safetensors key {key!r} in {path}.")
                weight_map[key] = shard.name
    if not weight_map:
        raise FileNotFoundError(f"No HuggingFace safetensors checkpoint found in {path}.")
    return weight_map, set(weight_map)


@torch.no_grad()
def _load_engram_tables(
    tables: list[HostOffloadEngram],
    path: Path,
    weight_map: dict[str, str],
    *,
    rows_per_chunk: int,
) -> None:
    from safetensors import safe_open

    for table in tables:
        key = f"layers.{table.layer_id}.engram.embed.weight"
        print_rank(logger.info, f"> Loading Host Engram weight: {key}")
        filename = weight_map.get(key)
        if filename is None:
            raise KeyError(f"HuggingFace checkpoint is missing {key}.")
        with safe_open(str(path / filename), framework="pt", device="cpu") as handle:
            source = handle.get_slice(key)
            source_shape = tuple(source.get_shape())
            expected_shape = (table.logical_num_embeddings, table.embedding_dim)
            if source_shape != expected_shape:
                raise ValueError(f"Shape mismatch for {key}: expected {expected_shape}, got {source_shape}.")
            table.weight.zero_()
            source_end = min(table.row_start + table.local_rows, table.logical_num_embeddings)
            for start in range(table.row_start, source_end, rows_per_chunk):
                end = min(start + rows_per_chunk, source_end)
                table.weight[start - table.row_start : end - table.row_start].copy_(
                    source[start:end].to(dtype=table.weight.dtype)
                )
        if table._buffer is not None:
            table._register_storage()


def load_deepseek_v41_from_pretrained(
    model,
    pretrained_model_name_or_path: str,
    *,
    strict: bool = True,
    engram_rows_per_chunk: int = 65_536,
):
    """Load a dequantized BF16 HuggingFace checkpoint into a sharded model.

    The model must already have been parallelized by :class:`FSDPTurbo`.
    Dense and routed-expert tensors are loaded through DCP directly into their
    owner-local DTensor shards. Host Engram tables are streamed in row chunks.
    """
    if engram_rows_per_chunk <= 0:
        raise ValueError("engram_rows_per_chunk must be positive.")
    path = Path(pretrained_model_name_or_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Pretrained model directory not found: {path}")

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import HuggingFaceStorageReader
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    tables = [module for module in model.modules() if isinstance(module, HostOffloadEngram)]
    saved_table_weights = [(table, table._parameters.pop("weight")) for table in tables]
    weight_map: dict[str, str] = {}
    try:
        native_state = get_model_state_dict(model)
        adapter = DeepSeekV41StateDictAdapter(model.model_args)
        expected = adapter.expected_hf_keys(native_state)
        weight_map, checkpoint_keys = _checkpoint_weight_map(path)
        expected.update(f"layers.{table.layer_id}.engram.embed.weight" for table in tables)
        if strict:
            missing = sorted(expected - checkpoint_keys)
            unexpected = sorted(key for key in checkpoint_keys - expected if not _is_ignored_hf_key(key))
            if missing or unexpected:
                raise RuntimeError(
                    "HuggingFace checkpoint keys do not match the DeepSeek-V4.1 model: "
                    f"missing={missing[:20]}, unexpected={unexpected[:20]}."
                )

        groups: dict[str, dict[str, Any]] = defaultdict(dict)
        for key, value in native_state.items():
            groups[_state_group(key)][key] = value

        reader = HuggingFaceStorageReader(str(path))
        metadata = reader.read_metadata().state_dict_metadata
        float8_keys = []
        for key in expected.intersection(checkpoint_keys):
            tensor_metadata = metadata.get(key)
            properties = getattr(tensor_metadata, "properties", None)
            dtype = getattr(properties, "dtype", None)
            if str(dtype).startswith("torch.float8"):
                float8_keys.append(key)
        if float8_keys:
            raise ValueError(
                "from_pretrained requires dequantized weights, but float8 tensors "
                f"remain in the checkpoint: {sorted(float8_keys)[:20]}."
            )

        partial_options = StateDictOptions(strict=False)
        for group_name in sorted(groups, key=lambda name: (name == "root", name)):
            print_rank(logger.info, f"> Loading HuggingFace weight group: {group_name}")
            hf_state = adapter.to_hf(groups[group_name])
            hf_state = {
                key: _match_checkpoint_layout(
                    key,
                    value,
                    _checkpoint_tensor_size(metadata, key),
                )
                for key, value in hf_state.items()
            }
            dcp.load(hf_state, storage_reader=reader)
            loaded_state = adapter.from_hf(hf_state)
            set_model_state_dict(model, loaded_state, options=partial_options)
            del hf_state, loaded_state
    finally:
        for table, weight in saved_table_weights:
            table.register_parameter("weight", weight)

    _load_engram_tables(
        tables,
        path,
        weight_map,
        rows_per_chunk=engram_rows_per_chunk,
    )
    return model


__all__ = [
    "DeepSeekV41StateDictAdapter",
    "load_deepseek_v41_from_pretrained",
]

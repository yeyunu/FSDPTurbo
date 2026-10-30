"""Ascend Host-offloaded sparse embedding used by DeepSeek-V4.1 Engram.

The authoritative EP-local BF16 weight and FP32 Adam moments stay on CPU. CANN's
ElasticBuffer moves only requested rows to the accelerator and routes their
gradients back to the owning EP rank. This module is optional: importing it does
not import ``cann_ops_transformer``; selecting the backend without the operator
package fails with an actionable error while the regular PyTorch Engram path
remains available.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch._library.opaque_object import register_opaque_type

try:
    from torch._custom_class_base import CustomClassBase as _OpaqueHandleBase
except ImportError:
    try:
        from torch._opaque_base import OpaqueBase as _OpaqueHandleBase
    except ImportError:
        try:
            from torch._library.opaque_object import OpaqueBase as _OpaqueHandleBase
        except ImportError:
            _OpaqueHandleBase = object


from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)
_ROW_ALIGNMENT = 128
_TABLE_ROW_ALIGNMENT = 2048
_INT32_MAX = torch.iinfo(torch.int32).max
_ENGRAM_GROUPS: dict[tuple[tuple[int, ...], int], dist.ProcessGroup] = {}


def _dedicated_engram_group(ep_group, layer_id):
    """ElasticBuffer needs a distinct HCCL communicator for every table."""
    ranks = tuple(dist.get_process_group_ranks(ep_group))
    key = (ranks, layer_id)
    group = _ENGRAM_GROUPS.get(key)
    if group is None:
        group = dist.new_group(ranks=list(ranks), backend="hccl", use_local_synchronization=True)
        _ENGRAM_GROUPS[key] = group
    return group


class _BufferHandle(_OpaqueHandleBase):
    def __init__(self, value: Any = None):
        self.value = value
        self._fetch_contexts = {}
        self._next_fetch_token = 0

    def store_fetch_context(self, context):
        token = self._next_fetch_token
        self._next_fetch_token += 1
        self._fetch_contexts[token] = context
        return token

    def pop_fetch_context(self, token):
        try:
            return self._fetch_contexts.pop(token)
        except KeyError as error:
            raise RuntimeError(f"Unknown or already consumed Engram fetch token: {token}") from error

    def __eq__(self, other):
        return isinstance(other, _BufferHandle) and self.value is other.value

    def __hash__(self):
        return 0 if self.value is None else id(self.value)

    def __fx_repr__(self):
        return "_BufferHandle()", {"_BufferHandle": _BufferHandle}


class _TableHandle(_OpaqueHandleBase):
    def __init__(self, value: Any = None):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, _TableHandle) and self.value is other.value

    def __hash__(self):
        return 0 if self.value is None else id(self.value)

    def __fx_repr__(self):
        return "_TableHandle()", {"_TableHandle": _TableHandle}


register_opaque_type(_BufferHandle, typ="reference")
register_opaque_type(_TableHandle, typ="reference")


@torch.library.custom_op("fsdp_turbo_engram::fetch", mutates_args="unknown")
def _fetch(
    weight: torch.Tensor,
    indices: torch.Tensor,
    buffer: _BufferHandle,
) -> tuple[torch.Tensor, torch.Tensor]:
    fetched, fetch_context = buffer.value.engram_fetch(indices.contiguous())()
    token = buffer.store_fetch_context(fetch_context)
    # The token names Python-owned state and is never consumed by an accelerator
    # kernel. Keep it on the host so backward does not introduce an NPU-to-host
    # synchronization merely to recover a Python dictionary key.
    return fetched.contiguous(), torch.tensor(token, dtype=torch.int64, device="cpu")


@_fetch.register_fake
def _fetch_fake(weight, indices, buffer):
    return (
        torch.empty(
            indices.shape[0],
            weight.shape[1],
            dtype=weight.dtype,
            device=indices.device,
        ),
        torch.empty((), dtype=torch.int64, device="cpu"),
    )


@torch.library.custom_op("fsdp_turbo_engram::fetch_grad", mutates_args=())
def _fetch_grad(
    grad_fetched: torch.Tensor,
    fetch_token: torch.Tensor,
    buffer: _BufferHandle,
) -> tuple[torch.Tensor, torch.Tensor]:
    fetch_context = buffer.pop_fetch_context(int(fetch_token.item()))
    return buffer.value.engram_fetch_grad(grad_fetched.contiguous(), fetch_context)


@_fetch_grad.register_fake
def _fetch_grad_fake(grad_fetched, fetch_token, buffer):
    num_unique = torch.library.get_ctx().new_dynamic_size()
    return (
        grad_fetched.new_empty(num_unique, grad_fetched.shape[1]),
        grad_fetched.new_empty(num_unique, dtype=torch.int32),
    )


@torch.library.custom_op("fsdp_turbo_engram::accumulate_sparse_grad", mutates_args="unknown")
def _accumulate_sparse_grad(
    grad_rows: torch.Tensor,
    local_ids: torch.Tensor,
    table: _TableHandle,
) -> torch.Tensor:
    table.value.accumulate_sparse_gradient(local_ids, grad_rows)
    return grad_rows.new_zeros(())


@_accumulate_sparse_grad.register_fake
def _accumulate_sparse_grad_fake(grad_rows, local_ids, table):
    return grad_rows.new_empty(())


class _HostFetch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, row_ids, buffer, table, keepalive):
        fetched, fetch_token = torch.ops.fsdp_turbo_engram.fetch(
            weight,
            row_ids.reshape(-1).to(torch.int32),
            buffer,
        )
        ctx.buffer = buffer
        ctx.table = table
        ctx.row_dim = weight.shape[1]
        ctx.save_for_backward(fetch_token)
        return fetched

    @staticmethod
    def backward(ctx, grad_output):
        (fetch_token,) = ctx.saved_tensors
        grad_rows, local_ids = torch.ops.fsdp_turbo_engram.fetch_grad(
            grad_output.reshape(-1, ctx.row_dim),
            fetch_token,
            ctx.buffer,
        )
        # AOTAutograd may eliminate a backward containing only an opaque side
        # effect.  Returning this scalar as the keepalive gradient creates a
        # real data dependency without checkpointing the anchor.
        sink = torch.ops.fsdp_turbo_engram.accumulate_sparse_grad(grad_rows, local_ids, ctx.table)
        return None, None, None, None, sink


class HostOffloadEngram(nn.Module):
    """CPU BF16 Engram table sharded over EP and served by ElasticBuffer."""

    uses_host_offload = True
    requires_cann_alignment = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        layer_id: int,
        num_max_tokens_per_rank: int,
        pin_memory: bool = True,
    ):
        super().__init__()
        if num_max_tokens_per_rank <= 0:
            raise ValueError("Host-offload Engram requires a positive lookup capacity.")
        if self.requires_cann_alignment and embedding_dim % _ROW_ALIGNMENT:
            raise ValueError(
                f"CANN EngramFetch requires embedding_dim to be a multiple of {_ROW_ALIGNMENT}, got {embedding_dim}."
            )
        physical_num_embeddings = (
            (num_embeddings + _TABLE_ROW_ALIGNMENT - 1) // _TABLE_ROW_ALIGNMENT
        ) * _TABLE_ROW_ALIGNMENT
        if physical_num_embeddings > _INT32_MAX:
            raise ValueError("CANN EngramFetch row IDs are int32 but the table is larger than int32.")
        self.logical_num_embeddings = num_embeddings
        self.num_embeddings = physical_num_embeddings
        self.embedding_dim = embedding_dim
        self.layer_id = layer_id
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.pin_memory = pin_memory
        self.init_std = 0.02
        # Ownership is unknown until FSDPTurbo has constructed its EP mesh.
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings,
                embedding_dim,
                dtype=torch.bfloat16,
                device="meta",
            )
        )
        self.mesh = None
        self.row_start = 0
        self.local_rows = num_embeddings
        self._buffer = None
        self._buffer_handle = _BufferHandle()
        self._table_handle = _TableHandle(self)
        self._storage_registered = False
        self._pending_sparse_grad: torch.Tensor | None = None
        self._deferred_sparse_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._replica_group = None
        self._replica_size = 1
        self._grad_keepalive = torch.zeros((), requires_grad=True)

    @staticmethod
    def _elastic_buffer_type():
        try:
            from cann_ops_transformer import ElasticBuffer
        except ImportError as error:
            raise RuntimeError(
                "engram_storage_backend='elastic_buffer' requires a CANN operator package "
                "providing cann_ops_transformer.ElasticBuffer."
            ) from error
        return ElasticBuffer

    def shard_(self, mesh):
        if mesh is None or mesh.ndim != 1 or mesh.size() <= 1:
            raise ValueError("Host-offload Engram requires a one-dimensional EP mesh with EP > 1.")
        if self.mesh is not None:
            if self.mesh != mesh:
                raise ValueError("Host-offload Engram cannot change EP topology after materialization.")
            return self
        if self.num_embeddings % mesh.size():
            raise ValueError(f"Host Engram rows ({self.num_embeddings}) must be divisible by EP ({mesh.size()}).")
        self.mesh = mesh
        self.local_rows = self.num_embeddings // mesh.size()
        self.row_start = mesh.get_local_rank() * self.local_rows
        try:
            local = torch.empty(
                self.local_rows,
                self.embedding_dim,
                dtype=self.weight.dtype,
                device="cpu",
                pin_memory=self.pin_memory,
            )
        except RuntimeError:
            if not self.pin_memory:
                raise
            print_rank(logger.warning, "Pinned Host allocation is unavailable; allocating pageable Engram storage.")
            local = torch.empty(self.local_rows, self.embedding_dim, dtype=self.weight.dtype)
        generator = torch.Generator(device="cpu")
        seed = (torch.initial_seed() + self.layer_id + self.row_start) % (2**64)
        generator.manual_seed(seed)
        nn.init.normal_(local, std=self.init_std, generator=generator)
        self.weight = nn.Parameter(local, requires_grad=self.weight.requires_grad)
        self.weight._engram_host_offload = True
        self.weight._engram_checkpoint_suffix = self._checkpoint_suffix()
        self._grad_keepalive = torch.zeros((), device=mesh.device_type, requires_grad=True)
        self._init_buffer()
        return self

    def wire_replica_mesh(self, mesh):
        if mesh is None:
            return
        flat = mesh if mesh.ndim == 1 else mesh._flatten()
        if flat.size() > 1:
            self._replica_group = flat.get_group()
            self._replica_size = flat.size()

    def pre_fsdp_hook(self, parallel_state):
        """Materialize the owner-local table before its parent is FSDP-wrapped."""
        self.shard_(parallel_state.get_ep_device_mesh())
        replica_mesh = parallel_state.get_device_mesh("edp_efsdp")
        if replica_mesh.ndim != 1:
            replica_mesh = replica_mesh._flatten(mesh_dim_name="engram_replica")
        self.wire_replica_mesh(replica_mesh)

    @torch.no_grad()
    def reduce_sparse_gradient_across_replicas(self):
        """Sparse reduce-by-key over replicas of one EP-owned shard."""
        if self._replica_group is None:
            return
        pending = self._materialize_pending_sparse_grad()
        if pending is None:
            ids = torch.empty(0, dtype=torch.int64)
            values = torch.empty(0, self.embedding_dim, dtype=torch.float32)
        else:
            ids = pending.indices()[0]
            values = pending.values()
        device = self._grad_keepalive.device
        count = torch.tensor([ids.numel()], device=device, dtype=torch.int64)
        gathered_counts = [torch.empty_like(count) for _ in range(self._replica_size)]
        dist.all_gather(gathered_counts, count, group=self._replica_group)
        counts = [int(item.item()) for item in gathered_counts]
        max_rows = max(counts)
        if max_rows == 0:
            return
        padded_ids = torch.zeros(max_rows, dtype=torch.int64, device=device)
        padded_values = torch.zeros(max_rows, self.embedding_dim, dtype=torch.float32, device=device)
        padded_ids[: ids.numel()] = ids.to(device)
        padded_values[: values.shape[0]] = values.to(device)
        gathered_ids = [torch.empty_like(padded_ids) for _ in counts]
        gathered_values = [torch.empty_like(padded_values) for _ in counts]
        dist.all_gather(gathered_ids, padded_ids, group=self._replica_group)
        dist.all_gather(gathered_values, padded_values, group=self._replica_group)
        ids_parts = [part[:count].cpu() for part, count in zip(gathered_ids, counts) if count]
        value_parts = [part[:count].cpu() for part, count in zip(gathered_values, counts) if count]
        self._pending_sparse_grad = torch.sparse_coo_tensor(
            torch.cat(ids_parts).unsqueeze(0),
            torch.cat(value_parts),
            self.weight.shape,
            dtype=torch.float32,
        ).coalesce()

    def _checkpoint_suffix(self):
        if self.mesh is None:
            return ""
        return f".ep_shard_{self.mesh.get_local_rank():05d}_of_{self.mesh.size():05d}"

    def _init_buffer(self):
        buffer_type = self._elastic_buffer_type()
        storage_bytes = buffer_type.get_engram_storage_size_hint(self.local_rows, self.embedding_dim, self.weight.dtype)
        self._buffer = buffer_type(
            _dedicated_engram_group(self.mesh.get_group(), self.layer_id),
            num_cpu_bytes=storage_bytes,
            num_max_tokens_per_rank=self.num_max_tokens_per_rank,
            with_grad=True,
        )
        self._buffer_handle = _BufferHandle(self._buffer)
        self._register_storage()

    def _register_storage(self):
        storage = self.weight.detach()
        self._buffer.engram_write(storage)
        self._storage_registered = getattr(self._buffer, "_engram_storage_ref", None) is storage

    def _check_lookup_inputs(self, indices):
        if self.mesh is None or self.weight.is_meta:
            raise RuntimeError("Host Engram must be EP-sharded by FSDPTurbo before forward.")
        request_count = indices.numel()
        if request_count > self.num_max_tokens_per_rank:
            raise ValueError(f"Engram lookup has {request_count} rows but capacity is {self.num_max_tokens_per_rank}.")

    def forward(self, indices):
        self._check_lookup_inputs(indices)
        if not self._storage_registered:
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    "Compiled Host Engram requires CANN direct storage registration; "
                    "this operator package needs a per-forward engram_write."
                )
            self._register_storage()
        fetched = _HostFetch.apply(
            self.weight,
            indices.reshape(-1),
            self._buffer_handle,
            self._table_handle,
            self._grad_keepalive,
        )
        return fetched.reshape(*indices.shape, self.embedding_dim)

    def accumulate_sparse_gradient(self, local_ids: torch.Tensor, grad_rows: torch.Tensor) -> None:
        """Buffer one backward's sparse gradient on its accelerator tensors.

        The D2H copies, bounds checks and sparse coalescing are deferred to the
        optimizer/clip consumers so the backward main stream is not stalled
        waiting for host synchronization. Parts replay in arrival order,
        preserving the eager coalescing order and therefore the float32
        reduction result bit for bit.
        """
        self._deferred_sparse_parts.append((local_ids.reshape(-1), grad_rows.reshape(-1, self.embedding_dim)))

    def _materialize_pending_sparse_grad(self) -> torch.Tensor | None:
        """Replay buffered parts through the original eager coalescing path."""
        for local_ids, grad_rows in self._deferred_sparse_parts:
            ids = local_ids.to(device="cpu", dtype=torch.int64)
            values = grad_rows.to(device="cpu", dtype=torch.float32)
            if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.local_rows):
                raise IndexError("EngramFetchGrad returned a row outside the owner-local shard.")
            gradient = torch.sparse_coo_tensor(
                ids.unsqueeze(0),
                values,
                self.weight.shape,
                dtype=torch.float32,
                check_invariants=False,
            ).coalesce()
            self._pending_sparse_grad = (
                gradient if self._pending_sparse_grad is None else (self._pending_sparse_grad + gradient).coalesce()
            )
        self._deferred_sparse_parts.clear()
        return self._pending_sparse_grad

    def pending_sparse_grad(self) -> torch.Tensor | None:
        """Return the coalesced sparse gradient, materializing buffered parts.

        Returns ``None`` when no backward accumulated rows for this table.
        """
        return self._materialize_pending_sparse_grad()

    def scale_pending_sparse_grad(self, scale: float) -> None:
        """Materialize and multiply the pending sparse gradient by ``scale``."""
        gradient = self._materialize_pending_sparse_grad()
        if gradient is None:
            return
        self._pending_sparse_grad = torch.sparse_coo_tensor(
            gradient.indices(),
            gradient.values() * scale,
            gradient.shape,
            check_invariants=False,
        ).coalesce()

    def clear_sparse_gradient(self) -> None:
        """Drop buffered and materialized gradients and parameter grads."""
        self._pending_sparse_grad = None
        self._deferred_sparse_parts.clear()
        self.weight.grad = None
        self._grad_keepalive.grad = None

    def mark_sparse_step_complete(self) -> None:
        """Discard the step's sparse gradient after the sparse optimizer ran."""
        self._pending_sparse_grad = None
        self._deferred_sparse_parts.clear()

    @staticmethod
    def custom_clip_grad_norm(
        model: torch.nn.Module,
        engram_tables: list,
        max_norm: float,
        norm_type: int | float,
        group: dist.ProcessGroup | None,
    ) -> float:
        """Clip dense and pending Host Engram gradients by one global norm."""
        from fsdp_turbo.training.clip_grads import _to_local_if_dtensor, compute_grad_norm

        if norm_type != 2.0:
            raise NotImplementedError("Host Engram currently supports global 2-norm clipping only.")
        for table in sorted(engram_tables, key=lambda item: item.layer_id):
            table.reduce_sparse_gradient_across_replicas()

        if hasattr(model, "clip_grad_norm_"):
            dense_norm = model.clip_grad_norm_(float("inf"), norm_type)
        else:
            dense_parameters = [
                parameter for parameter in model.parameters() if not getattr(parameter, "_engram_host_offload", False)
            ]
            dense_norm = torch.tensor(compute_grad_norm(dense_parameters, norm_type, group=group))
        if not isinstance(dense_norm, torch.Tensor):
            dense_norm = torch.tensor(dense_norm)

        # FSDP may return a CPU scalar while its process group requires an
        # accelerator tensor. The keepalive tensor follows the Engram mesh.
        collective_device = (
            engram_tables[0]._grad_keepalive.device if engram_tables[0].mesh is not None else dense_norm.device
        )
        dense_norm = dense_norm.detach().to(device=collective_device, dtype=torch.float32)
        sparse_squared_norm = torch.zeros((), dtype=torch.float32, device=collective_device)
        for table in engram_tables:
            pending_gradient = table.pending_sparse_grad()
            if pending_gradient is not None:
                sparse_squared_norm += pending_gradient.values().float().square().sum().to(sparse_squared_norm.device)
        if engram_tables[0].mesh is not None:
            dist.all_reduce(sparse_squared_norm, group=engram_tables[0].mesh.get_group())

        total_norm = torch.sqrt(dense_norm.square() + sparse_squared_norm)
        clip_coefficient = min(float(max_norm / (total_norm + 1e-6)), 1.0)
        if clip_coefficient < 1.0:
            if hasattr(model, "clip_grad_norm_"):
                model.clip_grad_norm_(float(dense_norm) * clip_coefficient, norm_type)
            else:
                for parameter in dense_parameters:
                    if parameter.grad is not None:
                        _to_local_if_dtensor(parameter.grad).detach().mul_(clip_coefficient)
            for table in engram_tables:
                table.scale_pending_sparse_grad(clip_coefficient)
        return float(total_norm)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        suffix = self._checkpoint_suffix()
        if suffix and prefix + "weight" in destination:
            destination[prefix + "weight" + suffix] = destination.pop(prefix + "weight")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        suffix = self._checkpoint_suffix()
        shard_key = prefix + "weight" + suffix
        if suffix and shard_key in state_dict:
            state_dict[prefix + "weight"] = state_dict.pop(shard_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if self._buffer is not None:
            self._register_storage()


class _TorchHostFetch(torch.autograd.Function):
    """EP-routed CPU lookup using only PyTorch collectives and copies."""

    @staticmethod
    def forward(ctx, weight, row_ids, table, keepalive):
        flat_ids = row_ids.reshape(-1)
        if flat_ids.numel():
            invalid_ids = ((flat_ids < 0) | (flat_ids >= table.logical_num_embeddings)).any()
            if bool(invalid_ids):
                raise IndexError(f"Engram row IDs must be in [0, {table.logical_num_embeddings}).")

        unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
        owners = torch.div(unique_ids, table.local_rows, rounding_mode="floor")
        order = torch.argsort(owners)
        send_ids = unique_ids[order]
        send_counts = torch.bincount(owners, minlength=table.mesh.size())
        recv_counts = torch.empty_like(send_counts)
        group = table.mesh.get_group()
        dist.all_to_all_single(recv_counts, send_counts, group=group)
        # Materialize both count vectors with one D2H synchronization. These
        # lists are reused by every collective in forward and backward.
        split_values = torch.cat((send_counts, recv_counts)).to(device="cpu")
        split_values = split_values.numpy().tolist()
        mesh_size = table.mesh.size()
        send_sizes = split_values[:mesh_size]
        recv_sizes = split_values[mesh_size:]
        recv_ids = torch.empty(sum(recv_sizes), device=row_ids.device, dtype=row_ids.dtype)
        dist.all_to_all_single(
            recv_ids,
            send_ids,
            output_split_sizes=recv_sizes,
            input_split_sizes=send_sizes,
            group=group,
        )

        local_ids = (recv_ids - table.row_start).to(device="cpu", dtype=torch.long)
        if table.pin_memory and weight.is_pinned():
            values_cpu = torch.empty(
                local_ids.numel(),
                table.embedding_dim,
                dtype=weight.dtype,
                device="cpu",
                pin_memory=True,
            )
            torch.index_select(weight.detach(), 0, local_ids, out=values_cpu)
        else:
            values_cpu = weight.detach().index_select(0, local_ids)
        values = values_cpu.to(device=row_ids.device, non_blocking=table.pin_memory)
        selected_ordered = values.new_empty((sum(send_sizes), table.embedding_dim))
        dist.all_to_all_single(
            selected_ordered,
            values,
            output_split_sizes=send_sizes,
            input_split_sizes=recv_sizes,
            group=group,
        )
        selected = selected_ordered[torch.argsort(order)]

        ctx.table = table
        ctx.send_sizes = send_sizes
        ctx.recv_sizes = recv_sizes
        ctx.save_for_backward(inverse, order, recv_ids)
        return selected[inverse]

    @staticmethod
    def backward(ctx, grad_output):
        inverse, order, recv_ids = ctx.saved_tensors
        table = ctx.table
        grad_unique = grad_output.new_zeros((order.numel(), table.embedding_dim))
        grad_unique.index_add_(0, inverse, grad_output.reshape(-1, table.embedding_dim))
        grad_ordered = grad_unique[order]
        owner_grads = grad_output.new_empty((sum(ctx.recv_sizes), table.embedding_dim))
        dist.all_to_all_single(
            owner_grads,
            grad_ordered,
            output_split_sizes=ctx.recv_sizes,
            input_split_sizes=ctx.send_sizes,
            group=table.mesh.get_group(),
        )
        table.accumulate_sparse_gradient(
            recv_ids - table.row_start,
            owner_grads,
        )
        return None, None, None, grad_output.new_zeros(())


class TorchHostOffloadEngram(HostOffloadEngram):
    """CPU BF16 Engram table routed over EP without ElasticBuffer."""

    requires_cann_alignment = False

    def _init_buffer(self):
        # Lookups are explicitly staged through CPU and the existing EP group.
        return None

    def _register_storage(self):
        return None

    @torch.compiler.disable
    def forward(self, indices):
        self._check_lookup_inputs(indices)
        fetched = _TorchHostFetch.apply(
            self.weight,
            indices.reshape(-1),
            self,
            self._grad_keepalive,
        )
        return fetched.reshape(*indices.shape, self.embedding_dim)


def host_engram_tables(model):
    return [module for module in model.modules() if getattr(module, "uses_host_offload", False)]


__all__ = [
    "HostOffloadEngram",
    "TorchHostOffloadEngram",
    "host_engram_tables",
]

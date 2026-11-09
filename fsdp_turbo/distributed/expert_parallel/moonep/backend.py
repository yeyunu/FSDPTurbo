# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Device backends for the MoonEP expert-parallel dispatcher.

The algorithm (Kimi K3 report S5.2.1) is device independent; what is not is how
the symmetric expert-weight memory is built and how the three MoonEP kernels are
launched. A backend therefore owns the library handle, the symmetric resources
of one projection or projection pool, and the two collective kernel launches.

Notation: ``S`` tokens/rank, ``K`` top-k, ``E`` routed experts, ``R`` EP ranks,
``epn = E/R``, ``B = epn`` prefetch slots/rank, ``NvS`` dispatched rows/rank.
"""

from __future__ import annotations

import abc
import importlib
import importlib.metadata
import logging
import os
import socket
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from fsdp_turbo.distributed.expert_parallel.moonep.runtime import MoonEPRuntime
logger = logging.getLogger(__name__)

_SUPPORTED_CUDA_MOONEP_VERSION = "0.0.1"


class ProjectionPoolResources(abc.ABC):
    """Symmetric resources shared by every layer with the same matrix shape.

    Sharing is what keeps the memory footprint flat in the number of MoE layers,
    at the cost of replaying the prefetch before each layer's backward GEMMs.
    """

    #: ``[epn, out, in]`` FP32 home gradients this rank owns and reduces into.
    owner_grad_local: torch.Tensor
    #: ``[B, out, in]`` FP32 slot gradients the peers' reductions read.
    reduce_slots_local: torch.Tensor

    @abc.abstractmethod
    def close(self) -> None:
        """Release the pooled symmetric resources."""


class ProjectionResources(abc.ABC):
    """Per-layer, per-projection symmetric weight resources."""

    #: ``[E+B, out, in]`` view handed to the MoonEP prefetch entry point.
    full_weight: torch.Tensor
    #: Expert GEMM operand, laid out per the backend's ``expert_weight_layout``.
    gemm_weight: torch.Tensor
    #: ``[epn, out, in]`` view of this rank's home experts; the parameter payload.
    home_weight: torch.Tensor
    #: The pool this projection draws its prefetch slots and FP32 buffers from.
    pool: ProjectionPoolResources

    @abc.abstractmethod
    def close(self) -> None:
        """Release the per-projection symmetric resources."""


class MoonEPBackend(abc.ABC):
    """Device-specific half of the MoonEP integration."""

    #: ``torch`` device type this backend drives.
    device_type: str
    #: Whether MoonEP exposes a usable asynchronous communication stream.
    supports_async_comm: bool
    #: Whether ``S``/``K`` must be known before the first forward.
    requires_preconfigured_static_shape: bool
    #: ``"full"`` when the whole ``[E+B]`` mapping is readable, else ``"compact"``.
    expert_weight_layout: str
    #: Whether MoonEP leaves the rows past the last live group undefined.
    has_static_tail: bool

    @abc.abstractmethod
    def validate_environment(self, runtime: "MoonEPRuntime") -> None:
        """Fail fast when the accelerator or topology cannot run MoonEP."""

    @abc.abstractmethod
    def create_buffer(self, runtime: "MoonEPRuntime", tokens_per_rank: int, hidden: int, top_k: int) -> Any:
        """Create the MoonEP communication buffer for one static token shape."""

    def prepare_buffer(self, runtime: "MoonEPRuntime", buffer: Any) -> None:
        """Hook for one-time work on a freshly created communication buffer."""

    @abc.abstractmethod
    def create_projection_pool(
        self, runtime: "MoonEPRuntime", role: str, out_features: int, in_features: int
    ) -> ProjectionPoolResources:
        """Allocate the prefetch slots and FP32 reduction buffers for a shape."""

    @abc.abstractmethod
    def create_projection(
        self, runtime: "MoonEPRuntime", pool: ProjectionPoolResources, out_features: int, in_features: int
    ) -> ProjectionResources:
        """Allocate one layer's home experts and build its weight views."""

    @abc.abstractmethod
    def prefetch(self, runtime: "MoonEPRuntime", projection: ProjectionResources, buffer: Any, plan: Any) -> None:
        """Copy this step's redundant expert weights into the prefetch slots."""

    # Three phases: the kernel reads every peer's slot gradients before its own
    # internal barrier.
    def stage_gradient(
        self,
        runtime: "MoonEPRuntime",
        projection: ProjectionResources,
        home_grad: torch.Tensor,
        slot_grad: torch.Tensor,
    ) -> None:
        """Publish ``[epn, out, in]`` home and ``[B, out, in]`` slot gradients.

        The copies are plain device writes into the pool's FP32 buffers, which
        both backends allocate symmetrically, so the publication itself is
        device independent even though the two allocators are not.
        """
        pool = projection.pool
        pool.owner_grad_local.copy_(home_grad)
        pool.reduce_slots_local.copy_(slot_grad)

    @abc.abstractmethod
    def rendezvous_before_reduce(self, runtime: "MoonEPRuntime", buffer: Any) -> None:
        """Make every rank's staged gradients visible to every other rank.

        Abstract rather than a default no-op: a backend that forgets it does not
        lose a feature, it gets a cross-rank data race in which the reduction
        reads slot buffers its peers have not written yet.
        """

    @abc.abstractmethod
    def reduce_staged_gradient(
        self,
        runtime: "MoonEPRuntime",
        projection: ProjectionResources,
        buffer: Any,
        plan: Any,
    ) -> torch.Tensor:
        """Fold the staged slot gradients back into their home ranks.

        Returns the ``[epn, out, in]`` FP32 gradient for this rank's experts,
        including every other rank's contributions.
        """

    def initialize_symmetric_heap(self, runtime: "MoonEPRuntime") -> None:
        """Hook for backends with a one-shot collective heap initialisation."""

    def prewarm_collectives(self, runtime: "MoonEPRuntime", buffer: Any) -> None:
        """Hook for compiling collective kernels before the first real step."""

    def release(self, runtime: "MoonEPRuntime") -> None:
        """Hook for backend-wide teardown that must precede resource release."""

    def finish_release(self, runtime: "MoonEPRuntime") -> None:
        """Backend cleanup after all projections, pools and buffers are freed."""


class _CudaChunk:
    """One CUDA VMM allocation plus the IPC file descriptor that exports it."""

    def __init__(self, imports, shape: list[int], dtype: torch.dtype):
        keepalive, fd, owned_handle = imports.nvl_dist_alloc(shape=shape, dtype=dtype)
        # Only needed to bind a multicast object; ``keepalive`` holds the mapping.
        imports.nvl_release_mem_handle(owned_handle)
        self.keepalive = keepalive
        self._fd: int | None = fd

    def duplicate_fd(self) -> int:
        if self._fd is None:
            raise RuntimeError("MoonEP symmetric chunk is already released.")
        return os.dup(self._fd)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.keepalive = None


class _CudaProjectionPool(ProjectionPoolResources):
    """Prefetch slots plus the two FP32 reduction mappings for one shape.

    ``owner_grad`` is a contiguous ``[E, out, in]`` mapping assembled from one
    ``[epn, out, in]`` allocation per rank, which is the global indexing the
    reduction kernel expects while every rank only backs its own ``epn`` rows.
    ``reduce_slots`` is the matching ``[R, B, out, in]`` mapping that lets each
    rank read every peer's slot gradients.
    """

    def __init__(self, backend: "CudaMoonEPBackend", runtime: "MoonEPRuntime", out_features: int, in_features: int):
        imports = backend.imports
        epn = runtime.experts_per_rank
        chunk_shape = [epn, out_features, in_features]

        self.slot_chunk = _CudaChunk(imports, chunk_shape, torch.bfloat16)
        # MoonEP keeps the allocation alive via a ``_keepalive`` attribute on the
        # tensor it returns, which a reshape drops: hold as returned, view apart.
        self.owner_grad = imports.create_nvl_dist_tensor(
            chunk_shape, torch.float32, runtime.rank, runtime.size, group=runtime.group
        )
        self.reduce_flat = imports.create_nvl_dist_tensor(
            chunk_shape, torch.float32, runtime.rank, runtime.size, group=runtime.group
        )
        self.reduce_slots = self.reduce_flat.view(runtime.size, epn, out_features, in_features)
        self.owner_grad_local = self.owner_grad[runtime.rank * epn : (runtime.rank + 1) * epn]
        self.reduce_slots_local = self.reduce_slots[runtime.rank]
        # Peers read these directly, so they must not start out uninitialised.
        self.owner_grad_local.zero_()
        self.reduce_slots_local.zero_()
        dist.barrier(group=runtime.group)

    def close(self) -> None:
        # Views before owners: the owning tensors carry MoonEP's ``_keepalive``.
        self.owner_grad_local = None
        self.reduce_slots_local = None
        self.reduce_slots = None
        self.owner_grad = None
        self.reduce_flat = None
        self.slot_chunk.close()


class _CudaProjection(ProjectionResources):
    def __init__(
        self,
        backend: "CudaMoonEPBackend",
        runtime: "MoonEPRuntime",
        pool: _CudaProjectionPool,
        out_features: int,
        in_features: int,
    ):
        imports = backend.imports
        epn = runtime.experts_per_rank
        chunk_shape = [epn, out_features, in_features]
        self.pool = pool
        self.home_chunk = _CudaChunk(imports, chunk_shape, torch.bfloat16)

        local_fd = self.home_chunk.duplicate_fd()
        try:
            exchanged = imports.exchange_ipc_fds(
                local_fd, list(range(runtime.size)), runtime.rank, runtime.size, runtime.group
            )
        finally:
            os.close(local_fd)

        # ``nvl_dist_map`` imports each descriptor into its own virtual mapping,
        # so every file descriptor handed to it is ours to close afterwards.
        expert_fds = [exchanged[peer] for peer in range(runtime.size)]
        slot_fd = pool.slot_chunk.duplicate_fd()
        try:
            # R remote expert blocks then this rank's slots. Only the local block
            # is physical; the rest are NVLink mappings, which is what makes
            # "zero redundant storage, globally readable" possible. Indexed by
            # global expert id, as MoonEP's cu_seqlens is.
            self.full_weight = imports.nvl_dist_map(
                chunk_shape=chunk_shape,
                dtype=torch.bfloat16,
                fds=expert_fds + [slot_fd],
                local_rank=runtime.rank,
                world_size=runtime.size + 1,
            )
        finally:
            for fd in expert_fds:
                os.close(fd)
            os.close(slot_fd)

        num_experts = runtime.num_experts
        self.home_weight = self.full_weight[runtime.rank * epn : (runtime.rank + 1) * epn]
        self.gemm_weight = self.full_weight
        # Unplanned slots are still covered by a GEMM group.
        self.full_weight[num_experts:].zero_()

    def close(self) -> None:
        self.home_weight = None
        self.gemm_weight = None
        self.full_weight = None
        self.home_chunk.close()


class CudaMoonEPBackend(MoonEPBackend):
    """MoonEP on NVLink/NVSwitch GPUs via the ``moonep`` CUDA extension."""

    device_type = "cuda"
    supports_async_comm = True
    requires_preconfigured_static_shape = False
    # The mapping covers every expert row, so the GEMMs can use it directly.
    expert_weight_layout = "full"
    has_static_tail = True

    def __init__(self):
        self.imports = _load_cuda_moonep()

    def validate_environment(self, runtime: "MoonEPRuntime") -> None:
        if not torch.accelerator.is_available():
            raise RuntimeError("The MoonEP CUDA backend requires an available CUDA device.")
        _validate_single_host(runtime.group)
        if not self.imports.nvl_multicast_supported():
            raise RuntimeError(
                "MoonEP requires CUDA multicast support (NVLink/NVSwitch). Use "
                "another EP dispatcher on PCIe-only or unsupported GPUs."
            )

    def create_buffer(self, runtime: "MoonEPRuntime", tokens_per_rank: int, hidden: int, top_k: int) -> Any:
        config = runtime.config
        return self.imports.Buffer(
            S=tokens_per_rank,
            H=hidden,
            K=top_k,
            E=runtime.num_experts,
            num_ep_ranks=runtime.size,
            B=runtime.prefetch_slots,
            num_sms=config.num_sms,
            token_padding=config.token_padding,
            group=runtime.group,
            comm_stream_priority=config.comm_stream_priority,
            # Programmatic dependent launch: what the kernels are written for.
            enable_pdl=True,
            explicitly_destroy=True,
        )

    def prepare_buffer(self, runtime: "MoonEPRuntime", buffer: Any) -> None:
        """Clear the symmetric hidden buffer once so its tail is finite.

        MoonEP leaves the rows past the last padded group undefined. The tail
        never reaches the combine output, but zeroing freshly allocated device
        memory once keeps NaN bit patterns out of the expert GEMMs entirely.
        """
        view = getattr(buffer, "hidden_nvsh_buffer_view", None)
        if view is not None:
            view.zero_()
            dist.barrier(group=runtime.group)

    def create_projection_pool(
        self, runtime: "MoonEPRuntime", role: str, out_features: int, in_features: int
    ) -> ProjectionPoolResources:
        validate_chunk_granularity(
            role,
            runtime.experts_per_rank,
            out_features,
            in_features,
            torch.bfloat16.itemsize,
            int(self.imports.get_vmm_granularity()),
        )
        return _CudaProjectionPool(self, runtime, out_features, in_features)

    def create_projection(
        self, runtime: "MoonEPRuntime", pool: ProjectionPoolResources, out_features: int, in_features: int
    ) -> ProjectionResources:
        return _CudaProjection(self, runtime, pool, out_features, in_features)

    def prefetch(self, runtime: "MoonEPRuntime", projection: ProjectionResources, buffer: Any, plan: Any) -> None:
        num_experts = runtime.num_experts
        self.imports.launch_prefetch(
            projection.full_weight[:num_experts],
            projection.full_weight[num_experts:],
            plan.experts_to_copy[runtime.rank],
            num_sms=runtime.config.num_sms,
        )

    def prewarm_collectives(self, runtime: "MoonEPRuntime", buffer: Any) -> None:
        """Run one synthetic expert step so every kernel is compiled up front.

        A CuTe DSL kernel first compiled inside the backward pass deadlocks: the
        host blocks in ``_load_cuda_library`` while a spinning cross-rank barrier
        occupies the device and every peer sits in the same place. Replays a whole
        step rather than each kernel, because the backward uses different JIT
        specializations of the same entry points.

        Everything it touches is scratch; the reduction runs with an empty plan.
        """
        context = buffer._require_ctx()
        device = context["meta_buf"].device
        tokens, hidden_size, top_k = runtime.static_shape
        num_experts = runtime.num_experts

        # A valid, perfectly balanced routing: token i takes experts i..i+K-1.
        offsets = torch.arange(top_k, dtype=torch.int64, device=device)
        topk = (
            ((torch.arange(tokens, dtype=torch.int64, device=device).unsqueeze(1) + offsets) % num_experts)
            .to(torch.int32)
            .contiguous()
        )
        tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=device)
        flat = topk.reshape(-1).to(torch.int64)
        tokens_per_expert.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
        hidden = torch.zeros(tokens, hidden_size, dtype=torch.bfloat16, device=device)
        route_weights = torch.zeros(tokens, top_k, dtype=torch.float32, device=device)

        # Forward variants, and the plan the backward variants need.
        dispatched, route_nvs, _, plan = buffer.dispatch(
            hidden, route_weights, topk, tokens_per_expert, async_finish=False, zero_copy=False
        )
        # Backward variant: dispatch replaying an existing plan.
        buffer.dispatch(hidden, plan=plan, async_finish=False, zero_copy=False)
        # Forward variant: combine without router weights.
        buffer.combine(plan=plan, hidden_nvsh=dispatched, async_finish=False, zero_copy=False)
        # Backward variant: combine gathering router weights back to [S, K].
        buffer.combine(
            plan=plan,
            hidden_nvsh=dispatched,
            route_weights_nvs=route_nvs,
            async_finish=False,
            zero_copy=False,
        )

        empty_plan = torch.full((runtime.size, runtime.prefetch_slots), -1, dtype=torch.int32, device=device)
        for pool in runtime.projection_pools():
            self.imports.launch_inter_rank_sync(context)
            self.imports.launch_grad_reduce(
                pool.owner_grad,
                pool.reduce_slots,
                empty_plan,
                rank=runtime.rank,
                num_sms=runtime.config.num_sms,
                meta_buf=context["meta_buf"],
                meta_stride=int(context["meta_chunk_padded"]),
                barrier_off=int(context["BARRIER_OFF"]),
                grid_sync_bar=context["grid_sync_bar"],
            )
        for projection in runtime.projections():
            projection.prefetch(buffer, plan)

        runtime.device_module.current_stream().synchronize()
        dist.barrier(group=runtime.group)

    def rendezvous_before_reduce(self, runtime: "MoonEPRuntime", buffer: Any) -> None:
        # Enough on CUDA: the staging copies are ordered ahead on the same stream.
        self.imports.launch_inter_rank_sync(buffer._require_ctx())

    def reduce_staged_gradient(
        self,
        runtime: "MoonEPRuntime",
        projection: ProjectionResources,
        buffer: Any,
        plan: Any,
    ) -> torch.Tensor:
        pool = projection.pool
        comm_context = buffer._require_ctx()
        self.imports.launch_grad_reduce(
            pool.owner_grad,
            pool.reduce_slots,
            plan.experts_to_copy,
            rank=runtime.rank,
            num_sms=runtime.config.num_sms,
            meta_buf=comm_context["meta_buf"],
            meta_stride=int(comm_context["meta_chunk_padded"]),
            barrier_off=int(comm_context["BARRIER_OFF"]),
            grid_sync_bar=comm_context["grid_sync_bar"],
        )
        return pool.owner_grad_local


def _load_cuda_moonep():
    """Import the CUDA MoonEP extension lazily and pin its interface version."""
    from dataclasses import dataclass

    @dataclass
    class _Imports:
        Buffer: Any
        launch_prefetch: Any
        launch_grad_reduce: Any
        launch_inter_rank_sync: Any
        create_nvl_dist_tensor: Any
        exchange_ipc_fds: Any
        nvl_dist_alloc: Any
        nvl_release_mem_handle: Any
        nvl_dist_map: Any
        get_vmm_granularity: Any
        nvl_multicast_supported: Any

    try:
        version = importlib.metadata.version("moonep")
        from moonep import Buffer
        from moonep.buffer import _exchange_ipc_fds, create_nvl_dist_tensor
        from moonep.grad_reduce import launch_grad_reduce
        from moonep.inter_rank_sync import launch_inter_rank_sync
        from moonep.prefetch import launch_prefetch
        from moonep._C import (
            get_vmm_granularity,
            nvl_dist_alloc,
            nvl_dist_map,
            nvl_multicast_supported,
            nvl_release_mem_handle,
        )
    except (ImportError, ModuleNotFoundError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            f"MoonEP dispatcher requires moonep=={_SUPPORTED_CUDA_MOONEP_VERSION}, which "
            "is not published on any index: install the sibling repository with "
            "`pip install -e ../MoonEP`."
        ) from exc

    if version != _SUPPORTED_CUDA_MOONEP_VERSION:
        raise RuntimeError(
            f"Unsupported MoonEP version {version!r}; this adapter requires "
            f"{_SUPPORTED_CUDA_MOONEP_VERSION!r} because it uses the versioned VMM "
            "and kernel-launcher interfaces."
        )

    return _Imports(
        Buffer=Buffer,
        launch_prefetch=launch_prefetch,
        launch_grad_reduce=launch_grad_reduce,
        launch_inter_rank_sync=launch_inter_rank_sync,
        create_nvl_dist_tensor=create_nvl_dist_tensor,
        exchange_ipc_fds=_exchange_ipc_fds,
        nvl_dist_alloc=nvl_dist_alloc,
        nvl_release_mem_handle=nvl_release_mem_handle,
        nvl_dist_map=nvl_dist_map,
        get_vmm_granularity=get_vmm_granularity,
        nvl_multicast_supported=nvl_multicast_supported,
    )


def validate_chunk_granularity(
    role: str,
    experts_per_rank: int,
    out_features: int,
    in_features: int,
    item_size: int,
    granularity: int,
) -> None:
    """Require the per-rank expert block to be virtual-memory aligned.

    The compact ``[epn+B]`` GEMM view maps the home block and the prefetch slots
    into adjacent virtual addresses. That view is only row-exact when each
    block's byte size is a whole number of allocation granules; otherwise the
    allocator's padding would shift the slot rows.
    """
    chunk_bytes = experts_per_rank * out_features * in_features * item_size
    if chunk_bytes % granularity:
        raise RuntimeError(
            f"MoonEP {role} per-rank expert block is {chunk_bytes} bytes, which is not a "
            f"multiple of the {granularity}-byte virtual-memory granularity. Choose expert "
            "matrix dimensions such that (E/R) * out_features * in_features * itemsize is "
            "granularity aligned."
        )


def _validate_single_host(group) -> None:
    """MoonEP's symmetric memory is intra-node on CUDA; reject split groups."""
    hosts = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    if len(set(hosts)) != 1:
        raise RuntimeError(
            "MoonEP only supports an intra-node EP group on CUDA; group members "
            f"were found on hosts {sorted(set(hosts))}."
        )


_BACKEND_FACTORIES: dict[str, str] = {
    "cuda": "fsdp_turbo.distributed.expert_parallel.moonep.backend:CudaMoonEPBackend",
    "npu": "fsdp_turbo.distributed.expert_parallel.moonep.backend_npu:NpuMoonEPBackend",
}


def get_moonep_backend(device_type: str) -> MoonEPBackend:
    """Instantiate the MoonEP backend for ``device_type``.

    Backends are resolved lazily so that a CUDA installation never imports
    ``npu_patch`` and an Ascend installation never imports ``moonep`` (one
    distribution now ships both packages, but only one of them is built).
    """
    target = _BACKEND_FACTORIES.get(device_type)
    if target is None:
        raise RuntimeError(
            f"No MoonEP backend for device type {device_type!r}; supported types are {sorted(_BACKEND_FACTORIES)}."
        )
    module_name, class_name = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"Failed to import the MoonEP {device_type} backend: {exc}") from exc
    return getattr(module, class_name)()

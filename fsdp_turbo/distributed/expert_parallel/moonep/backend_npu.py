# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Ascend backend for the MoonEP expert-parallel dispatcher.

``npu_patch`` keeps the ``moonep`` Python contract but replaces CUDA's NVLink
virtual memory with CANN VMM chunks published into a CANN SHMEM symmetric heap.
Three consequences shape this backend:

* every chunk the kernels address must be registered before the single
  collective heap initialization, hence ``requires_preconfigured_static_shape``;
* the global ``[E+B]`` projection maps only this rank's home block and prefetch
  slots, so the expert GEMM must use the compact view -- touching a remote row
  would fault. ``npu_patch`` dropped its ``SegmentedVAMapping`` helper, so that
  view is assembled here out of the ``_C`` VMM primitives;
* the kernels carry device-side all-rank barriers that desynchronise on a side
  stream, so ``supports_async_comm`` is ``False`` and the collectives run on the
  compute stream.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import socket
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from fsdp_turbo.distributed.expert_parallel.moonep.backend import (
    MoonEPBackend,
    ProjectionPoolResources,
    ProjectionResources,
    validate_chunk_granularity,
)
from fsdp_turbo.utils.device import get_accelerator_name

if TYPE_CHECKING:
    from fsdp_turbo.distributed.expert_parallel.moonep.runtime import MoonEPRuntime
logger = logging.getLogger(__name__)


#: ``npu_patch._C`` entry points this backend calls directly. The VMM primitives
#: build the compact GEMM view, and the barrier is the stream-ordered rendezvous
#: before the gradient reduction. They are taken from ``_C`` rather than through
#: ``buffer_c``, whose re-exports are incidental: ``buffer_c`` used to forward
#: ``acl_shmem_barrier_all_on_stream`` and stopped.
_REQUIRED_NATIVE = (
    "acl_vmm_granularity",
    "acl_vmm_reserve",
    "acl_vmm_release",
    "acl_vmm_map",
    "acl_vmm_unmap",
    "acl_vmm_set_access",
    "acl_shmem_barrier_all_on_stream",
)

#: ``npu_patch.buffer_c`` entry points that build and tag the symmetric chunks.
_REQUIRED_BUFFER_C = (
    "create_vmm_physical",
    "create_full_projection_tensor",
    "create_sym_tensor_from_ptr",
    "attach_local_source_vas",
)


class _AscendImports:
    """Lazily resolved handles into ``npu_patch``."""

    def __init__(self):
        try:
            api = importlib.import_module("npu_patch")
            buffer_c = importlib.import_module("npu_patch.buffer_c")
            native = importlib.import_module("npu_patch._C")
            # Private on purpose: the mask-to-card-id mapping is deliberately
            # kept in one place in npu_patch, and re-deriving it here is how the
            # two VMM id spaces get mixed up (see ``_NpuSegmentedMapping``).
            device = importlib.import_module("npu_patch._device")
        except ImportError as exc:
            raise RuntimeError(
                "The MoonEP dispatcher on Ascend requires the `npu_patch` package from "
                "the MoonEP repository, which is not published on any index: build it "
                "there with `bash npu_patch/scripts/build_base.sh -soc_type <SOC_TYPE>` "
                "followed by `MOONEP_BUILD_BACKEND=npu pip install -e npu_patch`."
            ) from exc

        missing = (
            [name for name in ("Buffer", "ShmemRuntime") if not hasattr(api, name)]
            + [name for name in _REQUIRED_BUFFER_C if not hasattr(buffer_c, name)]
            + [name for name in _REQUIRED_NATIVE if not hasattr(native, name)]
        )
        if missing:
            raise RuntimeError(
                f"The installed npu_patch build is missing {missing}; this adapter "
                "targets the buffer-backed symmetric-heap interface (PR-653)."
            )

        self.Buffer = api.Buffer
        self.ShmemRuntime = api.ShmemRuntime
        self.buffer_c = buffer_c
        self.native = native
        self.local_physical_id = device.local_physical_id


def _probe_vmm_in_place(imports: "_AscendImports", device: int) -> None:
    """Sweep CANN VMM sizes from inside the process that is about to fail.

    A standalone probe mapped every size a real run fails at, so the sweep has to
    happen in this process state. If small sizes also fail here the size is
    irrelevant and VMM is broken for this process; if only large ones do, compare
    the reported VAs.

    Enabled with ``moonep_config.vmm_probe_enabled``, logged at warning level.
    """
    columns = 1024
    element_size = torch.empty((), dtype=torch.bfloat16).element_size()
    logger.warning("vmm_probe device=%d: %s", device, _describe_device_memory(device))
    for mib in (2, 8, 32, 64, 128, 160, 192, 256, 384):
        rows = (mib * 1024 * 1024) // (columns * element_size)
        try:
            tensor, _handle, _shareable = imports.buffer_c.create_vmm_physical(
                [1, rows, columns], torch.bfloat16, device
            )
        except Exception as error:  # noqa: BLE001 - this is a diagnostic sweep
            logger.warning(
                "vmm_probe   %4d MiB  FAILED  %s",
                mib,
                str(error).strip().splitlines()[0],
            )
            continue
        allocation = tensor._vmm_allocation
        logger.warning(
            "vmm_probe   %4d MiB  ok  source_va=0x%x size=%d",
            mib,
            int(allocation.source_va),
            int(allocation.size),
        )
        allocation.destroy()


def _describe_device_memory(device: int) -> str:
    """Torch's accounting next to the driver's, for a VMM failure message."""
    parts = []
    try:
        handle = torch.device(get_accelerator_name(), device)
        parts.append(f"torch allocated={torch.accelerator.memory_allocated(handle) / 2**30:.2f}GiB")
        parts.append(f"reserved={torch.accelerator.memory_reserved(handle) / 2**30:.2f}GiB")
    except Exception:  # pragma: no cover - diagnostics must never mask the cause
        pass
    try:
        free, total = torch.accelerator.mem_get_info(torch.device(get_accelerator_name(), device))
        parts.append(f"driver free={free / 2**30:.2f}GiB of {total / 2**30:.2f}GiB")
    except Exception:  # pragma: no cover
        pass
    return ", ".join(parts) if parts else "device memory figures unavailable"


class _NpuChunk:
    """One registered CANN VMM chunk plus its granularity-exact byte size."""

    def __init__(self, imports: _AscendImports, shape: list[int], dtype: torch.dtype, device: int, role: str):
        element_size = torch.empty((), dtype=dtype).element_size()
        row_bytes = shape[1] * shape[2] * element_size
        self.exact_bytes = shape[0] * row_bytes
        try:
            tensor, mem_handle, _ = imports.buffer_c.create_vmm_physical(shape, dtype, device)
        except RuntimeError as error:
            # The driver reports these as a bare internal-error code that says
            # nothing about the cause, so attach the size, the shape and the
            # memory situation, and point at the one experiment that separates
            # the remaining explanations.
            raise RuntimeError(
                f"MoonEP could not allocate the {role} chunk of "
                f"{self.exact_bytes / 2**20:.0f} MiB ({list(shape)}, {dtype}) from CANN VMM "
                f"on device {device}: {error}\n"
                f"Device memory at the time: {_describe_device_memory(device)}.\n"
                "If the device still has ample free memory, this is not memory pressure "
                "and PYTORCH_NPU_ALLOC_CONF will not help. The remaining explanations are "
                "a per-mapping size ceiling in the driver's VMM path, or a cap on how much "
                "VMM a process may map in total. They need opposite fixes, so measure before "
                "changing anything: sweep create_vmm_physical over increasing sizes to find "
                "the ceiling, then retry this same total in several smaller chunks. If the "
                "small chunks succeed the limit is per mapping, otherwise it is per process."
            ) from error
        allocation = tensor._vmm_allocation
        if int(allocation.size) != self.exact_bytes:
            allocation.destroy()
            raise RuntimeError(
                f"MoonEP {role} chunk of {self.exact_bytes} bytes was padded to "
                f"{int(allocation.size)} bytes by the CANN VMM allocation granularity. "
                "The compact expert-GEMM view requires the per-rank expert block to be "
                "granularity aligned; choose expert matrix dimensions such that "
                "(E/R) * out_features * in_features * itemsize is a multiple of the "
                "VMM granularity."
            )
        self.tensor = tensor
        self.mem_handle = mem_handle
        self.allocation = allocation

    @property
    def source_va(self) -> int:
        return int(self.allocation.source_va)

    def close(self) -> None:
        if self.tensor is not None:
            self.tensor = None
            self.allocation.destroy()


class _NpuSegmentedMapping:
    """One reserved virtual range backed by several separate physical chunks.

    Replaces ``buffer_c.SegmentedVAMapping``, which ``npu_patch`` removed along
    with the rest of its unused surface. Only the shape this backend needs is
    kept: whole chunks, mapped end to end from the base, no sub-chunk offsets.

    CANN VMM uses two device-id spaces and they differ under
    ``ASCEND_RT_VISIBLE_DEVICES``: allocation and mapping take the visible
    index, while ``aclrtMemSetAccess`` takes the physical card id. Passing the
    visible index to the latter fails with a bare 507899, so the physical id
    comes from ``npu_patch._device``.
    """

    def __init__(self, imports: _AscendImports, handles, sizes, device: int):
        native = imports.native
        self.device_index = int(device)
        self._access_device_id = int(imports.local_physical_id(self.device_index))
        self._native = native
        handles = tuple(int(handle) for handle in handles)
        sizes = tuple(int(size) for size in sizes)
        if not handles or len(handles) != len(sizes):
            raise ValueError("VMM handles and sizes must be non-empty and paired")
        granularity = int(native.acl_vmm_granularity(self.device_index))
        for size in sizes:
            if size <= 0 or size % granularity:
                raise ValueError(
                    f"A VMM segment of {size} bytes cannot be mapped: segment sizes must "
                    f"be positive multiples of the {granularity}-byte allocation granularity."
                )
        self.size = sum(sizes)
        self.base = int(native.acl_vmm_reserve(self.size))
        self._mapped: list[int] = []
        try:
            cursor = self.base
            for handle, size in zip(handles, sizes):
                native.acl_vmm_map(cursor, size, handle, 0)
                native.acl_vmm_set_access(cursor, size, self._access_device_id)
                self._mapped.append(cursor)
                cursor += size
        except Exception:
            self.destroy()
            raise

    def destroy(self) -> None:
        while self._mapped:
            self._native.acl_vmm_unmap(self._mapped.pop())
        if self.base:
            self._native.acl_vmm_release(self.base)
            self.base = 0


class _NpuProjectionPool(ProjectionPoolResources):
    """Prefetch slots and FP32 reduction chunks shared by one matrix shape."""

    def __init__(self, backend: "NpuMoonEPBackend", runtime: "MoonEPRuntime", out_features: int, in_features: int):
        imports = backend.imports
        device = backend.device_index
        epn = runtime.experts_per_rank
        slots = runtime.prefetch_slots
        weight_shape = [slots, out_features, in_features]
        grad_shape = [epn, out_features, in_features]

        # Hand back cached-but-unused blocks first. The torch allocator holds on
        # to freed segments, and CANN VMM cannot allocate out of them, so a run
        # that has already churned through activations can fail here while torch
        # still reports plenty of room. Through the device module, not
        # ``torch.accelerator``: on torch_npu 2.10 the NPU allocator is not
        # registered as a generic DeviceAllocator, so
        # ``torch.accelerator.empty_cache`` aborts with an INTERNAL ASSERT.
        runtime.device_module.empty_cache()

        if runtime.config.vmm_probe_enabled:
            _probe_vmm_in_place(imports, device)

        self.slot_weight = _NpuChunk(imports, weight_shape, torch.bfloat16, device, "prefetch slot")
        self.owner_grad = _NpuChunk(imports, grad_shape, torch.float32, device, "home gradient")
        self.reduce_slots = _NpuChunk(imports, weight_shape, torch.float32, device, "slot gradient")

        # Peers read these chunks directly, so they must never start out as
        # uninitialised device memory. Plain device writes are legal before the
        # symmetric heap exists: the source VA is already mapped and accessible.
        self.slot_weight.tensor.zero_()
        self.owner_grad.tensor.zero_()
        self.reduce_slots.tensor.zero_()

        # FP32 [E+B] descriptor for reduce_projection_grads. Only the home and
        # slot segments are mapped; the library reads the attached source VAs.
        self.full_grad = imports.buffer_c.create_full_projection_tensor(
            self.owner_grad.tensor, self.reduce_slots.tensor, runtime.num_experts, slots, runtime.rank, device
        )
        imports.buffer_c.attach_local_source_vas(self.full_grad, self.owner_grad.source_va, self.reduce_slots.source_va)
        imports.buffer_c.attach_local_source_vas(
            self.reduce_slots.tensor, self.owner_grad.source_va, self.reduce_slots.source_va
        )
        dist.barrier(group=runtime.group)

    @property
    def owner_grad_local(self) -> torch.Tensor:
        return self.owner_grad.tensor

    @property
    def reduce_slots_local(self) -> torch.Tensor:
        return self.reduce_slots.tensor

    def close(self) -> None:
        if self.full_grad is not None:
            # The [E+B] descriptor holds a reserved VA that only _projection
            # .destroy() unmaps; dropping the reference alone would leak it.
            self.full_grad._projection.destroy()
            self.full_grad = None
        self.reduce_slots.close()
        self.owner_grad.close()
        self.slot_weight.close()


class _NpuProjection(ProjectionResources):
    def __init__(
        self,
        backend: "NpuMoonEPBackend",
        runtime: "MoonEPRuntime",
        pool: _NpuProjectionPool,
        out_features: int,
        in_features: int,
    ):
        imports = backend.imports
        device = backend.device_index
        epn = runtime.experts_per_rank
        slots = runtime.prefetch_slots
        self.pool = pool
        self.home = _NpuChunk(imports, [epn, out_features, in_features], torch.bfloat16, device, "home weight")
        self.home_weight = self.home.tensor

        # BF16 [E+B] descriptor for prefetch_projections, carrying this rank's
        # home and slot source VAs.
        self.full_weight = imports.buffer_c.create_full_projection_tensor(
            self.home.tensor, pool.slot_weight.tensor, runtime.num_experts, slots, runtime.rank, device
        )
        imports.buffer_c.attach_local_source_vas(self.full_weight, self.home.source_va, pool.slot_weight.source_va)

        # Compute view: this rank's home experts immediately followed by its
        # prefetch slots, in one contiguous virtual range. Both segments are
        # local physical memory, so the expert GEMMs never touch the reserved
        # but unmapped rows of the [E+B] projection. Ascend has no token
        # padding, so there is no reserved tail to absorb and no third segment
        # is needed.
        self.mapping = _NpuSegmentedMapping(
            imports,
            [self.home.mem_handle, pool.slot_weight.mem_handle],
            [self.home.exact_bytes, pool.slot_weight.exact_bytes],
            device,
        )
        self.gemm_weight = imports.buffer_c.create_sym_tensor_from_ptr(
            self.mapping.base,
            [epn + slots, out_features, in_features],
            torch.bfloat16,
            torch.device(get_accelerator_name(), device),
        )

    def close(self) -> None:
        self.gemm_weight = None
        if self.full_weight is not None:
            # Reserved-VA descriptor from create_full_projection_tensor;
            # destroy() only unmaps the view, physical memory stays with the
            # home/slot chunks' own allocations.
            self.full_weight._projection.destroy()
            self.full_weight = None
        self.home_weight = None
        if self.mapping is not None:
            self.mapping.destroy()
            self.mapping = None
        self.home.close()


class NpuMoonEPBackend(MoonEPBackend):
    """MoonEP on Ascend NPUs via the ``npu_patch`` CANN SHMEM extension."""

    device_type = "npu"
    # Default off. MoonEPRuntime sets the instance from moonep_config.npu_async_comm.
    supports_async_comm = False
    requires_preconfigured_static_shape = True
    # Only this rank's home block and prefetch slots are mapped; the remaining
    # rows of the [E+B] projection are reserved virtual address space.
    expert_weight_layout = "compact"
    # NvS == S*K exactly, so there are no undefined rows past the last group.
    has_static_tail = False

    def __init__(self):
        self.imports = _AscendImports()
        # Delegates to ``torch.npu.current_device`` once ``set_accelerator_compatible``
        # has installed the proxy; every MoonEP entry point runs after that.
        self.device_index = int(torch.accelerator.current_device())
        # ``supports_async_comm`` stays False until MoonEPRuntime applies
        # ``moonep_config.npu_async_comm``. The side-stream path can hang if the
        # kernels' device-side barriers desynchronise, so it is opt-in.
        self._heap_initialized = False
        self._shmem_ip_port: str | None = None

    def validate_environment(self, runtime: "MoonEPRuntime") -> None:
        if self.imports.ShmemRuntime.is_initialized():
            raise RuntimeError(
                "The npu_patch symmetric heap is already initialized in this process; "
                "MoonEP must own it exclusively so that every expert weight is registered "
                "before initialization."
            )
        host_count = _ep_group_host_count(runtime)
        if host_count > 1:
            # Multi-node EP needs the ip/port bootstrap; the UID broadcast path
            # only builds an intra-node symmetric space.
            self._shmem_ip_port = runtime.config.shmem_ip_port or _default_shmem_ip_port()
            logger.info(
                "MoonEP EP group spans %d hosts; using the ip/port symmetric-heap bootstrap %s.",
                host_count,
                self._shmem_ip_port,
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
            # Accepted for API compatibility; npu_patch stores it and
            # never reads it, since Ascend has no equivalent of PDL.
            enable_pdl=True,
            explicitly_destroy=True,
        )

    def create_projection_pool(
        self, runtime: "MoonEPRuntime", role: str, out_features: int, in_features: int
    ) -> ProjectionPoolResources:
        granularity = int(self.imports.native.acl_vmm_granularity(self.device_index))
        # BF16 weights and FP32 gradients share the same expert block layout, so
        # the narrower dtype decides: its block is half the size, and an aligned
        # BF16 block implies an aligned FP32 one. ``create_full_projection_tensor``
        # rejects a misaligned BF16 block outright -- it used to fall back to a
        # shifted layout -- so the shape has to be caught here, with a message
        # that names the dimensions to change.
        validate_chunk_granularity(
            role, runtime.experts_per_rank, out_features, in_features, torch.bfloat16.itemsize, granularity
        )
        return _NpuProjectionPool(self, runtime, out_features, in_features)

    def create_projection(
        self, runtime: "MoonEPRuntime", pool: ProjectionPoolResources, out_features: int, in_features: int
    ) -> ProjectionResources:
        return _NpuProjection(self, runtime, pool, out_features, in_features)

    def prefetch(self, runtime: "MoonEPRuntime", projection: ProjectionResources, buffer, plan) -> None:
        buffer.prefetch_projections(plan=plan, projections=(projection.full_weight,), async_finish=False)

    def rendezvous_before_reduce(self, runtime: "MoonEPRuntime", buffer) -> None:
        """Rendezvous before any rank reads its peers' slot gradients.

        ``moonep_grad_reduce`` accumulates the peers' slot buffers before reaching
        its own barrier, and ``reduce_projection_grads`` issues no cross-rank
        barrier of its own -- ``npu_patch`` only does that for dispatch, combine
        and planning. ``shmem_stream_barrier`` swaps the host round-trip for CANN
        SHMEM's stream-ordered barrier.
        """
        if runtime.config.shmem_stream_barrier:
            stream = runtime.device_module.current_stream().npu_stream
            self.imports.native.acl_shmem_barrier_all_on_stream(int(stream))
            return
        runtime.device_module.synchronize()
        dist.barrier(group=runtime.group)

    def reduce_staged_gradient(
        self,
        runtime: "MoonEPRuntime",
        projection: ProjectionResources,
        buffer,
        plan,
    ) -> torch.Tensor:
        pool = projection.pool
        buffer.reduce_projection_grads(
            plan=plan,
            projection_grads=((pool.full_grad, pool.reduce_slots_local),),
            async_finish=False,
        )
        return pool.owner_grad_local

    def initialize_symmetric_heap(self, runtime: "MoonEPRuntime") -> None:
        """Publish every registered VMM chunk into the symmetric heap.

        Collective, once per process. Must run after all ``Buffer`` and expert
        weight allocations and before the first dispatch.
        """
        if self._heap_initialized:
            return
        self.imports.ShmemRuntime.init_with_buffer(group=runtime.group, ip_port=self._shmem_ip_port)
        self._heap_initialized = True
        dist.barrier(group=runtime.group)

    def release(self, runtime: "MoonEPRuntime") -> None:
        """Finalize the symmetric heap before any backing memory is freed.

        ``aclshmem_finalize`` needs every registered VMM handle still alive.
        A successful finalize clears the library's backing lock, which is
        what licenses the later VmmAllocation destroys; a failed one keeps
        it, so ``_heap_initialized`` must also stay set to allow a retry on
        the next ``close`` instead of tearing live VMM handles.
        """
        if not self._heap_initialized:
            return
        runtime.device_module.synchronize()
        self.imports.ShmemRuntime.finalize()
        self._heap_initialized = False

    def finish_release(self, runtime: "MoonEPRuntime") -> None:
        """Clear the descriptor registry after every VMM owner is freed.

        Requires ``npu_patch @ad7d621+``: the registry reset raises unless a
        successful collective finalize already cleared the backing lock, so
        this must run after ``release`` and after all projections, pools and
        buffers are destroyed -- exactly where ``MoonEPRuntime.close`` calls
        it. It is what lets a next heap come up in the same process.
        """
        self.imports.buffer_c.reset_symmetric_descriptors()


def _ep_group_host_count(runtime: "MoonEPRuntime") -> int:
    """How many hosts the EP group spans, preferring the launcher environment.

    Avoids a collective where it can: the first one on a subgroup brings up that
    subgroup's HCCL communicator, so a setup failure would surface inside a sanity
    check. torchrun assigns ranks contiguously, so ``rank // LOCAL_WORLD_SIZE`` is
    the node index; otherwise fall back to an allgather of a hostname hash.
    """
    ranks = dist.get_process_group_ranks(runtime.group)
    local_size = os.environ.get("LOCAL_WORLD_SIZE")
    if local_size and local_size.isdigit() and int(local_size) > 0:
        per_node = int(local_size)
        return len({rank // per_node for rank in ranks})

    logger.debug("LOCAL_WORLD_SIZE is unset; probing EP group locality with an allgather.")
    # Truncated so the hash fits an int64 with room to spare.
    fingerprint = int(hashlib.sha256(socket.gethostname().encode()).hexdigest()[:15], 16)
    device = torch.device(get_accelerator_name(), runtime.device_module.current_device())
    local = torch.tensor([fingerprint], dtype=torch.int64, device=device)
    gathered = torch.empty(len(ranks), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local, group=runtime.group)
    return len(set(gathered.tolist()))


def _default_shmem_ip_port() -> str:
    """Derive the CANN SHMEM bootstrap endpoint from the torchrun environment.

    Port offset by one so the config store does not clash with the rendezvous.
    """
    address = os.environ.get("MASTER_ADDR")
    port = os.environ.get("MASTER_PORT")
    if not address or not port:
        raise RuntimeError(
            "Multi-node MoonEP needs moonep_config.shmem_ip_port, or MASTER_ADDR and "
            "MASTER_PORT in the environment, to bootstrap the CANN SHMEM symmetric heap."
        )
    return f"tcp://{address}:{int(port) + 1}"

# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import logging
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim import AdamW

from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


def _fmt_bytes(nbytes: int) -> str:
    """Human-readable formatting for byte counts."""
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.3f} GB"
    if nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.1f} MB"
    return f"{nbytes / 1024:.1f} KB"


# pylint: disable=no-member
class SwapOptimizerOperate:
    swap_to_device_stream = None
    swap_to_host_stream = None

    param_to_cpu_states_map = {}
    param_to_device_states_map = {}

    state_keys = ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']
    swap_state_keys = set()

    def __init__(
        self,
        mem_fraction_static=0.8,
        state_keys=None,
        param_prefetch=False,
        chunk_numel=4 << 20,
        resident_states_ratio=0.0,
        prefetch_budget_ratio=0.7,
        prefetch_batches=8,
    ):
        if SwapOptimizerOperate.swap_to_device_stream is None:
            SwapOptimizerOperate.swap_to_device_stream = torch.accelerator.Stream()
            SwapOptimizerOperate.swap_to_host_stream = torch.accelerator.Stream()

        self.param_to_group_map = {}
        for group in self.param_groups:
            for p in group['params']:
                self.param_to_group_map[p] = group

        if state_keys is not None:
            self.state_keys = state_keys
        # Instance-level: a class-level set would leak keys across instances (e.g. MultiOptimizer)
        self.swap_state_keys = set()
        self.param_prefetch = param_prefetch
        self.chunk_numel = chunk_numel
        self.resident_states_ratio = resident_states_ratio
        self._prefetch_budget_ratio = prefetch_budget_ratio
        self._prefetch_batches = prefetch_batches

        self._swap_params = list(self.param_to_group_map.keys())
        # Must be decided before states offload: resident params get no host copy and keep their storage
        self._resident_params = self._split_resident_params(resident_states_ratio)
        # Swapped params first, resident last, ascending numel: greedy admission packs the budget tightly
        self._swap_order = sorted(
            (p for p in self._swap_params if p not in self._resident_params),
            key=lambda t: _local(t).numel(),
        ) + sorted(
            (p for p in self._swap_params if p in self._resident_params),
            key=lambda t: _local(t).numel(),
        )
        self._prefetch_order = [p for p in self._swap_order if p not in self._resident_params]

        self.opt_states_initialization()

        self.mem_fraction_static = mem_fraction_static
        self.swap_numel = 0
        self._states_prefetch_fired = False
        self._prefetch_logged = False
        self._prefetch_numel = 0
        # Previous step's D2H completion event: orders H2D across streams without a CPU hard sync
        self._prev_step_d2h_event = None
        # Backward prefetch H2D completion events (one per sub-batch) and param → event-index map
        self._prefetch_events = []
        self._param_prefetch_event_index = {}
        self.memory_data_initialization()

    def _split_resident_params(self, ratio):
        """Collect the smallest params (ascending numel) whose states stay on device permanently.

        Admitted while cumulative numel stays within ``ratio × total swap numel``.
        """
        if ratio <= 0:
            return set()
        total = sum(_local(p).numel() for p in self._swap_params)
        threshold = int(ratio * total)
        resident, cumulative = set(), 0
        for p in sorted(self._swap_params, key=lambda t: _local(t).numel()):
            numel = _local(p).numel()
            if cumulative + numel > threshold:
                break
            resident.add(p)
            cumulative += numel
        return resident

    def opt_states_initialization(self):
        for group in self.param_groups:
            for param in group["params"]:
                device_state_dtensor = self.state[param]
                device_state_tensor = {}
                cpu_state = {}

                amsgrad = self.param_to_group_map[param]['amsgrad']

                for key in self.state_keys:
                    if key == 'max_exp_avg_sq' and not amsgrad:
                        device_state_dtensor[key] = None
                        device_state_tensor[key] = None
                        cpu_state[key] = None
                    else:
                        self.swap_state_keys.add(key)
                        device_state_dtensor[key] = torch.zeros_like(
                            param, memory_format=torch.preserve_format, device=torch.accelerator.current_device_index()
                        )
                        device_state_tensor[key] = _local(device_state_dtensor[key])

                        if param in self._resident_params:
                            # Resident: states stay on device — no host copy, storage not released
                            cpu_state[key] = None
                        else:
                            cpu_state[key] = torch.empty_like(device_state_tensor[key], pin_memory=True, device='cpu')
                            cpu_state[key].copy_(device_state_tensor[key], non_blocking=True)

                            device_state_tensor[key].storage().resize_(0)

                self.param_to_device_states_map[param] = device_state_tensor
                self.param_to_cpu_states_map[param] = cpu_state

    def memory_data_initialization(self):
        params = list(self.param_to_device_states_map.keys())
        self.byte_param = 4 if params[0].dtype == torch.float32 else 2
        self.total_memory = torch.accelerator.get_device_properties(
            torch.accelerator.current_device_index()
        ).total_memory

        # cpu_offload: step() temporarily uploads param- and grad-sized tensors; reserve two extra slots conservatively
        self._extra_state_slots = 2 if any(_local(p).device.type == 'cpu' for p in params) else 0

    @staticmethod
    def _memory_allocated():
        # torch.accelerator.memory_allocated() asserts with torch_npu; query the NPU backend directly when available
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.npu.memory_allocated()
        return torch.accelerator.memory_allocated()

    def get_swap_numel_from_unused_memory(self):
        used_memory = self._memory_allocated()
        unused_memory = self.total_memory - used_memory

        divisor = self.byte_param * (len(self.swap_state_keys) + self._extra_state_slots)
        swap_numel = unused_memory * self.mem_fraction_static // divisor
        return swap_numel

    @staticmethod
    def _batch_copy(pairs):
        """Issue a batch of ``(dst, src)`` copies; grouped by dtype via ``_foreach_copy_``, falling back to plain ``copy_`` for a single pair."""
        if not pairs:
            return
        if len(pairs) == 1:
            pairs[0][0].copy_(pairs[0][1], non_blocking=True)
            return
        dtype_groups = {}
        for dst, src in pairs:
            dtype_groups.setdefault(dst.dtype, []).append((dst, src))
        for group in dtype_groups.values():
            torch._foreach_copy_([dst for dst, _ in group], [src for _, src in group], non_blocking=True)

    def _load_states(self, params):
        """Load a chunk's states from host back to device; returns the number of copy pairs actually enqueued."""
        state_copies = []
        param_copies = []
        for param in params:
            cpu_state = self.param_to_cpu_states_map[param]
            device_state = self.param_to_device_states_map[param]
            for key in self.state_keys:
                dev = device_state.get(key)
                src = cpu_state.get(key)
                # src is None or a 0-numel local shard (high ranks under uneven FSDP2 sharding)
                if dev is None or src is None or src.numel() == 0 or dev.storage().size() != 0:
                    continue
                dev.storage().resize_(src.numel())
                state_copies.append((dev, src))

            # cpu_offload: prefetch param into a device buffer
            if self.param_prefetch and _local(param).device.type == 'cpu':
                if param not in getattr(self, 'param_to_device_buffer', {}):
                    self.param_to_device_buffer[param] = torch.empty_like(
                        _local(param), device=torch.accelerator.current_device_index()
                    )
                param_copies.append((self.param_to_device_buffer[param], _local(param)))
        self._batch_copy(state_copies)
        self._batch_copy(param_copies)
        return len(state_copies) + len(param_copies)

    def _states_on_device(self, param):
        """Whether the param's states are loaded on device (non-zero storage; 0-numel shards are always 0)."""
        device_states = self.param_to_device_states_map.get(param, {})
        return any(s is not None and s.storage().size() != 0 for s in device_states.values())

    def _h2d_chunk(self, chunk, admitted_param_numel):
        """Load a whole chunk on the swap stream; budget unit is param numel.

        Returns ``(loaded_param_numel, chunk_event)``, the event being ``None``
        when nothing was enqueued.
        """
        if not chunk:
            return 0, None
        loaded_param_numel = 0
        available = self._max_swap_numel
        to_load = []
        with torch.accelerator.stream(self.swap_to_device_stream):
            for param in chunk:
                if param.grad is None:
                    continue
                param_numel = _local(param).numel()
                on_device = self._states_on_device(param)
                if on_device:
                    # Already on device (e.g. prefetched): skip unless the param itself needs uploading
                    need_param_upload = (
                        self.param_prefetch
                        and _local(param).device.type == 'cpu'
                        and param not in getattr(self, 'param_to_device_buffer', {})
                    )
                    if not need_param_upload:
                        continue
                elif admitted_param_numel + param_numel > available:
                    continue
                to_load.append(param)
                admitted_param_numel += param_numel
                loaded_param_numel += param_numel
            enqueued_copies = self._load_states(to_load)
            # Record only when copies were enqueued: an idle-queue fence would block on the whole prefetch tail
            event = self.swap_to_device_stream.record_event() if enqueued_copies else None
        return loaded_param_numel, event

    def prefetch_states_to_device(self):
        """Budgeted, fire-once batch-prefetch of optimizer states to device during backward.

        Called by the trainer from a backward pre-hook; frozen params are
        filtered by requires_grad. The admitted prefix is split into ordered
        sub-batches, each with its own event, so step() waits only what its
        chunk needs. A drain gate orders the first H2D after the trigger point.
        """
        if self._states_prefetch_fired:
            return
        self._states_prefetch_fired = True
        backward_done = torch.accelerator.current_stream().record_event()
        # Cross-step ordering: the swap stream waits for the previous step's D2H completion event
        if self._prev_step_d2h_event is not None:
            self.swap_to_device_stream.wait_event(self._prev_step_d2h_event)
        prefetch_budget_numel = int(self.get_swap_numel_from_unused_memory() * self._prefetch_budget_ratio)
        params_list = [p for p in self._prefetch_order if p.requires_grad]
        admitted_param_numel = 0
        to_load = []
        with torch.accelerator.stream(self.swap_to_device_stream):
            self.swap_to_device_stream.wait_event(backward_done)
            for param in params_list:
                param_numel = _local(param).numel()
                if admitted_param_numel + param_numel > prefetch_budget_numel:
                    break
                to_load.append(param)
                admitted_param_numel += param_numel
            # Ordered sub-batches, one event each: step() waits only what its chunk needs
            n_batches = min(self._prefetch_batches, len(to_load))
            if n_batches:
                per_batch = (len(to_load) + n_batches - 1) // n_batches
                for start in range(0, len(to_load), per_batch):
                    sub_batch = to_load[start : start + per_batch]
                    self._load_states(sub_batch)
                    self._prefetch_events.append(self.swap_to_device_stream.record_event())
                    for param in sub_batch:
                        self._param_prefetch_event_index[param] = len(self._prefetch_events) - 1
        # States already reserved on device must not be double-charged in step()
        self._prefetch_numel = admitted_param_numel
        if admitted_param_numel:
            if not self._prefetch_logged:
                self._prefetch_logged = True
                bytes_states = admitted_param_numel * len(self.swap_state_keys) * self.byte_param
                print_rank(
                    logger.info,
                    f"[Swap Optimizer] backward prefetch enqueued {_fmt_bytes(bytes_states)} "
                    f"of optimizer states on swap stream ({admitted_param_numel} numel)",
                )
        else:
            print_rank(
                logger.warning,
                "[Swap Optimizer] backward prefetch loaded 0 numel: free memory too "
                "low at trigger point, states will be H2D inside step()",
            )

    def _prepare_fused_adamw(self, param, group):
        """Prepare one param's tensors for the batched fused adamw kernel.

        Returns ``(entry, param_was_cpu)`` where entry is
        ``(param_local, grad_local, exp_avg_local, exp_avg_sq_local,
        max_exp_avg_sq_local)``, or ``(None, False)`` for 0-numel shards.
        When ``param_was_cpu`` is True the caller must copy param_local back
        into ``_local_tensor`` after the update.
        """
        amsgrad = group['amsgrad']
        state = self.state[param]

        # 0-numel shard: nothing to update, empty storage is invalid for the kernel
        if _local(param).numel() == 0:
            return None, False

        # OOM guard: states never loaded (empty storage) — fail loudly rather than corrupt numerics
        if not self._states_on_device(param):
            raise AssertionError("OOM: cannot fit param on NPU even after all D2H. Try increasing mem_fraction_static.")

        if len(state) == 0:
            state['exp_avg'] = torch.zeros_like(
                param, memory_format=torch.preserve_format, device=torch.accelerator.current_device_index()
            )
            state['exp_avg_sq'] = torch.zeros_like(
                param, memory_format=torch.preserve_format, device=torch.accelerator.current_device_index()
            )
        if 'max_exp_avg_sq' not in state:
            state['max_exp_avg_sq'] = (
                torch.zeros_like(
                    param, memory_format=torch.preserve_format, device=torch.accelerator.current_device_index()
                )
                if amsgrad
                else None
            )

        dev = torch.accelerator.current_device_index()
        # cpu_offload: param/grad may still live on CPU, upload them one by one
        param_was_cpu = _local(param).device.type == 'cpu'

        # Prefer the prefetched buffer to avoid a blocking H2D on the default stream
        if param_was_cpu and hasattr(self, 'param_to_device_buffer') and param in self.param_to_device_buffer:
            param_local = self.param_to_device_buffer[param]
        else:
            param_local = _local(param)
            if param_local.device.type == 'cpu':
                param_local = param_local.to(dev)

        grad_local = _local(param.grad)
        if grad_local.device.type == 'cpu':
            grad_local = grad_local.to(dev)
        exp_avg_local = _local(state['exp_avg'])
        exp_avg_sq_local = _local(state['exp_avg_sq'])

        if param_local.device.type == 'cpu':
            param_local = param_local.to(dev)
        if exp_avg_local.device.type == 'cpu':
            exp_avg_local = exp_avg_local.to(dev)
            exp_avg_sq_local = exp_avg_sq_local.to(dev)

        max_exp_avg_sq_local = None
        if amsgrad and state.get('max_exp_avg_sq') is not None:
            max_exp_avg_sq_local = _local(state['max_exp_avg_sq'])
            if max_exp_avg_sq_local.device.type == 'cpu':
                max_exp_avg_sq_local = max_exp_avg_sq_local.to(dev)

        return (
            (param_local, grad_local, exp_avg_local, exp_avg_sq_local, max_exp_avg_sq_local),
            param_was_cpu,
        )

    def _batched_fused_adamw(self, prepared):
        """Issue torch._fused_adamw_ once per (param-group, dtype) bucket.

        Bucketing by group keeps the scalar hyperparams homogeneous; by dtype
        it avoids the mixed-dtype precision pitfall. The group-level step
        tensor is read-only inside the kernel and shared across the bucket.
        """
        buckets = {}
        for param, entry, _ in prepared:
            group = self.param_to_group_map[param]
            key = (id(group), entry[0].dtype)
            if key not in buckets:
                buckets[key] = {
                    'group': group,
                    'params': [],
                    'grads': [],
                    'exp_avgs': [],
                    'exp_avg_sqs': [],
                    'maxs': [],
                }
            bucket = buckets[key]
            bucket['params'].append(entry[0])
            bucket['grads'].append(entry[1])
            bucket['exp_avgs'].append(entry[2])
            bucket['exp_avg_sqs'].append(entry[3])
            bucket['maxs'].append(entry[4])
        for bucket in buckets.values():
            group = bucket['group']
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            torch._fused_adamw_(
                bucket['params'],
                bucket['grads'],
                bucket['exp_avgs'],
                bucket['exp_avg_sqs'],
                bucket['maxs'] if amsgrad else [],
                [group['step']] * len(bucket['params']),
                lr=group['lr'],
                beta1=beta1,
                beta2=beta2,
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                amsgrad=amsgrad,
                maximize=group['maximize'],
            )


def _local(t):
    """Return the local shard of a DTensor; plain tensors pass through."""
    return t.to_local() if isinstance(t, DTensor) else t


class AdamWSwap(AdamW, SwapOptimizerOperate):
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        mem_fraction_static: Optional[float] = 0.8,
        state_keys: Optional[List[str]] = None,
        param_prefetch: bool = False,
        chunk_numel: Optional[int] = 4 << 20,
        resident_states_ratio: float = 0.0,
        prefetch_budget_ratio: float = 0.7,
        prefetch_batches: int = 8,
    ):
        """
        AdamWSwap swaps optimizer states to host memory to reduce peak device memory usage.

        At construction each param's states are copied into pinned host memory
        and their device storage is released; inside step() states are loaded
        back in ascending-numel chunks for the update, then written back to
        host asynchronously. See docs/zh/features/swap_optimizer.md for the
        scheduling details.

        Args:
            mem_fraction_static(float): Fraction of free device memory used for batched H2D, default 0.8.
            state_keys(list): State keys participating in swap, default ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'].
            param_prefetch(bool): Whether to prefetch a param device buffer (effective under cpu_offload), default False.
            chunk_numel(int): Chunk budget cap in numel, default 4M numel.
            resident_states_ratio(float): Resident ratio: smallest params (ascending numel) keep their
                states on device permanently and never swap; 0 disables, default 0.
            prefetch_budget_ratio(float): Fraction of free memory at the backward prefetch trigger
                point used for prefetch; the rest is left to later backward allocations, default 0.7.
            prefetch_batches(int): Number of ordered sub-batches the backward prefetch is split
                into, default 8.
            Other parameters: Refer to torch.optim.AdamW.

        Example:
            >>> AdamWSwap(params, mem_fraction_static=0.9)
        """
        # pylint: disable=unexpected-keyword-arg
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=False,
            maximize=maximize,
            capturable=False,
            differentiable=False,
            fused=True,
        )

        SwapOptimizerOperate.__init__(
            self,
            mem_fraction_static=mem_fraction_static,
            state_keys=state_keys,
            param_prefetch=param_prefetch,
            chunk_numel=chunk_numel,
            resident_states_ratio=resident_states_ratio,
            prefetch_budget_ratio=prefetch_budget_ratio,
            prefetch_batches=prefetch_batches,
        )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
                if group['step'].is_cpu:
                    group['step'] = group['step'].to(torch.accelerator.current_device_index())
            else:
                group['step'] = torch.tensor(1, dtype=torch.int64, device=torch.accelerator.current_device_index())

        # Cross-step ordering: the swap stream waits for the previous step's D2H completion event
        if self._prev_step_d2h_event is not None:
            self.swap_to_device_stream.wait_event(self._prev_step_d2h_event)
        self.swap_numel = self.get_swap_numel_from_unused_memory()
        self._max_swap_numel = self.swap_numel + self._prefetch_numel
        self._states_prefetch_fired = False
        admitted_param_numel = self._prefetch_numel
        self._prefetch_numel = 0
        # Take over the prefetch events so a step that never reaches them leaves no stale waits
        prefetch_events = self._prefetch_events
        prefetch_event_index = self._param_prefetch_event_index
        self._prefetch_events, self._param_prefetch_event_index = [], {}

        # Ping-pong double buffering: two adjacent chunks of states are resident at once
        budget = min(self.chunk_numel, max(self.swap_numel // 2, 1))
        chunks = self._chunk_by_numel(self._swap_order, budget)

        chunk_events = []
        if chunks:
            loaded_param_numel, event = self._h2d_chunk(chunks[0][0], admitted_param_numel)
            admitted_param_numel += loaded_param_numel
            chunk_events.append(event)
            # Fail if not even one state of the first chunk fits
            if loaded_param_numel == 0 and any(
                p.grad is not None and _local(p).numel() > 0 and not self._states_on_device(p) for p in chunks[0][0]
            ):
                raise AssertionError(
                    "OOM: cannot fit even one param's optimizer state on Device. Try increasing mem_fraction_static."
                )

        for ci, (batch, _chunk_numel) in enumerate(chunks):
            event = chunk_events[ci]
            if event is not None:
                torch.accelerator.current_stream().wait_event(event)
            # Ping-pong: issue the next chunk's H2D to hide under this chunk's adamw
            if ci + 1 < len(chunks):
                loaded_param_numel, event = self._h2d_chunk(chunks[ci + 1][0], admitted_param_numel)
                admitted_param_numel += loaded_param_numel
                chunk_events.append(event)
            prepared = []
            d2h_params = []
            needed_event = -1
            for param in batch:
                needed_event = max(needed_event, prefetch_event_index.get(param, -1))
                is_resident = param in self._resident_params
                if param.grad is None:
                    # No grad this step: send prefetched states back to host; only release budget when actually loaded
                    if not is_resident:
                        if self._states_on_device(param):
                            admitted_param_numel -= _local(param).numel()
                        if param in getattr(self, 'param_to_device_buffer', {}):
                            del self.param_to_device_buffer[param]
                        d2h_params.append(param)
                    continue
                entry, param_was_cpu = self._prepare_fused_adamw(param, self.param_to_group_map[param])
                if entry is not None:
                    prepared.append((param, entry, param_was_cpu))
                if not is_resident:
                    admitted_param_numel -= _local(param).numel()
                    d2h_params.append(param)
            # Wait the deepest prefetch sub-batch this chunk consumes; -1 waits nothing
            if needed_event >= 0:
                torch.accelerator.current_stream().wait_event(prefetch_events[needed_event])
            if prepared:
                self._batched_fused_adamw(prepared)
                updated = [(param, entry[0], was_cpu) for param, entry, was_cpu in prepared]
            else:
                updated = []

            if d2h_params:
                # Chunk-level async D2H: one stream switch and one event per chunk
                update_event = torch.accelerator.current_stream().record_event()
                with torch.accelerator.stream(self.swap_to_host_stream):
                    self.swap_to_host_stream.wait_event(update_event)
                    state_copies = []
                    dev_tensors = []
                    for param in d2h_params:
                        cpu_state = self.param_to_cpu_states_map[param]
                        device_state = self.param_to_device_states_map.get(param)
                        if device_state is None:
                            continue
                        for key in self.state_keys:
                            dev = device_state.get(key)
                            if dev is None:
                                continue
                            dev_local = _local(dev)
                            if dev_local.storage().size() == 0:
                                continue
                            state_copies.append((_local(cpu_state[key]), dev_local))
                            dev_tensors.append(dev_local)
                    self._batch_copy(state_copies)
                    for dev_local in dev_tensors:
                        dev_local.record_stream(self.swap_to_host_stream)
                        dev_local.storage().resize_(0)
                # param D2H stays blocking (pageable host memory)
                for param, param_local, param_was_cpu in updated:
                    if param_was_cpu:
                        param._local_tensor.copy_(param_local)
                        if param in getattr(self, 'param_to_device_buffer', {}):
                            del self.param_to_device_buffer[param]

        # Completion event for this step's D2H: the next step's H2D / backward prefetch wait on it across streams
        self._prev_step_d2h_event = self.swap_to_host_stream.record_event()
        return loss

    def _chunk_by_numel(self, params, budget):
        """Chunk an already numel-ordered param list by budget.

        A chunk closes before exceeding the budget, so params larger than the
        budget get a chunk of their own.
        """
        chunks, cur, cur_numel = [], [], 0
        for p in params:
            param_numel = _local(p).numel()
            if cur and cur_numel + param_numel > budget:
                chunks.append((cur, cur_numel))
                cur, cur_numel = [], 0
            cur.append(p)
            cur_numel += param_numel
            if cur_numel >= budget:
                chunks.append((cur, cur_numel))
                cur, cur_numel = [], 0
        if cur:
            chunks.append((cur, cur_numel))
        return chunks

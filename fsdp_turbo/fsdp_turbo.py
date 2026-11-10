# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import (
    expert_fully_shard_modules,
)
from fsdp_turbo.distributed.expert_parallel.utils import DispatchSplitPolicy
from fsdp_turbo.distributed.fully_shard_parallel.fully_shard_parallel import (
    fully_shard_parallel_modules,
)
from fsdp_turbo.distributed.parallel_state import init_parallel_state
from fsdp_turbo.distributed.tensor_parallel.tensor_parallel import (
    tensor_parallel_modules,
)
from fsdp_turbo.memory.chunk_batch.chunk_batch import chunk_batch_modules
from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_context_parallel import (
    ulysses_context_parallelize_modules,
)
from fsdp_turbo.distributed.context_parallel.kv_allgather.kv_allgather_context_parallel import (
    kv_allgather_context_parallelize_modules,
)
from fsdp_turbo.distributed.context_parallel.kcp.kcp_context_parallel import (
    kcp_context_parallelize_modules,
)
from fsdp_turbo.distributed.context_parallel.patch import apply_cp_loss_patches
from fsdp_turbo.memory.recompute.recompute import recompute_modules
from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, uses_hybrid_cp_zigzag_layout
from fsdp_turbo.distributed.expert_parallel.expert_parallel import (
    expert_parallelize_modules,
)
from fsdp_turbo.distributed.vocab_parallel.vocab_parallel import (
    vocab_parallelize_modules,
)
from fsdp_turbo.distributed.vocab_parallel.vocab_fully_shard_parallel import (
    vocab_fully_shard_modules,
)
from fsdp_turbo.utils.patch import apply_module_patches


class FSDPTurbo(torch.nn.Module):
    """Apply configured model parallelism, memory policies and quantization."""

    def __init__(self, config: FSDPTurboConfig, model: torch.nn.Module):
        super().__init__()
        self.config = config
        self.model = model

        self.parallel_state = init_parallel_state(self.config)
        self.apply_quantization_modules()
        self.apply_tp_modules()
        self._capture_tp_info_on_mx_linear()
        self.apply_cp_modules()
        self.apply_module_patches()
        self.apply_ep_modules()
        self.apply_vp_modules()
        self.apply_pre_fsdp_module_hooks()
        self.apply_recompute_modules()
        self.apply_swap_activation_modules()
        self.apply_chunk_batch_modules()
        self.apply_fsdp_modules()

    def apply_fsdp_modules(self):
        # Use 2D (dp, fsdp) mesh so FSDP2 treats dense params as HSDP and
        # all-reduces gradients across the dp (replicate) dim.  Using the 1D
        # fsdp mesh skips the dp all-reduce, yielding wrong gradients when the
        # fsdp domain is not filled (data_parallel_size > 1).
        self.model = fully_shard_parallel_modules(
            self.model,
            self.parallel_state.get_device_mesh("dp_fsdp"),
            self.config.distributed.fsdp_plan,
            self.config.distributed.ep_plan,
        )

    def apply_pre_fsdp_module_hooks(self):
        """Allow modules with nonstandard storage to prepare before FSDP."""
        ignored_modules = self.config.distributed.fsdp_plan.ignored_modules
        for name, module in self.model.named_modules():
            prepare = getattr(module, "pre_fsdp_hook", None)
            if not callable(prepare):
                continue
            prepare(self.parallel_state)
            if name not in ignored_modules:
                ignored_modules.append(name)

    def apply_tp_modules(self):
        if self.config.distributed.tensor_parallel_size == 1:
            return
        self.model = tensor_parallel_modules(
            self.model,
            self.parallel_state.get_tp_device_mesh(),
            self.config.distributed.tp_plan,
        )

    def apply_cp_modules(self) -> None:
        """Configure native CP, 2D KDA/MLA adapters or legacy attention/loss wrappers."""
        cp_plan = self.config.distributed.cp_plan
        if cp_plan.kda_backend == "hybrid_p2p":
            from fsdp_turbo.distributed.context_parallel.hybrid_cp import HybridCPRuntime, bind_hybrid_cp

            runtime = HybridCPRuntime(
                self.parallel_state.get_cp_group(),
                self.parallel_state.get_ulysses_group(),
                self.parallel_state.get_kda_p2p_group() if self.config.distributed.kda_p2p_parallel_size > 1 else None,
            )
            bind_hybrid_cp(self.model, runtime, required_roles=("kda",))
            if self.parallel_state.get_cp_group_size() > 1:
                apply_cp_loss_patches(
                    cp_plan.loss_function_patches,
                    runtime.cp_group,
                    zigzag=uses_hybrid_cp_zigzag_layout(cp_plan),
                )
            return
        ulysses_enabled = self.config.distributed.ulysses_parallel_size > 1
        kcp_enabled = self.config.distributed.kcp_parallel_size > 1
        # Qwen4-Exp class-forward patches own both the QSA/Ulysses layout and
        # the KV-AllGather path.  Install them for pure KV-AllGather runs as
        # well (Ulysses size 1); the size-1 Ulysses group makes its all-to-all
        # a no-op while the flattened CP group still spans KV ranks.
        cp_enabled = self.parallel_state.get_cp_group_size() > 1
        if self.parallel_state.get_kvallgather_group_size() > 1:
            self.model = kv_allgather_context_parallelize_modules(
                self.model,
                self.parallel_state.get_kvallgather_group(),
                cp_plan,
            )

        if not kcp_enabled and (ulysses_enabled or (cp_enabled and cp_plan.ulysses_function_patches)):
            self.model = ulysses_context_parallelize_modules(
                self.model,
                self.parallel_state.get_ulysses_group(),
                cp_plan,
            )

        if kcp_enabled:
            self.model = kcp_context_parallelize_modules(
                self.model,
                self.parallel_state.get_cp_group(),
                cp_plan,
                kcp_state_group=self.parallel_state.get_kcp_state_group(),
                ulysses_group=self.parallel_state.get_ulysses_group(),
            )

        # Attention wrappers compose on their own axes. A causal-LM loss must
        # instead see the complete context-parallel domain and is installed once.
        if self.parallel_state.get_cp_group_size() > 1 and cp_plan.loss_function_patches:
            apply_cp_loss_patches(cp_plan.loss_function_patches, self.parallel_state.get_cp_group())

    def apply_ep_modules(self):
        if self.config.distributed.expert_parallel_size > 1:
            plan = self.config.distributed.ep_plan
            if plan.dispatcher == "moonep":
                from fsdp_turbo.distributed.expert_parallel.moonep.plan import fill_moonep_static_shape

                fill_moonep_static_shape(
                    plan,
                    self.model,
                    batch_size=self.config.data.batch_size,
                    max_seq_length=self.config.data.max_seq_length,
                    ulysses_size=self.config.distributed.ulysses_parallel_size,
                )
            # Rides on each expert container; dispatchers read per-module settings.
            dispatch_split_policy = DispatchSplitPolicy(
                enabled=plan.reuse_dispatch_splits,
                cache_cap=plan.dispatch_split_cache_cap,
            )
            self.model = expert_parallelize_modules(
                self.model,
                self.parallel_state.get_ep_device_mesh(),
                plan,
                edp_mesh=self.parallel_state.get_device_mesh("edp"),
                dispatch_split_policy=dispatch_split_policy,
            )
            if plan.dispatcher != "moonep":
                self.model = expert_fully_shard_modules(
                    self.model,
                    self.parallel_state.get_device_mesh("edp_efsdp"),
                    plan,
                    self.config.distributed.fsdp_plan,
                    enable_lp_all_gather=(
                        self.config.quantization.quantization_plan.enable_fsdp_low_precision_all_gather
                    ),
                )

    def apply_vp_modules(self):
        """Apply vocabulary parallelism, then V-FSDP on top of the VP shard.

        VP shards ``[vocab, dim]`` embedding weights along the vocab dim and
        replaces their forward with the all-to-all dispatcher; V-FSDP then
        optionally applies ``fully_shard`` on top.  Must run BEFORE
        ``apply_fsdp_modules``: ``validate_vp_config`` has already removed the
        V-FSDP patterns from the FSDP plan and, for singleton V-FSDP, added
        them to its ignored_modules so the VP DTensor parameters stay outside
        ancestor FSDP units.
        """
        vp_plan = self.config.distributed.vp_plan
        if vp_plan is None:
            return
        self.model = vocab_parallelize_modules(
            self.model,
            self.parallel_state.get_vp_device_mesh(),
            vp_plan,
        )
        self.model = vocab_fully_shard_modules(
            self.model,
            self.parallel_state.get_device_mesh("vdp_vfsdp"),
            vp_plan,
            self.config.distributed.fsdp_plan,
        )

    def apply_module_patches(self):
        """Apply configured replacements before FSDP wraps the model.

        Each configured target must match either a module-level callable or at
        least one module in ``self.model.named_modules()``. Failing early keeps
        a misspelled or architecture-specific target from silently disabling a
        required patch.
        """
        for spec in self.config.module_patches:
            self.model, matched = apply_module_patches(self.model, spec)
            if not matched:
                raise RuntimeError(f"Configured module patch did not match the model: target={spec['target']!r}.")

    def apply_recompute_modules(self) -> None:
        """Apply configured module recomputation and MXFP8 replay contexts."""
        if not self.config.memory.recompute:
            return
        from contextlib import nullcontext
        from fsdp_turbo.quantization.core.recompute_phase import (
            mark_module_for_recompute,
            mxfp8_recompute_context,
        )

        def mxfp8_context_factory(module):
            if module is None:
                return None
            mark_module_for_recompute(module)
            return lambda: (nullcontext(), mxfp8_recompute_context(module))

        self.model = recompute_modules(
            self.model, self.config.memory.recompute_plan, context_fn_factory=mxfp8_context_factory
        )

    def apply_swap_activation_modules(self):
        if not self.config.memory.swap_activation:
            return
        from fsdp_turbo.memory.swap_activation.swap_activation import (
            swap_activation_modules,
        )

        self.model = swap_activation_modules(self.model, self.config.memory.swap_activation_plan.apply_modules)

    def _capture_tp_info_on_mx_linear(self):
        if self.config.distributed.tensor_parallel_size <= 1:
            return
        from fsdp_turbo.quantization.mx_formats.mx_linear import MXLinear
        from fsdp_turbo.quantization.qat.w8a16.w8a16_linear import W8A16Linear
        from fsdp_turbo.quantization.qat.w4a16.w4a16_linear import W4A16Linear
        from torch.distributed.tensor import Shard, Partial
        from fsdp_turbo.utils.str_match import module_name_match

        tp_mesh = self.parallel_state.get_device_mesh("tp")
        colwise = self.config.distributed.tp_plan.colwise_parallel or []
        rowwise = self.config.distributed.tp_plan.rowwise_parallel or []
        for name, mod in self.model.named_modules():
            if not isinstance(mod, (MXLinear, W8A16Linear, W4A16Linear)):
                continue
            if any(module_name_match(p, name) for p in colwise):
                mod._tp_mesh = tp_mesh
                mod._tp_output_placements = [Shard(-1)]
            elif any(module_name_match(p, name) for p in rowwise):
                mod._tp_mesh = tp_mesh
                mod._tp_output_placements = [Partial()]

    def apply_chunk_batch_modules(self):
        if not self.config.memory.chunk_batch:
            return
        self.model = chunk_batch_modules(self.model, self.config.memory.chunk_batch_plan)

    def apply_quantization_modules(self):
        """Apply quantization based on quantization_format + quantization_recipe."""
        if not self.config.quantization.quantization_plan.quant_recipe:
            return
        try:
            quantization_plan = self.config.quantization.quantization_plan
            quantization_plan._recompute_aware = (
                self.config.memory.recompute
                and quantization_plan.quant_recipe == "mxfp8"
                and quantization_plan.enable_fsdp_low_precision_all_gather
                and quantization_plan.fsdp_low_precision_all_gather_mode == "on-demand"
            )

            from fsdp_turbo.quantization.converter.model_converter import (
                build_model_converter,
            )

            model_converters = build_model_converter(self.config.quantization.quantization_plan)
            model_converters.convert(self.model)
        except Exception as e:
            raise RuntimeError("Failed to convert quantization plan") from e

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def close(self):
        runtime = getattr(self.model, "_moonep_runtime", None)
        if runtime is not None:
            runtime.close()

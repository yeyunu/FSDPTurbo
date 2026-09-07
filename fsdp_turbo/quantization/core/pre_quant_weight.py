# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Any, Optional, Tuple, Callable
import warnings

import torch

import torch.utils._pytree as pytree
from torch.distributed.tensor import DTensor

from fsdp_turbo.quantization.core.post_quant_weight import PostQuantWeight
from fsdp_turbo.quantization.cache import cached_weight_quant
from fsdp_turbo.quantization.core.recompute_phase import (
    MXFP8WeightPhase,
    get_module_fsdp_state,
    select_recompute_aware_phase,
)


_npu_dtype_cast = getattr(getattr(torch.ops, "npu", None), "_npu_dtype_cast", None)
_npu_ops = {_npu_dtype_cast.default} if _npu_dtype_cast is not None else set()

# foreach AdamW ops are benign (in-place updates / m-v state init):
# silence their per-step noise, keep warnings for other non-preserved ops.
_ops_quiet_no_preserve = {
    torch.ops.aten.zeros_like.default,
    torch.ops.aten._foreach_mul_.Scalar,
    torch.ops.aten._foreach_addcdiv_.ScalarList,
}

_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.expand.default,
    torch.ops.c10d.scatter_.default,
} | _npu_ops


class PreQuantWeight(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        quantizer: Callable,
        config: Optional[Any] = None,
        dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        tensor = kwargs.get("_tensor", tensor)

        if tensor is None:
            return torch.Tensor._make_wrapper_subclass(cls, torch.Size([0]))
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        quantizer: Callable[[torch.Tensor], Any],
        config: Optional[Any] = None,
        dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        self._tensor = tensor.contiguous()
        self.config = config
        self._dtype = dtype if dtype is not None else tensor.dtype
        self._name = name
        self._quantizer = quantizer

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return PreQuantWeight(
                args[0]._tensor,
                args[0]._quantizer,
                args[0].config,
                args[0]._dtype,
                args[0]._name,
            )
        config: Optional[Any] = None
        dtype: Optional[torch.dtype] = None
        name: Optional[str] = None
        quantizer: Callable = None

        def unwrap(t):
            nonlocal config, dtype, name, quantizer

            if config is None:
                config = t.config
                dtype = t._dtype
                name = t._name
                quantizer = t._quantizer
            return t._tensor

        args, kwargs = pytree.tree_map_only(PreQuantWeight, unwrap, (args, kwargs or {}))
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            if func not in _ops_quiet_no_preserve:
                warnings.warn(
                    f"PreQuantWeight type is not preserved for Operator {func}, "
                    f"enable_fsdp_low_precision_all_gather is disabled"
                )
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: PreQuantWeight(x, quantizer, config, dtype, name),
            out,
        )

    def __tensor_flatten__(self):
        tensors = ["_tensor"]
        metadata = {
            "config": self.config,
            "dtype": self._dtype,
            "name": self._name,
            "quantizer": self._quantizer,
        }
        return tensors, metadata

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return PreQuantWeight(
            inner_tensors["_tensor"],
            flatten_spec["quantizer"],
            flatten_spec["config"],
            flatten_spec["dtype"],
            flatten_spec["name"],
        )

    def __repr__(self):
        return f"PreQuantWeight(tensor={self._tensor}, dtype={self._dtype})"

    def fsdp_pre_all_gather(self, mesh, orig_size, contiguous_orig_stride, module, mp_policy):
        fsdp_state = get_module_fsdp_state(module)
        reshard_after_forward = fsdp_state._fsdp_param_group._reshard_after_forward

        source_tensor = self._tensor
        hp_tensor = source_tensor
        if hp_tensor.dtype == torch.float32:
            hp_tensor = hp_tensor.to(torch.bfloat16)
        if not hp_tensor.is_contiguous():
            hp_tensor = hp_tensor.contiguous()

        weight_fwd, scale_fwd, weight_bwd, scale_bwd = cached_weight_quant(
            hp_tensor, self._quantizer, cache_source=source_tensor
        )

        world_size = mesh.size() if mesh is not None else 1

        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState

        training_state = fsdp_state._fsdp_param_group._training_state
        mode = getattr(self.config, 'fsdp_low_precision_all_gather_mode', 'on-demand')
        recompute_phase = select_recompute_aware_phase(fsdp_state, self.config, training_state)

        if reshard_after_forward and mode != "all":
            is_backward_pass = (
                recompute_phase == MXFP8WeightPhase.BACKWARD
                if recompute_phase is not None
                else training_state == TrainingState.PRE_BACKWARD
            )

            if is_backward_pass:
                sharded_tensors = (weight_bwd,) if world_size == 1 else (weight_bwd, scale_bwd)
            else:
                sharded_tensors = (weight_fwd,) if world_size == 1 else (weight_fwd, scale_fwd)

            fwd_usage = not is_backward_pass
            bwd_usage = is_backward_pass
        else:
            fwd_usage = bwd_usage = True
            sharded_tensors = (
                (weight_fwd, weight_bwd) if world_size == 1 else (weight_fwd, scale_fwd, weight_bwd, scale_bwd)
            )

        metadata = (fwd_usage, bwd_usage, scale_fwd, scale_bwd)

        return sharded_tensors, metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        fwd_usage, bwd_usage, local_scale_fwd, local_scale_bwd = metadata

        weight_fwd, scale_fwd = None, None
        weight_bwd, scale_bwd = None, None

        num_expected_weights = int(fwd_usage) + int(bwd_usage)

        if len(all_gather_outputs) == num_expected_weights:
            weight_fwd = all_gather_outputs[0] if fwd_usage else None
            scale_fwd = local_scale_fwd if fwd_usage else None
            weight_bwd = all_gather_outputs[-1] if bwd_usage else None
            scale_bwd = local_scale_bwd if bwd_usage else None
        elif len(all_gather_outputs) == num_expected_weights * 2:
            weight_fwd = all_gather_outputs[0] if fwd_usage else None
            scale_fwd = all_gather_outputs[1] if fwd_usage else None
            weight_bwd = all_gather_outputs[-2] if bwd_usage else None
            scale_bwd = all_gather_outputs[-1] if bwd_usage else None
        else:
            raise ValueError(f"Unexpected gather outputs length: {len(all_gather_outputs)}")

        if out is not None:
            if isinstance(out, DTensor):
                out = out._local_tensor
            if not isinstance(out, PostQuantWeight):
                raise TypeError(f"Expected PostQuantWeight, but got {type(out).__name__}")

            # FSDP reuses all-gather output storages. Clear the inactive pair
            # so a stale alias cannot consume a buffer from the wrong phase.
            out._weight_fwd = weight_fwd if fwd_usage else None
            out._scale_fwd = scale_fwd if fwd_usage else None
            out._weight_bwd = weight_bwd if bwd_usage else None
            out._scale_bwd = scale_bwd if bwd_usage else None
        else:
            out = PostQuantWeight(
                weight_fwd,
                scale_fwd,
                weight_bwd,
                scale_bwd,
                param_dtype,
            )
        return out, all_gather_outputs


torch.serialization.add_safe_globals([PreQuantWeight])

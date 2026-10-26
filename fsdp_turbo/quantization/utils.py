# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import re
from typing import Callable, Dict, Type
import torch
from torch import nn

from fsdp_turbo.ops.moe import grouped_matmul
from fsdp_turbo.quantization.mxfp8_config import QuantBaseConfig
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match

logger = logging.getLogger(__name__)

# Per-shard LP-AG quant breaks the mx scale grid (wants ceil(K/64)) for out not aligned to 64*ws -> bf16
_SMALL_LINEAR_DIM = 64


_QUANTIZE_CONFIG_HANDLER: Dict[
    Type[QuantBaseConfig],
    Callable[[torch.nn.Module, QuantBaseConfig], torch.nn.Module],
] = {}


def register_quantize_module_handler(config_type):
    @functools.wraps(config_type)
    def decorator(func):
        _QUANTIZE_CONFIG_HANDLER[config_type] = func
        return func

    return decorator


class TensorWithTranspose:
    def __init__(
        self,
        fp8_dtype: torch.dtype,
        data: torch.Tensor,
        scale: torch.Tensor,
        data_t: torch.Tensor,
        scale_t: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ):
        self.fp8_dtype = fp8_dtype
        self.data = data
        self.scale = scale
        self.data_t = data_t
        self.scale_t = scale_t
        self.ori_dtype = dtype

    def get_by_trans(self, transpose=False):
        if transpose:
            return self.data_t, self.scale_t
        return self.data, self.scale


def _apply_modules_match(fqn: str, config, apply_attr: str, ignored_attr: str = "") -> bool:
    """Return True if *fqn* matches any pattern in ``config.<apply_attr>``
    and does not match any in ``config.<ignored_attr>``.
    """
    if ignored_attr:
        for pattern in getattr(config, ignored_attr, []):
            if module_name_match(pattern, fqn):
                return False
    m = re.match(r"(.*?(?:layers|blocks).\d+)", fqn)
    if m is None:
        return False
    prefix = m.group(1)
    return any(module_name_match(pattern, prefix) for pattern in getattr(config, apply_attr, []))


def quant_grouped_gemm(
    x,
    weight,
    split_list,
    config,
    grad_enabled=True,
    weight_owner=None,
    weight_attr: str | None = None,
):
    """Quantization-aware grouped matmul, dispatched by config type.

    split_list: per-expert token counts. W8A16Config -> fake quant +
    grouped_matmul; other configs -> mx_quant_group_gemm.
    """
    from fsdp_turbo.quantization.qat.w8a16.w8a16_config import W8A16Config
    from fsdp_turbo.quantization.qat.w8a16.w8a16_gmm import reshape_for_gmm
    from fsdp_turbo.quantization.qat.w8a16.w8a16_linear import W8A16FakeQuantization
    from fsdp_turbo.quantization.mx_formats.mx_gmm import mx_quant_group_gemm as mx_gemm

    if isinstance(config, W8A16Config):
        num_experts = split_list.size(0) if isinstance(split_list, torch.Tensor) else len(split_list)
        weight_q = W8A16FakeQuantization.apply(weight)
        weight_3d = reshape_for_gmm(weight_q, num_experts)
        return grouped_matmul(x, split_list, weight_3d)

    group_list = torch.cumsum(split_list, dim=0)
    return mx_gemm(
        x,
        weight,
        group_list,
        config,
        grad_enabled=grad_enabled,
        weight_owner=weight_owner,
        weight_attr=weight_attr,
    )


def resolve_expert_weights(module):
    """Resolve an MoE module's expert weights to local tensors.

    Unwraps nn.Module weights and DTensors (expert-parallel distribution);
    shared by the MX and W8A16 EP compute functions.
    """
    gate_up = module.gate_up_proj
    down = module.down_proj
    if isinstance(gate_up, torch.nn.Module):
        gate_up = gate_up.weight
    if isinstance(down, torch.nn.Module):
        down = down.weight
    gate_up = gate_up.to_local() if hasattr(gate_up, 'to_local') else gate_up
    down = down.to_local() if hasattr(down, 'to_local') else down
    return gate_up, down


def module_filter_fn(mod: nn.Module, fqn: str, config: QuantBaseConfig) -> bool:
    if not isinstance(mod, nn.Linear):
        return False
    if not _apply_modules_match(fqn, config, 'mxfp8_apply_modules', 'mxfp8_ignored_modules'):
        return False
    if mod.out_features < _SMALL_LINEAR_DIM or mod.in_features < _SMALL_LINEAR_DIM:
        print_rank(
            logger.warning,
            f"[MXFP8] skip quantizing {fqn}: shape=({mod.in_features}, {mod.out_features}) "
            f"has a dim < {_SMALL_LINEAR_DIM}, keeping bf16",
        )
        return False
    return True


def w8a16_module_filter_fn(mod: nn.Module, fqn: str, config: QuantBaseConfig) -> bool:
    return isinstance(mod, nn.Linear) and _apply_modules_match(
        fqn, config, 'w8a16_apply_modules', 'w8a16_ignored_modules'
    )


def w4a16_module_filter_fn(mod: nn.Module, fqn: str, config: QuantBaseConfig) -> bool:
    return isinstance(mod, nn.Linear) and _apply_modules_match(
        fqn, config, 'w4a16_apply_modules', 'w4a16_ignored_modules'
    )


def moe_filter_fn(mod: nn.Module, fqn: str, config: QuantBaseConfig) -> bool:
    return "experts" in fqn.lower() and _apply_modules_match(
        fqn, config, 'mxfp8_apply_modules', 'mxfp8_ignored_modules'
    )


def w8a16_moe_filter_fn(mod: nn.Module, fqn: str, config: QuantBaseConfig) -> bool:
    return "experts" in fqn.lower() and _apply_modules_match(
        fqn, config, 'w8a16_apply_modules', 'w8a16_ignored_modules'
    )


def w4a16_moe_filter_fn(mod: nn.Module, fqn: str, config: QuantBaseConfig) -> bool:
    return "experts" in fqn.lower() and _apply_modules_match(
        fqn, config, 'w4a16_apply_modules', 'w4a16_ignored_modules'
    )


def view_as_n_dim(input_tensor, dim=2):
    if dim < 2:
        raise AssertionError("dim should be greater than or equal to 2")
    if len(input_tensor.shape) != dim:
        return input_tensor.view(-1, *input_tensor.shape[-dim + 1 :])
    return input_tensor

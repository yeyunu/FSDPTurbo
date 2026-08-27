# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, NamedTuple

import torch

from fsdp_turbo.fsdp_turbo_config import QuantizeConfig


def _load_npu_dtype(getter_name):
    try:
        from fsdp_turbo.ops.npu.mx_quant import get_float4_dtype, get_hifloat8_dtype
    except ImportError:
        return None
    return {"hifloat8": get_hifloat8_dtype, "float4": get_float4_dtype}[getter_name]()


_HIFLOAT8 = _load_npu_dtype("hifloat8")
_FLOAT4_E2M1 = _load_npu_dtype("float4")


def float4_dtype():
    """NPU float4 e2m1 dtype, or None when torch_npu is unavailable."""
    return _FLOAT4_E2M1


class FP8Format:
    def __init__(self, range_max: float, ebits: int, mbits: int, dtype: Optional[torch.dtype]):
        self.max = range_max
        self.ebits = ebits
        self.mbits = mbits
        self.dtype = dtype

    @property
    def quant_type(self):
        if self.dtype is None:
            if _HIFLOAT8 is None:
                raise ImportError(
                    "torch_npu.hifloat8 is not available. HIF8 format requires Ascend NPU with torch_npu installed."
                )
            return _HIFLOAT8
        return self.dtype


class FormatEnum(Enum):
    E4M3 = FP8Format(448, 4, 3, torch.float8_e4m3fn)
    E5M2 = FP8Format(57344, 5, 2, torch.float8_e5m2)
    HIF8 = FP8Format(57344, 5, 2, None)
    E2M1 = FP8Format(6, 2, 1, _FLOAT4_E2M1)


class _FormatConfig(NamedTuple):
    inputs: FormatEnum = FormatEnum.E4M3
    weight: FormatEnum = FormatEnum.E4M3
    grads: FormatEnum = FormatEnum.E4M3


class Format(Enum):
    E4M3 = _FormatConfig()
    HYBRID = _FormatConfig(grads=FormatEnum.E5M2)
    HIF8 = _FormatConfig(inputs=FormatEnum.HIF8, weight=FormatEnum.HIF8, grads=FormatEnum.HIF8)
    W4A8 = _FormatConfig(weight=FormatEnum.E2M1)

    @classmethod
    def from_config_fp8(cls, key: str):
        return getattr(cls, key.upper(), None)


@dataclass
class QuantBaseConfig:
    quant_format: str = "E4M3"
    block_size: int = 32

    def get_key_dtype(self, key):
        key_format = Format.from_config_fp8(self.quant_format)
        config = key_format.value
        if key == 'inputs':
            return config.inputs.value.quant_type
        elif key == 'weight':
            return config.weight.value.quant_type
        else:
            return config.grads.value.quant_type


@dataclass
class MXFP8LinearConfig(QuantBaseConfig):
    enable_fsdp_low_precision_all_gather: bool = True
    fsdp_low_precision_all_gather_mode: str = "on-demand"
    fsdp_world_size: int = 1
    mxfp8_ignored_modules: list[str] = field(default_factory=list)
    mxfp8_apply_modules: list[str] = field(default_factory=list)
    converter: str = "quantize.linear.mx"
    _recompute_aware: bool = field(default=False, init=False, repr=False)


def get_mxfp8linear_config(quant_config: QuantizeConfig):
    config = MXFP8LinearConfig(
        quant_format=quant_config.quant_format,
        block_size=quant_config.block_size,
        mxfp8_ignored_modules=quant_config.quant_ignored_modules or [],
        mxfp8_apply_modules=quant_config.quant_apply_modules or [],
        enable_fsdp_low_precision_all_gather=quant_config.enable_fsdp_low_precision_all_gather,
        fsdp_low_precision_all_gather_mode=quant_config.fsdp_low_precision_all_gather_mode,
        fsdp_world_size=quant_config.fsdp_world_size,
    )
    config._recompute_aware = quant_config._recompute_aware
    return config

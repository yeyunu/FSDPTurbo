# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, NamedTuple

import torch

from fsdp_turbo.fsdp_turbo_config import QuantizeConfig


class FP4Format:
    DEFAULT_QUANT_TYPE = torch.float4_e2m1fn_x2

    def __init__(
        self,
        range_max: float,
        ebits: int,
        mbits: int,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self._max = range_max
        self._ebits = ebits
        self._mbits = mbits
        self._dtype = dtype

    @property
    def max(self) -> float:
        return self._max

    @property
    def ebits(self) -> int:
        return self._ebits

    @property
    def mbits(self) -> int:
        return self._mbits

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return self._dtype

    @property
    def quant_type(self) -> torch.dtype:
        return self._dtype if self._dtype is not None else self.DEFAULT_QUANT_TYPE


class FormatEnum(Enum):
    E2M1 = FP4Format(6, 2, 1, torch.float4_e2m1fn_x2)


class _FormatConfig(NamedTuple):
    inputs: FormatEnum = FormatEnum.E2M1
    weight: FormatEnum = FormatEnum.E2M1
    grads: FormatEnum = FormatEnum.E2M1


class Format(Enum):
    E2M1 = _FormatConfig()

    @classmethod
    def from_config_fp4(cls, key: str):
        return getattr(cls, key.upper(), None)


@dataclass
class QuantBaseConfig:
    quant_format: str = "E2M1"
    block_size: int = 32

    def get_key_dtype(self, key):
        key_format = Format.from_config_fp4(self.quant_format)
        config = key_format.value
        if key == 'inputs':
            return config.inputs.value.quant_type
        elif key == 'weights':
            return config.weight.value.quant_type
        else:
            return config.grads.value.quant_type


@dataclass
class MXFP4LinearConfig(QuantBaseConfig):
    enable_fsdp_low_precision_all_gather: bool = True
    fsdp_low_precision_all_gather_mode: str = "on-demand"
    mxfp4_ignored_modules: list[str] = field(default_factory=list)
    mxfp4_apply_modules: list[str] = field(default_factory=list)
    converter: str = "quantize.linear.mx"


def get_mxfp4linear_config(quant_config: QuantizeConfig):
    return MXFP4LinearConfig(
        quant_format=quant_config.quant_format,
        block_size=quant_config.block_size,
        mxfp4_ignored_modules=quant_config.quant_ignored_modules or [],
        mxfp4_apply_modules=quant_config.quant_apply_modules or [],
        enable_fsdp_low_precision_all_gather=quant_config.enable_fsdp_low_precision_all_gather,
        fsdp_low_precision_all_gather_mode=quant_config.fsdp_low_precision_all_gather_mode,
    )

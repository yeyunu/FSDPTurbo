# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from dataclasses import dataclass, field
from typing import List

import torch

from fsdp_turbo.quantization.mxfp8_config import QuantBaseConfig
from fsdp_turbo.fsdp_turbo_config import QuantizeConfig


@dataclass
class W8A16Config(QuantBaseConfig):
    w8a16_apply_modules: List[str] = field(default_factory=list)
    w8a16_ignored_modules: List[str] = field(default_factory=list)

    def get_key_dtype(self, key):
        return torch.float8_e4m3fn


def get_w8a16_config(quant_config: QuantizeConfig) -> W8A16Config:
    return W8A16Config(
        w8a16_apply_modules=quant_config.quant_apply_modules or [],
        w8a16_ignored_modules=quant_config.quant_ignored_modules or [],
    )

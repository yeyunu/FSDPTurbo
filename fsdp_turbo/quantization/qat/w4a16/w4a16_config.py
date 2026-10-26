# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from dataclasses import dataclass, field
from typing import List

import torch

from fsdp_turbo.quantization.mxfp4_config import QuantBaseConfig
from fsdp_turbo.fsdp_turbo_config import QuantizeConfig


@dataclass
class W4A16Config(QuantBaseConfig):
    w4a16_apply_modules: List[str] = field(default_factory=list)
    w4a16_ignored_modules: List[str] = field(default_factory=list)

    def get_key_dtype(self, key):
        return torch.float4_e2m1fn_x2


def get_w4a16_config(quant_config: QuantizeConfig) -> W4A16Config:
    return W4A16Config(
        w4a16_apply_modules=quant_config.quant_apply_modules or [],
        w4a16_ignored_modules=quant_config.quant_ignored_modules or [],
    )

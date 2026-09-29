# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn

from fsdp_turbo.quantization.mxfp8_config import get_mxfp8linear_config
from fsdp_turbo.quantization.converter.model_converter import register_model_converter
from fsdp_turbo.quantization.utils import moe_filter_fn
from fsdp_turbo.fsdp_turbo_config import QuantizeConfig


class MXMoeConverter:
    """Converts MoE expert modules to MXFP8GMM for FP8 quantized grouped matmul.

    Replaces modules matching moe_filter_fn (containing "experts" in FQN)
    with MXFP8GMM instances that use FP8 quantized grouped matrix multiplication.
    """

    def __init__(self, config: QuantizeConfig):
        self.config = get_mxfp8linear_config(config)

    def convert(self, model: nn.Module):
        """Converts MoE expert modules in the model to MXFP8GMM.

        This will mutate the model inplace.
        """
        from fsdp_turbo.quantization.quantize import quantize_
        from fsdp_turbo.quantization.mx_formats.mx_gmm import _mx_gmm_transform

        quantize_(
            model,
            config=self.config,
            filter_fn=moe_filter_fn,
            device=next((p.device for p in model.parameters()), None),
            handler=_mx_gmm_transform,
        )


register_model_converter(MXMoeConverter, "quantize.moe.mx")

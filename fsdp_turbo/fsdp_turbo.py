# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules
from fsdp_turbo.distributed.fully_shard_parallel.fully_shard_parallel import \
    fully_shard_parallel_modules
from fsdp_turbo.distributed.parallel_state import init_parallel_state
from fsdp_turbo.distributed.tensor_parallel.tensor_parallel import tensor_parallel_modules
from fsdp_turbo.memory.recompute.recompute import recompute_modules
from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig
from fsdp_turbo.distributed.expert_parallel.expert_parallel import expert_parallelize_modules


class FSDPTurbo(torch.nn.Module):
    def __init__(self, config: FSDPTurboConfig, model: torch.nn.Module):
        super(FSDPTurbo, self).__init__()
        self.config = config
        self.model = model

        self.parallel_state = init_parallel_state(self.config)
        self.apply_tp_modules()
        self.apply_ep_modules()
        self.apply_recompute_modules()
        self.apply_quantization_modules()
        self.apply_fsdp_modules()

    def apply_fsdp_modules(self):
        self.model = fully_shard_parallel_modules(self.model, self.parallel_state.get_fsdp_device_mesh(), self.config.distributed.fsdp_plan)

    def apply_tp_modules(self):
        if self.config.distributed.tensor_parallel_size == 1:
            return
        self.model = tensor_parallel_modules(self.model, self.parallel_state.get_tp_device_mesh(), self.config.distributed.tp_plan)

    def apply_ep_modules(self):
        if self.config.distributed.expert_parallel_size > 1:
            self.model = expert_parallelize_modules(self.model, self.parallel_state.get_ep_device_mesh(), self.config.distributed.ep_plan)
            self.model = expert_fully_shard_modules(
                self.model,
                self.parallel_state.get_efsdp_device_mesh(),
                self.config.distributed.ep_plan,
                self.config.distributed.fsdp_plan,
            )

    def apply_recompute_modules(self):
        if not self.config.memory.recompute:
            return
        self.model = recompute_modules(self.model, self.config.memory.recompute_plan)

    def apply_quantization_modules(self):
        """Apply quantization based on quantization_format + quantization_recipe."""
        if not self.config.quantization.quantization_plan.quant_recipe:
            return
        try:
            from fsdp_turbo.quantization.converter.model_converter import build_model_converter

            model_converters = build_model_converter(self.config.quantization.quantization_plan)
            model_converters.convert(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to convert quantization plan") from e

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

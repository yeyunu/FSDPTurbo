# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Select the best available KDA implementation for NPU devices."""

# Backend imports are intentionally conditional to keep optional dependencies lazy.
# pylint: disable=ungrouped-imports

import logging

import torch

from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.utils.device import IS_FLA_NPU_AVAILABLE, IS_TRITON_ASCEND_KERNELS_AVAILABLE
from fsdp_turbo.utils.log import print_rank


logger = logging.getLogger(__name__)


# The AscendC implementation still uses Triton kernels for its backward and
# auxiliary operations, so both packages are required for this path.
# NOTE: The current AscendC entry is training-only: it rejects a non-None
# initial_state and does not return a final recurrent state. Because backend
# selection happens at import time, cached generation will not automatically
# fall back to Triton when fla_npu is installed. Runtime routing must be added
# before enabling the AscendC path for generation/cache workloads.
if IS_FLA_NPU_AVAILABLE and IS_TRITON_ASCEND_KERNELS_AVAILABLE:
    from fsdp_turbo.ops.npu.chunk_kda_ascendc import chunk_kda_ascendc as _chunk_kda_impl

    _CHUNK_KDA_BACKEND = 'ascendc'
elif IS_TRITON_ASCEND_KERNELS_AVAILABLE:
    from triton_ascend_kernels.attention.fla.kda.chunk import chunk_kda as _chunk_kda_impl

    _CHUNK_KDA_BACKEND = 'triton'
else:
    from fsdp_turbo.ops.cpu.chunk_kda import chunk_kda_cpu as _chunk_kda_impl

    _CHUNK_KDA_BACKEND = 'eager'

    print_rank(
        logger.warning,
        "AscendC and Triton Ascend KDA implementations are unavailable; falling back to the eager implementation.",
    )

# This module executes at import time, before BaseTrainer._init_distributed
# configures logging, so a logger record would be dropped; print straight to
# stdout via print_rank (rank-0 only) so the marker shows in every launch mode.
print_rank(print, f"[chunk_kda] backend={_CHUNK_KDA_BACKEND}")


@register_op('chunk_kda', 'npu')
def chunk_kda_npu(q, k, v, g, beta, *args, **kwargs):
    """Run KDA with the selected NPU implementation."""
    if _CHUNK_KDA_BACKEND == 'triton' and kwargs.get('use_beta_sigmoid_in_kernel', False):
        beta = torch.sigmoid(beta)
        kwargs['use_beta_sigmoid_in_kernel'] = False
    return _chunk_kda_impl(q, k, v, g, beta, *args, **kwargs)


__all__ = ['chunk_kda_npu']

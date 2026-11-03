# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CUDA causal flash attention via SDPA.

On CUDA, ``scaled_dot_product_attention`` dispatches to the fused
flash/cuDNN kernels, so the SDPA reference implementation registered for
CPU is reused as the CUDA implementation (including the bottom-right causal
alignment for a query block against a longer K/V prefix). Registering it
here also keeps GPU runs from logging the registry's CPU-fallback warning.
"""

from fsdp_turbo.ops.cpu.flash_attn import causal_flash_attention_cpu
from fsdp_turbo.ops.registry import register_op


@register_op("causal_flash_attention", "cuda")
def causal_flash_attention_cuda(query, key, value, *, scaling=None, dropout=0.0, is_causal=True):
    """SDPA path for CUDA; see the CPU reference for the semantics."""
    return causal_flash_attention_cpu(query, key, value, scaling=scaling, dropout=dropout, is_causal=is_causal)

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import logging

# Register the lazy adapter before imports of optional native libraries.
from fsdp_turbo.ops.npu.tilekernels_mhc import tilekernels_mhc_post_npu
from fsdp_turbo.ops.npu.cce_loss import cce_loss_npu
from fsdp_turbo.ops.npu.attn_res import attn_res_npu
from fsdp_turbo.ops.npu.attention_projection import attention_output_projection_npu
from fsdp_turbo.ops.npu.clipped_swiglu import clipped_swiglu_npu
from fsdp_turbo.utils.log import log_warning_once

from fsdp_turbo.ops.npu.engram import engram_backward_npu, engram_forward_npu, engram_wkv_npu
from fsdp_turbo.ops.npu.grouped_matmul import grouped_matmul_npu, w4a16_grouped_matmul_npu
from fsdp_turbo.ops.npu.grouped_matmul_mc2 import all2all_grouped_matmul_npu, grouped_matmul_all2all_npu
from fsdp_turbo.ops.npu.moonep_grouped_matmul import (
    moonep_grouped_matmul_npu,
    moonep_grouped_matmul_dgrad_npu,
    moonep_grouped_matmul_wgrad_npu,
)
from fsdp_turbo.ops.npu.mx_grouped_matmul import mx_grouped_matmul_npu
from fsdp_turbo.ops.npu.mx_quant_matmul import mx_quant_matmul_npu, w4a16_quant_matmul_npu
from fsdp_turbo.ops.npu.mx_quant import (
    dynamic_mx_quant_npu,
    dynamic_mx_quant_with_dual_axis_npu,
    grouped_dynamic_mx_quant_npu,
    swiglu_npu,
    mx_to_block_mx_quant_npu,
    get_float4_dtype,
    get_hifloat8_dtype,
    get_e8m0_dtype,
)
from fsdp_turbo.ops.npu.permute import permute_npu, unpermute_npu
from fsdp_turbo.ops.npu.rms_norm import rms_norm_npu
from fsdp_turbo.ops.npu.rope import inplace_rope_npu, partial_rope_npu, rope_npu
from fsdp_turbo.ops.npu.partial_rotary import cann_partial_rotary
from fsdp_turbo.ops.npu.fused_attention import npu_fusion_attention_npu
from fsdp_turbo.ops.npu.qsa import qsa_compress_block_keys_npu, qsa_selected_token_mask_npu

from fsdp_turbo.ops.npu.situ_glu import situ_glu_npu
from fsdp_turbo.ops.npu.sparse_attention import (
    indexed_sparse_attention_npu,
    npu_sparse_flash_mla,
)
from fsdp_turbo.ops.npu.compressor import fused_compressor
from fsdp_turbo.ops.npu.lightning_indexer import fused_lightning_indexer
from fsdp_turbo.ops.npu.mhc import fused_mhc_post, fused_mhc_pre_sinkhorn_premix
from fsdp_turbo.ops.npu.vision_flash_attention import fused_vision_flash_attention
from fsdp_turbo.ops.npu.hf32_linear import hf32_linear_npu
from fsdp_turbo.ops.npu.flash_attn import flash_attention_npu
from fsdp_turbo.ops.npu.rotary_mul import rotary_mul_npu
from fsdp_turbo.ops.npu.sparse_indexer_loss import sparse_indexer_gradient_npu

_logger = logging.getLogger(__name__)

# Linear-attention kernels need optional packages (fla, fla_npu, triton_ascend_kernels). Their absence must not
# unregister the core MoE / norm / attention ops below: a failed package import is swallowed in
# fsdp_turbo.ops, after which every dispatch silently falls back to the (slow, dtype-different) CPU implementation.
try:
    from fsdp_turbo.ops.npu.causal_conv1d import causal_conv1d_npu, causal_conv1d_update_npu
except ImportError as _error:
    causal_conv1d_npu = causal_conv1d_update_npu = None
    log_warning_once(_logger, f"NPU causal convolution is unavailable: {_error}")
try:
    from fsdp_turbo.ops.npu.gdn import chunk_gdn_npu, fused_recurrent_gdn_npu
except ImportError as _error:
    chunk_gdn_npu = fused_recurrent_gdn_npu = None
    log_warning_once(_logger, f"NPU GDN is unavailable: {_error}")
try:
    from fsdp_turbo.ops.npu.chunk_kda import chunk_kda_npu
except ImportError as _error:
    chunk_kda_npu = None
    log_warning_once(_logger, f"NPU KDA is unavailable: {_error}")
try:
    from fsdp_turbo.ops.npu.sinkhorn import hc_split_sinkhorn_npu
except ImportError:  # triton is optional; the package must still load without it
    hc_split_sinkhorn_npu = None
try:
    from fsdp_turbo.ops.npu.lightning_indexer_score import lightning_indexer_score_npu
except ImportError:  # triton is optional; the package must still load without it
    lightning_indexer_score_npu = None
try:
    from fsdp_turbo.ops.npu.sparse_flash_attention import sparse_flash_attention_npu
except ImportError:  # depends on the torch_npu wheel exposing the SFA op
    sparse_flash_attention_npu = None

__all__ = [
    "sparse_indexer_gradient_npu",
    'attention_output_projection_npu',
    'cce_loss_npu',
    'attn_res_npu',
    'clipped_swiglu_npu',
    'causal_conv1d_npu',
    'causal_conv1d_update_npu',
    'chunk_gdn_npu',
    'fused_recurrent_gdn_npu',
    'chunk_kda_npu',
    'engram_forward_npu',
    'engram_backward_npu',
    'engram_wkv_npu',
    'grouped_matmul_npu',
    'w4a16_grouped_matmul_npu',
    'all2all_grouped_matmul_npu',
    'grouped_matmul_all2all_npu',
    'moonep_grouped_matmul_npu',
    'moonep_grouped_matmul_dgrad_npu',
    'moonep_grouped_matmul_wgrad_npu',
    'mx_grouped_matmul_npu',
    'mx_quant_matmul_npu',
    'w4a16_quant_matmul_npu',
    'dynamic_mx_quant_npu',
    'dynamic_mx_quant_with_dual_axis_npu',
    'grouped_dynamic_mx_quant_npu',
    'swiglu_npu',
    'permute_npu',
    'unpermute_npu',
    'rms_norm_npu',
    'npu_fusion_attention_npu',
    'inplace_rope_npu',
    'rope_npu',
    'partial_rope_npu',
    'cann_partial_rotary',
    'qsa_selected_token_mask_npu',
    'qsa_compress_block_keys_npu',
    'hc_split_sinkhorn_npu',
    'lightning_indexer_score_npu',
    'sparse_flash_attention_npu',
    'mx_to_block_mx_quant_npu',
    'get_float4_dtype',
    'get_hifloat8_dtype',
    'get_e8m0_dtype',
    'situ_glu_npu',
    'indexed_sparse_attention_npu',
    'npu_sparse_flash_mla',
    'fused_compressor',
    'fused_lightning_indexer',
    'fused_mhc_post',
    'fused_mhc_pre_sinkhorn_premix',
    'fused_vision_flash_attention',
    'hf32_linear_npu',
    'flash_attention_npu',
    'rotary_mul_npu',
    'tilekernels_mhc_post_npu',
]

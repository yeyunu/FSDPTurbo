# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from fsdp_turbo.ops.cpu.cce_loss import cce_loss_cpu
from fsdp_turbo.ops.cpu.payload_quant import (
    dequant_fp8_blockwise_cpu,
    dequant_int8_per_row_cpu,
    dynamic_block_quant_cpu,
    dynamic_quant_cpu,
    fp8_payload_quant_supported_cpu,
)
from fsdp_turbo.ops.cpu.attn_res import attn_res_cpu
from fsdp_turbo.ops.cpu.attention_projection import attention_output_projection_cpu
from fsdp_turbo.ops.cpu.clipped_swiglu import clipped_swiglu_cpu
from fsdp_turbo.ops.cpu.causal_conv1d import causal_conv1d_cpu, causal_conv1d_update_cpu
from fsdp_turbo.ops.cpu.gdn import chunk_gdn_cpu, recurrent_gdn_cpu
from fsdp_turbo.ops.cpu.chunk_kda import chunk_kda_cpu
from fsdp_turbo.ops.cpu.engram import engram_backward_cpu, engram_forward_cpu, engram_wkv_cpu
from fsdp_turbo.ops.cpu.fused_attention import npu_fusion_attention_cpu
from fsdp_turbo.ops.cpu.glm_indexer import lightning_indexer_cpu, indexer_kl_loss_cpu
from fsdp_turbo.ops.cpu.selected_attention import selected_attention_cpu
from fsdp_turbo.ops.cpu.grouped_matmul import grouped_matmul_cpu, w4a16_grouped_matmul_cpu
from fsdp_turbo.ops.cpu.grouped_matmul_mc2 import all2all_grouped_matmul_cpu, grouped_matmul_all2all_cpu
from fsdp_turbo.ops.cpu.lightning_indexer import fused_lightning_indexer_cpu
from fsdp_turbo.ops.cpu.lightning_indexer_score import lightning_indexer_score_torch
from fsdp_turbo.ops.cpu.moonep_grouped_matmul import (
    moonep_grouped_matmul_cpu,
    moonep_grouped_matmul_dgrad_cpu,
    moonep_grouped_matmul_wgrad_cpu,
)
from fsdp_turbo.ops.cpu.mx_quant_matmul import mx_quant_matmul_cpu, w4a16_quant_matmul_cpu
from fsdp_turbo.ops.cpu.mx_quant import (
    dynamic_mx_quant_cpu,
    dynamic_mx_quant_with_dual_axis_cpu,
    grouped_dynamic_mx_quant_cpu,
    swiglu_cpu,
    mx_to_block_mx_quant_cpu,
)
from fsdp_turbo.ops.cpu.permute import permute_cpu, unpermute_cpu
from fsdp_turbo.ops.cpu.qsa import qsa_compress_block_keys_cpu, qsa_reference_mask, qsa_selected_token_mask_cpu
from fsdp_turbo.ops.cpu.rms_norm import rms_norm_cpu
from fsdp_turbo.ops.cpu.rope import rope_torch
from fsdp_turbo.ops.cpu.sinkhorn import hc_split_sinkhorn_cpu
from fsdp_turbo.ops.cpu.sparse_attention import indexed_sparse_attention_torch, sparse_flash_attention_torch
from fsdp_turbo.ops.cpu.situ_glu import situ_glu_cpu
from fsdp_turbo.ops.cpu.flash_attn import flash_attention_cpu
from fsdp_turbo.ops.cpu.rotary_mul import rotary_mul_cpu
from fsdp_turbo.ops.cpu.lightning_indexer import lightning_indexer_topk_cpu, lightning_indexer_kl_loss_cpu
from fsdp_turbo.ops.cpu.tilekernels_mhc import tilekernels_mhc_post_cpu

from fsdp_turbo.ops.cpu.sparse_indexer_loss import sparse_indexer_gradient_torch

from fsdp_turbo.ops.cpu.vision_flash_attention import fused_vision_flash_attention_cpu

__all__ = [
    'lightning_indexer_topk_cpu',
    'lightning_indexer_kl_loss_cpu',
    'lightning_indexer_cpu',
    'indexer_kl_loss_cpu',
    'selected_attention_cpu',
    'fused_vision_flash_attention_cpu',
    'sparse_indexer_gradient_torch',
    'attention_output_projection_cpu',
    'dequant_fp8_blockwise_cpu',
    'dequant_int8_per_row_cpu',
    'dynamic_block_quant_cpu',
    'dynamic_quant_cpu',
    'fp8_payload_quant_supported_cpu',
    'cce_loss_cpu',
    'attn_res_cpu',
    'clipped_swiglu_cpu',
    'causal_conv1d_cpu',
    'causal_conv1d_update_cpu',
    'chunk_kda_cpu',
    'chunk_gdn_cpu',
    'recurrent_gdn_cpu',
    'engram_forward_cpu',
    'engram_backward_cpu',
    'engram_wkv_cpu',
    'npu_fusion_attention_cpu',
    'grouped_matmul_cpu',
    'w4a16_grouped_matmul_cpu',
    'all2all_grouped_matmul_cpu',
    'grouped_matmul_all2all_cpu',
    'lightning_indexer_score_torch',
    'fused_lightning_indexer_cpu',
    'moonep_grouped_matmul_cpu',
    'moonep_grouped_matmul_dgrad_cpu',
    'moonep_grouped_matmul_wgrad_cpu',
    'mx_quant_matmul_cpu',
    'w4a16_quant_matmul_cpu',
    'dynamic_mx_quant_cpu',
    'dynamic_mx_quant_with_dual_axis_cpu',
    'grouped_dynamic_mx_quant_cpu',
    'swiglu_cpu',
    'permute_cpu',
    'unpermute_cpu',
    'qsa_reference_mask',
    'qsa_selected_token_mask_cpu',
    'qsa_compress_block_keys_cpu',
    'rms_norm_cpu',
    'rope_torch',
    'hc_split_sinkhorn_cpu',
    'indexed_sparse_attention_torch',
    'sparse_flash_attention_torch',
    'mx_to_block_mx_quant_cpu',
    'situ_glu_cpu',
    'flash_attention_cpu',
    'rotary_mul_cpu',
    'tilekernels_mhc_post_cpu',
]

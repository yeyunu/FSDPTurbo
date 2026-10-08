from fsdp_turbo.ops.npu.causal_conv1d import causal_conv1d_npu, causal_conv1d_update_npu
from fsdp_turbo.ops.npu.chunk_kda import chunk_kda_npu
from fsdp_turbo.ops.npu.grouped_matmul import grouped_matmul_npu
from fsdp_turbo.ops.npu.grouped_matmul_mc2 import all2all_grouped_matmul_npu, grouped_matmul_all2all_npu
from fsdp_turbo.ops.npu.mx_grouped_matmul import mx_grouped_matmul_npu
from fsdp_turbo.ops.npu.mx_quant_matmul import mx_quant_matmul_npu
from fsdp_turbo.ops.npu.mx_quant import (
    dynamic_mx_quant_npu,
    dynamic_mx_quant_with_dual_axis_npu,
    grouped_dynamic_mx_quant_npu,
    swiglu_npu,
    mx_to_block_mx_quant_npu,
)
from fsdp_turbo.ops.npu.permute import permute_npu, unpermute_npu
from fsdp_turbo.ops.npu.rms_norm import rms_norm_npu

try:
    from fsdp_turbo.ops.npu.sinkhorn import hc_split_sinkhorn_npu
except ImportError:  # triton is optional; the package must still load without it
    hc_split_sinkhorn_npu = None
from fsdp_turbo.ops.npu.situ_glu import situ_glu_npu
from fsdp_turbo.ops.npu.sparse_attention import (
    indexed_sparse_attention_npu,
    npu_sparse_flash_mla,
)
from fsdp_turbo.ops.npu.compressor import fused_compressor
from fsdp_turbo.ops.npu.lightning_indexer import fused_lightning_indexer
from fsdp_turbo.ops.npu.mhc import fused_mhc_post, fused_mhc_pre_sinkhorn_premix
from fsdp_turbo.ops.npu.vision_flash_attention import fused_vision_flash_attention

__all__ = [
    'causal_conv1d_npu',
    'causal_conv1d_update_npu',
    'chunk_kda_npu',
    'grouped_matmul_npu',
    'all2all_grouped_matmul_npu',
    'grouped_matmul_all2all_npu',
    'mx_grouped_matmul_npu',
    'mx_quant_matmul_npu',
    'dynamic_mx_quant_npu',
    'dynamic_mx_quant_with_dual_axis_npu',
    'grouped_dynamic_mx_quant_npu',
    'swiglu_npu',
    'permute_npu',
    'unpermute_npu',
    'rms_norm_npu',
    'hc_split_sinkhorn_npu',
    'mx_to_block_mx_quant_npu',
    'situ_glu_npu',
    'indexed_sparse_attention_npu',
    'npu_sparse_flash_mla',
    'fused_compressor',
    'fused_lightning_indexer',
    'fused_mhc_post',
    'fused_mhc_pre_sinkhorn_premix',
    'fused_vision_flash_attention',
]

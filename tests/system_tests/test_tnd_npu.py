# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Device checks for the merged packed-TND contracts.

Run on an idle Ascend card with ``RUN_NPU_TESTS=1`` and the DeepSeek-V4.1 CANN
environment. Only the ``cu_seqlens_ori_kv`` metadata ABI needs a device; the
attention arm is covered on CPU.
"""

import torch

from fsdp_turbo.utils.env import env_flag


def _npu_available() -> bool:
    """Whether this host can run the device tests at all."""
    try:
        import torch_npu  # noqa: F401  (registers torch.npu)
    except ImportError:
        return False
    return torch.npu.is_available()


pytestmark = __import__("pytest").mark.skipif(
    not env_flag("RUN_NPU_TESTS") or not _npu_available(),
    reason="requires an idle NPU and the DeepSeek-V4.1 CANN/custom-op environment",
)

# aclnnSparseFlashMlaMetadata requires head_dim=512; use the production layout.
HEAD_DIM = 512
HEADS = 64
RATIO = 2
WINDOW = 128


def _one_document_segment():
    """A 256-token document whose first half is another rank's query segment."""
    from fsdp_turbo.ops.sparse_attention import indexed_sparse_attention

    query_rows, kv_rows, cmp_rows = 128, 256, 128
    scale = HEAD_DIM**-0.5
    query = torch.randn(query_rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    kv = torch.randn(kv_rows, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    compressed = torch.randn(cmp_rows, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    sink = torch.zeros(HEADS, dtype=torch.float32, device="npu")
    topk = min(512, cmp_rows)
    indices = torch.full((query_rows, topk), -1, dtype=torch.int32, device="npu")
    indices[:, :topk] = torch.arange(topk, device="npu").unsqueeze(0)
    return indexed_sparse_attention(
        query,
        kv,
        sink,
        None,
        scale,
        use_sparse_flash_attn=True,
        compressed_key_value=compressed,
        compressed_topk_indices=indices,
        compress_ratio=RATIO,
        window_size=WINDOW,
        layout="TND",
        # Query [128, 256) vs full-document KV and compressed rows: lengths differ.
        cu_seqlens=torch.tensor([0, query_rows], dtype=torch.int32, device="npu"),
        cu_seqlens_ori_kv=torch.tensor([0, kv_rows], dtype=torch.int32, device="npu"),
        cu_seqlens_cmp_kv=torch.tensor([0, cmp_rows], dtype=torch.int32, device="npu"),
    )


def test_sparse_flash_mla_accepts_a_kv_prefix_longer_than_the_query():
    """Metadata must be built from cu_seqlens_ori_kv, not from the query rows."""
    out = _one_document_segment()
    torch.npu.synchronize()
    assert out.shape == (128, HEADS, HEAD_DIM)
    assert torch.isfinite(out.float()).all()

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""DeepSeek-V4.1 adapters for CANN lightning indexers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.registry import register_op


_FP4_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_SUPPORTED_PAGE_SIZES = (64, 128)


class _DirectArenaOps:
    """Normalize the arena Python entry points to the ds41 call schema."""

    def __init__(self, quant_lightning_indexer, quant_sparse_lightning_indexer):
        self._qli = quant_lightning_indexer
        self._qsli = quant_sparse_lightning_indexer

    def quant_lightning_indexer(self, q, k, w, q_descale, k_descale, topk, quant_mode, **kwargs):
        return self._qli(
            q,
            k,
            w,
            q_descale,
            k_descale,
            topk=topk,
            quant_mode=quant_mode,
            **kwargs,
        )

    def quant_sparse_lightning_indexer(
        self,
        q,
        k,
        w,
        q_descale,
        k_descale,
        candidate_block_indices,
        candidate_block_length,
        topk,
        quant_mode,
        candidate_block_size,
        **kwargs,
    ):
        return self._qsli(
            q,
            k,
            w,
            q_descale,
            k_descale,
            candidate_block_indices,
            candidate_block_length,
            topk=topk,
            quant_mode=quant_mode,
            candidate_block_size=candidate_block_size,
            **kwargs,
        )


def _get_ds41_ops():
    # The arena package owns the precompiled QLI/QSLI resources. Import its
    # narrow entry points first: importing cann_ops_transformer.ops eagerly
    # initializes every installed transformer op and may query the NPU or
    # build unrelated extensions before it discovers that ds41 is absent.
    arena_error = None
    try:
        from ops.quant_lightning_indexer_dsl import quant_lightning_indexer
        from ops.quant_sparse_lightning_indexer_dsl import (
            quant_sparse_lightning_indexer,
        )

        return _DirectArenaOps(quant_lightning_indexer, quant_sparse_lightning_indexer)
    except (ImportError, AttributeError) as error:
        arena_error = error

    transformer_error = None
    try:
        from cann_ops_transformer.ops import ds41

        if hasattr(ds41, "quant_lightning_indexer") and hasattr(ds41, "quant_sparse_lightning_indexer"):
            return ds41
        transformer_error = AttributeError("ds41 does not expose both QLI and QSLI")
    except (ImportError, AttributeError) as error:
        transformer_error = error

    raise RuntimeError(
        "use_fused_lightning_indexer=True selected the Ascend950DT QLI/QSLI "
        "backend but could not find its entry points. "
        "Install matching cannbotdsl and cannbot_arena_net_ops "
        f"packages. Arena import failed: {arena_error!r}; "
        f"cann_ops_transformer ds41 import failed: {transformer_error!r}."
    ) from arena_error


def _pack_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension in 32-value MXFP4 groups.

    The DSL ABI uses uint8 as the carrier for two E2M1 values and uint8 as the
    carrier for E8M0 scales. Four scales for logical D=128 are exposed as
    ``[..., 2, 2]``.
    """
    if x.shape[-1] != 128:
        raise ValueError(f"MXFP4 lightning indexer requires D=128, got {x.shape[-1]}")

    blocks = x.float().reshape(*x.shape[:-1], 4, 32)
    amax = blocks.abs().amax(dim=-1)
    minimum_amax = 6.0 * (2.0**-126)
    exponent = torch.ceil(torch.log2(amax.clamp_min(minimum_amax) / 6.0))
    exponent = exponent.clamp(-126, 127)
    scale = torch.exp2(exponent).unsqueeze(-1)
    scaled = (blocks / scale).clamp(-6.0, 6.0)

    magnitude = scaled.abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for threshold in _FP4_THRESHOLDS:
        code.add_(magnitude >= threshold)
    code = code | ((scaled < 0).to(torch.uint8) << 3)
    code = code.flatten(-2)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    scale_bytes = (exponent.to(torch.int16) + 127).to(torch.uint8)
    return packed.contiguous(), scale_bytes.reshape(*x.shape[:-1], 2, 2).contiguous()


def _make_paged_key(key: torch.Tensor, page_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert dense ``[B,K,128]`` keys to PA_BBND and a sequential block table."""
    if page_size not in _SUPPORTED_PAGE_SIZES:
        raise ValueError(f"Precompiled DS41 indexers support page_size 64 or 128, got {page_size}")
    if key.ndim != 3 or key.shape[-1] != 128:
        raise ValueError(f"Indexer key must have shape [B,K,128], got {tuple(key.shape)}")

    batch_size, key_length, _ = key.shape
    pages_per_sequence = math.ceil(key_length / page_size)
    padded_length = pages_per_sequence * page_size
    padded = F.pad(key, (0, 0, 0, padded_length - key_length))
    paged = padded.reshape(batch_size, pages_per_sequence, page_size, 128)
    paged = paged.reshape(batch_size * pages_per_sequence, page_size, 1, 128)
    packed_key, key_scale = _pack_mxfp4(paged)
    block_table = torch.arange(batch_size * pages_per_sequence, dtype=torch.int32, device=key.device).reshape(
        batch_size, pages_per_sequence
    )
    return packed_key, key_scale, block_table


def _sequence_metadata(
    query: torch.Tensor,
    key_length: int,
    compress_ratio: int,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch_size, sequence_length = query.shape[:2]
    device = query.device
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * sequence_length,
        sequence_length,
        dtype=torch.int32,
        device=device,
    )
    if attention_mask is None:
        seqused_q = torch.full((batch_size,), sequence_length, dtype=torch.int32, device=device)
    else:
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                "Lightning indexer attention_mask must have shape "
                f"{(batch_size, sequence_length)}, got {tuple(attention_mask.shape)}"
            )
        seqused_q = attention_mask.to(device=device, dtype=torch.int32).sum(dim=-1, dtype=torch.int32)
    seqused_k = torch.div(seqused_q, compress_ratio, rounding_mode="floor")
    seqused_k.clamp_(max=key_length)
    residual = seqused_q.remainder(compress_ratio) if compress_ratio != 1 else None
    return cu_seqlens_q, seqused_q.contiguous(), seqused_k.contiguous(), residual


def _fused_quant_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weights: torch.Tensor,
    *,
    topk: int,
    compress_ratio: int,
    index_offset: int,
    attention_mask: torch.Tensor | None,
    page_size: int,
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
    candidate_block_indices: torch.Tensor | None = None,
    candidate_block_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run QLI or QSLI and return indices plus optional QLI candidates."""
    if query.device.type != "npu":
        raise ValueError("Fused DS41 lightning indexers require NPU tensors")
    if query.ndim != 4 or query.shape[2:] != (32, 128):
        raise ValueError(f"Indexer query must have shape [B,S,32,128], got {tuple(query.shape)}")
    if head_weights.shape != query.shape[:3]:
        raise ValueError(f"Indexer weights must have shape {tuple(query.shape[:3])}, got {tuple(head_weights.shape)}")
    if topk != 512:
        raise ValueError(f"Packaged DS41 AOT indexers require topk=512, got {topk}")
    if compress_ratio not in (1, 2):
        raise ValueError(f"Packaged DS41 AOT indexers support compress_ratio 1 or 2, got {compress_ratio}")

    batch_size, sequence_length = query.shape[:2]
    packed_query, query_scale = _pack_mxfp4(query.reshape(-1, 32, 128))
    packed_key, key_scale, block_table = _make_paged_key(key, page_size)
    cu_q, used_q, used_k, residual = _sequence_metadata(query, key.shape[1], compress_ratio, attention_mask)
    weights = head_weights.reshape(-1, 32).float().contiguous()
    common_kwargs = dict(
        cu_seqlens_q=cu_q,
        seqused_q=used_q,
        seqused_k=used_k,
        cmp_residual_k=residual,
        block_table=block_table,
        metadata=None,
        max_seqlen_q=sequence_length,
        mask_mode=3,
        cmp_ratio=compress_ratio,
        layout_q="TND",
        layout_kv="PA_BBND",
        return_value=False,
    )
    ds41 = _get_ds41_ops()

    if candidate_block_indices is None:
        indices, _, candidates, candidate_lengths = ds41.quant_lightning_indexer(
            packed_query,
            packed_key,
            weights,
            query_scale,
            key_scale,
            topk,
            1,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            **common_kwargs,
        )
        if candidate_topk_blocks <= 0:
            candidates = candidate_lengths = None
        if index_offset:
            indices = torch.where(indices >= 0, indices + index_offset, indices)
    else:
        if candidate_block_length is None:
            raise ValueError("candidate_block_length is required with candidate indices")
        output_offset = torch.full(
            (batch_size * sequence_length, 1),
            index_offset,
            dtype=torch.int32,
            device=query.device,
        )
        indices, _ = ds41.quant_sparse_lightning_indexer(
            packed_query,
            packed_key,
            weights,
            query_scale,
            key_scale,
            candidate_block_indices,
            candidate_block_length,
            topk,
            1,
            candidate_block_size,
            output_idx_offset=output_offset,
            **common_kwargs,
        )
        candidates = candidate_lengths = None

    return indices.reshape(batch_size, sequence_length, topk), candidates, candidate_lengths


def _fused_candidate_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weights: torch.Tensor,
    *,
    topk: int,
    compress_ratio: int,
    index_offset: int,
    attention_mask: torch.Tensor | None,
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
    candidate_block_indices: torch.Tensor | None = None,
    candidate_block_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    # Ascend 910B, 910C, and 910_93 are not currently supported.
    pass


def _get_npu_device_name(query: torch.Tensor) -> str:
    if query.device.type != "npu":
        raise ValueError("Fused DS41 lightning indexers require NPU tensors")
    return torch.accelerator.get_device_name(query.device.index)


def _select_lightning_indexer_backend(device_name: str) -> str:
    normalized_name = device_name.upper().replace("-", "").replace(" ", "")
    if "ASCEND950" in normalized_name or "ASCEND910_95" in normalized_name:
        return "quant"
    if any(model in normalized_name for model in ("ASCEND910B", "ASCEND910C", "ASCEND910_93")):
        return "candidate"
    raise RuntimeError(
        "use_fused_lightning_indexer=True supports only Ascend 910B/910C "
        "and Ascend950DT, but the current device is "
        f"{device_name!r}"
    )


@register_op("fused_lightning_indexer", "npu")
def fused_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weights: torch.Tensor,
    *,
    topk: int,
    compress_ratio: int,
    index_offset: int,
    attention_mask: torch.Tensor | None,
    page_size: int,
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
    candidate_block_indices: torch.Tensor | None = None,
    candidate_block_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Select the fused DS41 indexer implementation for the current Ascend SoC."""
    common_kwargs = dict(
        topk=topk,
        compress_ratio=compress_ratio,
        index_offset=index_offset,
        attention_mask=attention_mask,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        candidate_block_indices=candidate_block_indices,
        candidate_block_length=candidate_block_length,
    )
    backend = _select_lightning_indexer_backend(_get_npu_device_name(query))
    if backend == "quant":
        return _fused_quant_lightning_indexer(
            query,
            key,
            head_weights,
            page_size=page_size,
            **common_kwargs,
        )
    return _fused_candidate_lightning_indexer(
        query,
        key,
        head_weights,
        **common_kwargs,
    )


__all__ = ["fused_lightning_indexer"]

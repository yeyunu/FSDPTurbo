# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""DeepSeek-V4.1 adapters for CANN lightning indexers."""

from __future__ import annotations

import math
from importlib import import_module

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


_MX_BLOCK_SIZE = 32


class _DirectArenaOps:
    """Normalize the arena Python entry points to the ds41 call schema."""

    def __init__(
        self,
        quant_lightning_indexer,
        quant_sparse_lightning_indexer,
        quant_lightning_indexer_metadata,
        quant_sparse_lightning_indexer_metadata,
    ):
        self._qli = quant_lightning_indexer
        self._qsli = quant_sparse_lightning_indexer
        self.quant_lightning_indexer_metadata = quant_lightning_indexer_metadata
        self.quant_sparse_lightning_indexer_metadata = quant_sparse_lightning_indexer_metadata

    def quant_lightning_indexer(self, q, k, w, descale_q, descale_k, topk, quant_mode, **kwargs):
        return self._qli(
            q,
            k,
            w,
            descale_q,
            descale_k,
            topk=topk,
            quant_mode=quant_mode,
            **kwargs,
        )

    def quant_sparse_lightning_indexer(
        self,
        q,
        k,
        w,
        descale_q,
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
            descale_q,
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
        from ops.quant_lightning_indexer_metadata_dsl import quant_lightning_indexer_metadata
        from ops.quant_sparse_lightning_indexer_metadata_dsl import quant_sparse_lightning_indexer_metadata

        return _DirectArenaOps(
            quant_lightning_indexer,
            quant_sparse_lightning_indexer,
            quant_lightning_indexer_metadata,
            quant_sparse_lightning_indexer_metadata,
        )
    except (ImportError, AttributeError) as error:
        arena_error = error

    transformer_error = None
    try:
        from cann_ops_transformer.ops import ds41

        if all(
            hasattr(ds41, name)
            for name in (
                "quant_lightning_indexer",
                "quant_sparse_lightning_indexer",
                "quant_lightning_indexer_metadata",
                "quant_sparse_lightning_indexer_metadata",
            )
        ):
            return ds41
        transformer_error = AttributeError("ds41 must expose QLI/QSLI and both metadata entry points")
    except (ImportError, AttributeError) as error:
        transformer_error = error

    raise RuntimeError(
        "use_fused_lightning_indexer=True selected the Ascend950DT QLI/QSLI "
        "backend but could not find its entry points. "
        "Install matching cannbotdsl and cannbot_arena_net_ops "
        f"packages. Arena import failed: {arena_error!r}; "
        f"cann_ops_transformer ds41 import failed: {transformer_error!r}."
    ) from arena_error


class _CandidateOps:
    def __init__(self, lightning_indexer, candidate_source, candidate_consumer):
        self.lightning_indexer = lightning_indexer
        self.candidate_source = candidate_source
        self.candidate_consumer = candidate_consumer


def _get_candidate_ops():
    try:
        custom_ops = import_module("cann_ops_transformer_custom_li")
        return _CandidateOps(
            custom_ops.lightning_indexer,
            custom_ops.lightning_indexer_candidate,
            custom_ops.sparse_lightning_indexer,
        )
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "use_fused_lightning_indexer=True requires lightning_indexer_v2 "
            "from cann_ops_transformer and lightning_indexer_candidate and "
            "sparse_lightning_indexer from cann_ops_transformer_custom."
        ) from error


def _pack_mxfp4_npu(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack an operand with the fused TorchNPU quantizer used by DS41 QLI/QSLI."""
    if torch_npu is None:
        raise RuntimeError("MXFP4 lightning indexer quantization requires torch_npu")

    data, scale = torch_npu.npu_dynamic_mx_quant(
        x.contiguous(),
        axis=-1,
        dst_type=torch_npu.float4_e2m1fn_x2,
        block_size=_MX_BLOCK_SIZE,
        round_mode="rint",
        scale_alg=2,
        dst_type_max=0.0,
    )
    return (
        data.view(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // 2).contiguous(),
        scale.view(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // 64, 2).contiguous(),
    )


def _pack_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension in 32-value MXFP4 groups.

    The DSL ABI uses uint8 as the carrier for two E2M1 values and uint8 as the
    carrier for E8M0 scales. Four scales for logical D=128 are exposed as
    ``[..., 2, 2]``.
    """
    if x.shape[-1] != 128:
        raise ValueError(f"MXFP4 lightning indexer requires D=128, got {x.shape[-1]}")
    if x.device.type != "npu":
        raise ValueError("MXFP4 lightning indexer quantization requires an NPU tensor")
    return _pack_mxfp4_npu(x)


def _make_paged_key(key: torch.Tensor, page_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert dense ``[B,K,128]`` keys to PA_BBND and a sequential block table."""
    if page_size <= 0 or page_size > 1024 or page_size % 16:
        raise ValueError(f"DS41 indexers require page_size to be a multiple of 16 in [16,1024], got {page_size}")
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


def _pack_sparse_key(key: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Store each logical 8-token block as 512 K bytes followed by 32 scale bytes."""
    pages, page_size = key.shape[:2]
    return torch.cat(
        (
            key.reshape(pages, page_size // 8, 512),
            scale.reshape(pages, page_size // 8, 32),
        ),
        dim=-1,
    ).contiguous()


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
    if not 1 <= topk <= 8192:
        raise ValueError(f"DS41 indexers require topk in [1,8192], got {topk}")
    if not 1 <= compress_ratio <= 128:
        raise ValueError(f"DS41 indexers require compress_ratio in [1,128], got {compress_ratio}")
    if candidate_block_indices is not None:
        if candidate_block_length is None:
            raise ValueError("candidate_block_length is required with candidate indices")
        if candidate_block_size != 8:
            raise ValueError("QSLI requires candidate_block_size=8")
        tokens = query.shape[0] * query.shape[1]
        if candidate_block_indices.shape != (tokens, 1, 2048):
            raise ValueError("QSLI candidate indices must have shape [B*S,1,2048]")
        if candidate_block_length.shape != (tokens, 1):
            raise ValueError("QSLI candidate lengths must have shape [B*S,1]")

    batch_size, sequence_length = query.shape[:2]
    packed_query, query_scale = _pack_mxfp4(query.reshape(-1, 32, 128))
    packed_key, key_scale, block_table = _make_paged_key(key, page_size)
    cu_q, used_q, used_k, residual = _sequence_metadata(query, key.shape[1], compress_ratio, attention_mask)
    weights = head_weights.reshape(-1, 32).float().contiguous()
    sequence_kwargs = dict(
        cu_seqlens_q=cu_q,
        seqused_q=used_q,
        seqused_k=used_k,
        cmp_residual_k=residual,
        max_seqlen_q=sequence_length,
        mask_mode=3,
        cmp_ratio=compress_ratio,
        layout_q="TND",
        layout_k="PA_BBND",
    )
    metadata_kwargs = dict(
        **sequence_kwargs,
        batch_size=batch_size,
        max_seqlen_k=key.shape[1],
        num_heads_q=32,
        num_heads_k=1,
        head_dim=128,
        topk=topk,
    )
    common_kwargs = dict(
        **sequence_kwargs,
        block_table=block_table,
        output_idx_offset=torch.full(
            (batch_size * sequence_length, 1),
            index_offset,
            dtype=torch.int32,
            device=packed_query.device,
        ),
        return_value=False,
    )
    ds41 = _get_ds41_ops()

    if candidate_block_indices is None:
        metadata = ds41.quant_lightning_indexer_metadata(
            **metadata_kwargs,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
        )
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
            metadata=metadata,
            **common_kwargs,
        )
        if candidate_topk_blocks <= 0:
            candidates = candidate_lengths = None
    else:
        candidate_block_indices = candidate_block_indices.to(device=packed_query.device, dtype=torch.int32).contiguous()
        candidate_block_length = candidate_block_length.to(device=packed_query.device, dtype=torch.int32).contiguous()
        metadata = ds41.quant_sparse_lightning_indexer_metadata(
            candidate_block_length,
            **metadata_kwargs,
            quant_mode=1,
            candidate_block_size=candidate_block_size,
        )
        indices, _ = ds41.quant_sparse_lightning_indexer(
            packed_query,
            _pack_sparse_key(packed_key, key_scale),
            weights,
            query_scale,
            candidate_block_indices,
            candidate_block_length,
            topk,
            1,
            candidate_block_size,
            descale_k=None,
            metadata=metadata,
            **common_kwargs,
        )
        candidates = candidate_lengths = None

    return (
        indices.reshape(batch_size, sequence_length, topk),
        candidates,
        candidate_lengths,
    )


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
    """Run LightningIndexerV2 or its sparse candidate consumer."""
    if query.device.type != "npu":
        raise ValueError("Fused DS41 lightning indexers require NPU tensors")
    if query.ndim != 4 or query.shape[2:] != (32, 128):
        raise ValueError(f"Indexer query must have shape [B,S,32,128], got {tuple(query.shape)}")
    if key.ndim != 3 or key.shape[0] != query.shape[0] or key.shape[-1] != 128:
        raise ValueError(f"Indexer key must have shape [B,K,128], got {tuple(key.shape)}")
    if head_weights.shape != query.shape[:3]:
        raise ValueError(f"Indexer weights must have shape {tuple(query.shape[:3])}, got {tuple(head_weights.shape)}")
    if not 0 < topk <= 2048:
        raise ValueError(f"Candidate lightning indexers require topk in [1, 2048], got {topk}")
    if compress_ratio < 1 or compress_ratio > 128 or compress_ratio & (compress_ratio - 1):
        raise ValueError(
            f"Candidate lightning indexers require a power-of-two compress_ratio in [1, 128], got {compress_ratio}"
        )
    if candidate_block_indices is not None and candidate_topk_blocks != -1:
        raise ValueError("candidate_topk_blocks must be -1 when consuming candidate indices")
    if candidate_topk_blocks != -1 and (
        candidate_topk_blocks <= 0 or candidate_topk_blocks > 2048 or candidate_topk_blocks % 64
    ):
        raise ValueError(
            "Candidate lightning indexer source requires "
            "candidate_topk_blocks to be -1 or a multiple of 64 in [64, 2048], "
            f"got {candidate_topk_blocks}"
        )
    uses_candidates = candidate_block_indices is not None or candidate_topk_blocks > 0
    if uses_candidates and (
        candidate_block_size < 2 or candidate_block_size > 64 or candidate_block_size & (candidate_block_size - 1)
    ):
        raise ValueError(
            "Candidate lightning indexers require a power-of-two "
            f"candidate_block_size in [2, 64], got {candidate_block_size}"
        )

    batch_size, sequence_length = query.shape[:2]
    expected_key_length = sequence_length // compress_ratio
    if key.shape[1] != expected_key_length:
        raise ValueError(
            "Candidate lightning indexers require the physical compressed K length "
            f"to equal floor(S/compress_ratio)={expected_key_length}, got {key.shape[1]}"
        )

    valid_queries = None
    if attention_mask is not None:
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                "Lightning indexer attention_mask must have shape "
                f"{(batch_size, sequence_length)}, got {tuple(attention_mask.shape)}"
            )
        valid_queries = attention_mask.to(device=query.device, dtype=torch.bool)
    residual = (
        torch.full(
            (batch_size,),
            sequence_length % compress_ratio,
            dtype=torch.int32,
            device=query.device,
        )
        if compress_ratio != 1
        else None
    )
    common_kwargs = dict(
        seqused_k=None,
        cmp_residual_k=residual,
        metadata=None,
        max_seqlen_q=-1,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=compress_ratio,
    )
    candidate_ops = _get_candidate_ops()
    contiguous_query = query.contiguous()
    contiguous_key = key.unsqueeze(2).contiguous()
    contiguous_weights = head_weights.float().contiguous()

    if candidate_block_indices is not None:
        indices, _ = candidate_ops.candidate_consumer(
            contiguous_query,
            contiguous_key,
            contiguous_weights,
            topk,
            candidate_block_indices,
            candidate_block_length=candidate_block_length,
            candidate_block_size=candidate_block_size,
            **common_kwargs,
        )
        candidates = candidate_lengths = None
    elif candidate_topk_blocks > 0:
        indices, _, candidates, candidate_lengths = candidate_ops.candidate_source(
            contiguous_query,
            contiguous_key,
            contiguous_weights,
            topk,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            **common_kwargs,
        )
    else:
        indices, _ = candidate_ops.lightning_indexer(
            contiguous_query,
            contiguous_key,
            contiguous_weights,
            topk,
            return_value=0,
            **common_kwargs,
        )
        candidates = candidate_lengths = None

    if index_offset:
        indices = torch.where(indices >= 0, indices + index_offset, indices)
    if valid_queries is not None:
        invalid_queries = ~valid_queries[:, :, None, None]
        indices = indices.masked_fill(invalid_queries, -1)
        if candidates is not None:
            candidates = candidates.masked_fill(invalid_queries, -1)
    return indices.squeeze(2), candidates, candidate_lengths


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
        "use_fused_lightning_indexer=True supports only Atlas A3 (Ascend 910_93), "
        "Ascend 910B/910C, and Ascend950DT, but the current device is "
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

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""NPU implementations for differentiable indexed sparse attention."""

import os
from contextlib import nullcontext
from functools import lru_cache

import torch

from fsdp_turbo.ops.cpu.sparse_attention import indexed_sparse_attention_torch
from fsdp_turbo.ops.registry import register_op


def _get_cann_ops():
    try:
        from cann_ops_transformer import ops
    except ImportError as error:
        raise RuntimeError(
            "use_sparse_flash_attn=True requires cann_ops_transformer with "
            "SparseFlashMla and SparseFlashMlaGrad support."
        ) from error
    return ops


def _cmp_mask_mode(has_compressed_kv: bool) -> int:
    if has_compressed_kv:
        return 3
    mode = int(os.getenv("FSDP_TURBO_CMP_MASK_MODE_WITHOUT_KV", "0"))
    if mode not in (0, 3):
        raise ValueError("FSDP_TURBO_CMP_MASK_MODE_WITHOUT_KV must be 0 (A5) or 3 (A2/A3).")
    return mode


@lru_cache(maxsize=16)
def _is_a2_or_a3(device_index: int | None) -> bool:
    device_name = torch.accelerator.get_device_name(device_index)
    normalized_name = device_name.upper().replace("-", "").replace(" ", "")
    return any(model in normalized_name for model in ("ASCEND910B", "ASCEND910C", "ASCEND910_93"))


def _residual_tensor(residual, batch_size: int, device, required: bool):
    if not required:
        return None
    if isinstance(residual, torch.Tensor):
        return residual.to(device=device, dtype=torch.int32)
    return torch.full((batch_size,), residual, dtype=torch.int32, device=device)


def _as_int32_cu_seqlens(cu_seqlens):
    """Normalize cumulative sequence lengths at the CANN operator boundary."""
    if cu_seqlens is None or cu_seqlens.dtype == torch.int32:
        return cu_seqlens
    return cu_seqlens.to(dtype=torch.int32)


def _input_summary(
    query: torch.Tensor,
    original_kv: torch.Tensor,
    compressed_kv: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    attention_sink: torch.Tensor,
    compress_ratio: int,
    window_size: int,
) -> str:
    def describe(tensor):
        if tensor is None:
            return "None"
        return (
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"stride={tensor.stride()}, contiguous={tensor.is_contiguous()}"
        )

    return (
        f"q=({describe(query)}), ori_kv=({describe(original_kv)}), "
        f"cmp_kv=({describe(compressed_kv)}), "
        f"cmp_indices=({describe(compressed_indices)}), "
        f"sinks=({describe(attention_sink)}), cmp_ratio={compress_ratio}, "
        f"window_size={window_size}"
    )


def _validate_sparse_flash_mla_inputs(
    query: torch.Tensor,
    original_kv: torch.Tensor,
    compressed_kv: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    attention_sink: torch.Tensor,
    compress_ratio: int,
    window_size: int,
) -> None:
    """Fail before kernel launch when an A5 SparseFlashMla contract is violated."""
    summary = _input_summary(
        query,
        original_kv,
        compressed_kv,
        compressed_indices,
        attention_sink,
        compress_ratio,
        window_size,
    )

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"SparseFlashMla {message}. {summary}")

    require(query.ndim == 4, "expects q with shape [B,S,N,D]")
    require(original_kv.ndim == 3, "expects ori_kv with shape [B,S,D]")
    require(query.shape[0] == original_kv.shape[0], "batch sizes must match")
    require(query.shape[1] > 0, "requires a non-empty query sequence")
    require(original_kv.shape[1] > 0, "requires a non-empty original KV")
    require(query.shape[-1] == 512, "requires query head_dim=512")
    require(original_kv.shape[-1] == 512, "requires original KV head_dim=512")
    require(1 <= query.shape[2] <= 128, "requires 1..128 query heads on A5")
    require(query.dtype in (torch.bfloat16, torch.float16), "requires BF16 or FP16 q")
    require(original_kv.dtype == query.dtype, "requires q and ori_kv dtypes to match")
    require(original_kv.device == query.device, "requires q and ori_kv on one device")
    require(query.is_contiguous(), "requires contiguous q")
    require(isinstance(compress_ratio, int), "requires an integer cmp_ratio")
    require(1 <= compress_ratio <= 128, "requires cmp_ratio in [1, 128]")
    require(window_size > 0, "requires a positive window_size")
    require(attention_sink.ndim == 1, "expects sinks with shape [N]")
    require(attention_sink.device == query.device, "requires sinks on the q device")
    require(
        attention_sink.numel() == query.shape[2],
        "requires one sink value per query head",
    )

    has_compressed_kv = compressed_kv is not None
    require(
        has_compressed_kv == (compressed_indices is not None),
        "requires cmp_kv and cmp_sparse_indices to be both present or both absent",
    )
    if not has_compressed_kv:
        require(compress_ratio == 1, "requires cmp_ratio=1 without compressed KV")
        return

    require(compressed_kv.ndim == 3, "expects cmp_kv with shape [B,S,D]")
    require(
        compressed_kv.shape[0] == query.shape[0],
        "requires q and cmp_kv batch sizes to match",
    )
    require(compressed_kv.shape[1] > 0, "requires a non-empty compressed KV")
    require(compressed_kv.shape[-1] == 512, "requires compressed KV head_dim=512")
    require(compressed_kv.dtype == query.dtype, "requires q and cmp_kv dtypes to match")
    require(compressed_kv.device == query.device, "requires q and cmp_kv on one device")
    require(
        compressed_indices.ndim == 3,
        "expects cmp_sparse_indices with shape [B,S,K] before padding",
    )
    require(
        compressed_indices.shape[:2] == query.shape[:2],
        "requires cmp_sparse_indices batch/query dimensions to match q",
    )
    require(
        compressed_indices.dtype == torch.int32,
        "requires INT32 cmp_sparse_indices",
    )
    require(
        compressed_indices.device == query.device,
        "requires cmp_sparse_indices on the q device",
    )
    require(compressed_indices.shape[-1] > 0, "requires a non-empty compressed top-k")
    require(
        compressed_indices.shape[-1] <= 1024,
        "supports at most 1024 compressed indices",
    )
    require(
        compressed_kv.shape[1] * compress_ratio + original_kv.shape[1] % compress_ratio == original_kv.shape[1],
        "requires ori_len == cmp_len * cmp_ratio + residual",
    )

    if os.getenv("FSDP_TURBO_VALIDATE_SPARSE_FLASH_MLA", "0") != "1":
        return

    indices = compressed_indices.to(dtype=torch.int32)
    valid = indices >= 0
    invalid_seen = (~valid).cummax(dim=-1).values
    violations = torch.stack(
        (
            (indices < -1).any(),
            (indices >= compressed_kv.shape[1]).any(),
            (invalid_seen & valid).any(),
        )
    ).to(device="cpu")
    below_minus_one, out_of_range, valid_after_invalid = violations.tolist()
    require(not below_minus_one, "only accepts -1 as the invalid index sentinel")
    require(not out_of_range, "contains a compressed index outside cmp_kv")
    require(
        not valid_after_invalid,
        "requires valid compressed indices before all -1 padding",
    )


@lru_cache(maxsize=128)
def _cached_forward_metadata(
    device_type,
    device_index,
    stream_id,
    deterministic,
    residual,
    *,
    num_heads_q,
    num_heads_kv,
    head_dim,
    batch_size,
    max_seqlen_q,
    max_seqlen_ori_kv,
    max_seqlen_cmp_kv,
    cmp_topk,
    cmp_ratio,
    cmp_mask_mode,
    ori_win_left,
    has_cmp_kv,
):
    """Create BSND forward metadata; stream and deterministic mode isolate cache entries."""
    device = torch.device(device_type, device_index)
    device_context = torch.npu.device(device) if device_type == "npu" else nullcontext()
    with device_context:
        return _get_cann_ops().sparse_flash_mla_metadata(
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            cu_seqlens_q=None,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            cmp_residual_kv=_residual_tensor(residual, batch_size, device, has_cmp_kv and cmp_ratio != 1),
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_ori_kv=max_seqlen_ori_kv,
            max_seqlen_cmp_kv=max_seqlen_cmp_kv,
            ori_topk=0,
            cmp_topk=cmp_topk,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=0,
            layout_q="BSND",
            layout_kv="BSND",
            has_ori_kv=True,
            has_cmp_kv=has_cmp_kv,
        )


@lru_cache(maxsize=128)
def _cached_backward_metadata(
    device_type,
    device_index,
    stream_id,
    deterministic,
    residual,
    *,
    num_heads_q,
    num_heads_kv,
    head_dim,
    batch_size,
    max_seqlen_q,
    max_seqlen_ori_kv,
    max_seqlen_cmp_kv,
    cmp_topk,
    cmp_ratio,
    cmp_mask_mode,
    ori_win_left,
    has_cmp_kv,
):
    """Create BSND backward metadata; stream and deterministic mode isolate cache entries."""
    device = torch.device(device_type, device_index)
    device_context = torch.npu.device(device) if device_type == "npu" else nullcontext()
    with device_context:
        return _get_cann_ops().sparse_flash_mla_grad_metadata(
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            cu_seqlens_q=None,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            cmp_residual_kv=_residual_tensor(residual, batch_size, device, has_cmp_kv and cmp_ratio != 1),
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_ori_kv=max_seqlen_ori_kv,
            max_seqlen_cmp_kv=max_seqlen_cmp_kv,
            ori_topk=0,
            cmp_topk=cmp_topk,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=0,
            layout_q="BSND",
            layout_kv="BSND",
            has_ori_kv=True,
            has_cmp_kv=has_cmp_kv,
        )


def _metadata_cache_context(
    layout, residual, device_type, device_index, cu_seqlens, cu_seqlens_cmp, cu_seqlens_kv=None
):
    """Resolve BSND cache eligibility, device index, and current stream.

    A cached entry has no per-batch cu_seqlens at all, so any of the three
    lengths being present (TND, or a KV prefix longer than the query) disables
    the cache: reusing BSND metadata with TND bounds would be silent corruption.
    """
    cacheable = (
        layout == "BSND"
        and cu_seqlens is None
        and cu_seqlens_cmp is None
        and cu_seqlens_kv is None
        and isinstance(residual, int)
    )
    stream_id = None
    if cacheable and device_type == "npu":
        if device_index is None:
            device_index = torch.npu.current_device()
        with torch.npu.device(torch.device(device_type, device_index)):
            stream_id = torch.npu.current_stream().npu_stream
            is_capturing = getattr(torch.npu, "is_current_stream_capturing", None)
            if is_capturing is not None and is_capturing():
                cacheable = False
    return cacheable, device_index, stream_id


def _forward_metadata(
    *,
    batch_size,
    num_query_heads,
    num_kv_heads,
    head_dim,
    topk,
    has_compressed_kv,
    compress_ratio,
    query_length,
    original_kv_length,
    compressed_kv_length,
    residual,
    window_size,
    device_type,
    device_index,
    layout,
    cu_seqlens,
    cu_seqlens_cmp,
    cu_seqlens_kv=None,
):
    """Reuse cached BSND forward metadata; generate TND metadata for each call."""
    attributes = dict(
        num_heads_q=num_query_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        max_seqlen_q=query_length,
        max_seqlen_ori_kv=original_kv_length,
        max_seqlen_cmp_kv=compressed_kv_length,
        cmp_topk=topk if has_compressed_kv else 0,
        cmp_ratio=compress_ratio,
        cmp_mask_mode=_cmp_mask_mode(has_compressed_kv),
        ori_win_left=window_size - 1,
        has_cmp_kv=has_compressed_kv,
    )
    cacheable, device_index, stream_id = _metadata_cache_context(
        layout, residual, device_type, device_index, cu_seqlens, cu_seqlens_cmp, cu_seqlens_kv
    )
    if cacheable:
        return _cached_forward_metadata(
            device_type,
            device_index,
            stream_id,
            torch.are_deterministic_algorithms_enabled(),
            residual,
            **attributes,
        )

    device = torch.device(device_type, device_index)
    device_context = torch.npu.device(device) if device_type == "npu" else nullcontext()
    with device_context:
        cu_seqlens = _as_int32_cu_seqlens(cu_seqlens)
        return _get_cann_ops().sparse_flash_mla_metadata(
            **attributes,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_ori_kv=_as_int32_cu_seqlens(cu_seqlens_kv if cu_seqlens_kv is not None else cu_seqlens),
            cu_seqlens_cmp_kv=_as_int32_cu_seqlens(cu_seqlens_cmp),
            cmp_residual_kv=_residual_tensor(residual, batch_size, device, has_compressed_kv and compress_ratio != 1),
            ori_topk=0,
            ori_mask_mode=4,
            ori_win_right=0,
            layout_q=layout,
            layout_kv=layout,
            has_ori_kv=True,
        )


def _backward_metadata(
    *,
    batch_size,
    num_query_heads,
    num_kv_heads,
    head_dim,
    topk,
    has_compressed_kv,
    compress_ratio,
    query_length,
    original_kv_length,
    compressed_kv_length,
    residual,
    window_size,
    device_type,
    device_index,
    layout,
    cu_seqlens,
    cu_seqlens_cmp,
    cu_seqlens_kv=None,
):
    """Reuse cached BSND backward metadata; generate TND metadata for each call."""
    attributes = dict(
        num_heads_q=num_query_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        max_seqlen_q=query_length,
        max_seqlen_ori_kv=original_kv_length,
        max_seqlen_cmp_kv=compressed_kv_length,
        cmp_topk=topk if has_compressed_kv else 0,
        cmp_ratio=compress_ratio,
        cmp_mask_mode=_cmp_mask_mode(has_compressed_kv),
        ori_win_left=window_size - 1,
        has_cmp_kv=has_compressed_kv,
    )
    cacheable, device_index, stream_id = _metadata_cache_context(
        layout, residual, device_type, device_index, cu_seqlens, cu_seqlens_cmp, cu_seqlens_kv
    )
    if cacheable:
        return _cached_backward_metadata(
            device_type,
            device_index,
            stream_id,
            torch.are_deterministic_algorithms_enabled(),
            residual,
            **attributes,
        )

    device = torch.device(device_type, device_index)
    device_context = torch.npu.device(device) if device_type == "npu" else nullcontext()
    with device_context:
        cu_seqlens = _as_int32_cu_seqlens(cu_seqlens)
        return _get_cann_ops().sparse_flash_mla_grad_metadata(
            **attributes,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_ori_kv=_as_int32_cu_seqlens(cu_seqlens_kv if cu_seqlens_kv is not None else cu_seqlens),
            cu_seqlens_cmp_kv=_as_int32_cu_seqlens(cu_seqlens_cmp),
            cmp_residual_kv=_residual_tensor(residual, batch_size, device, has_compressed_kv and compress_ratio != 1),
            ori_topk=0,
            ori_mask_mode=4,
            ori_win_right=0,
            layout_q=layout,
            layout_kv=layout,
            has_ori_kv=True,
        )


class _SparseFlashMlaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        original_kv,
        compressed_kv,
        compressed_indices,
        attention_sink,
        softmax_scale,
        compress_ratio,
        window_size,
        layout,
        cu_seqlens,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
    ):
        if layout == "TND":
            _, num_query_heads, head_dim = query.shape
            batch_size = cu_seqlens.numel() - 1
            num_kv_heads = original_kv.shape[1]
            query_length = int(torch.diff(cu_seqlens).max().item())
            # The CP arm keeps each document's full prefix and its own compressed rows.
            cu_seqlens_kv = cu_seqlens if cu_seqlens_ori_kv is None else cu_seqlens_ori_kv
            original_kv_length = int(torch.diff(cu_seqlens_kv).max().item())
            sequence_lengths = torch.diff(cu_seqlens)
            if compressed_kv is not None:
                if cu_seqlens_cmp_kv is not None:
                    cu_seqlens_cmp = cu_seqlens_cmp_kv
                else:
                    compressed_lengths = torch.div(sequence_lengths, compress_ratio, rounding_mode="floor")
                    cu_seqlens_cmp = torch.nn.functional.pad(
                        compressed_lengths.cumsum(0, dtype=torch.int32),
                        (1, 0),
                        value=0,
                    )
                residual = torch.remainder(sequence_lengths, compress_ratio)
            else:
                cu_seqlens_cmp = None
                residual = 0
        else:
            cu_seqlens_kv = cu_seqlens
            batch_size, query_length, num_query_heads, head_dim = query.shape
            num_kv_heads = original_kv.shape[2]
            original_kv_length = original_kv.shape[1]
            cu_seqlens_cmp = None
            residual = original_kv_length % compress_ratio if compressed_kv is not None and compress_ratio != 1 else 0
        has_compressed_kv = compressed_kv is not None
        cu_seqlens = _as_int32_cu_seqlens(cu_seqlens)
        cu_seqlens_cmp = _as_int32_cu_seqlens(cu_seqlens_cmp)
        compressed_kv_length = (
            int(torch.diff(cu_seqlens_cmp).max().item())
            if layout == "TND" and has_compressed_kv
            else compressed_kv.shape[1]
            if has_compressed_kv
            else 0
        )
        topk = compressed_indices.shape[-1] if has_compressed_kv else 0
        metadata_kwargs = dict(
            batch_size=batch_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            topk=topk,
            has_compressed_kv=has_compressed_kv,
            compress_ratio=compress_ratio,
            query_length=query_length,
            original_kv_length=original_kv_length,
            compressed_kv_length=compressed_kv_length,
            residual=residual,
            window_size=window_size,
            device_type=query.device.type,
            device_index=query.device.index,
            layout=layout,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cmp=cu_seqlens_cmp,
            cu_seqlens_kv=cu_seqlens_kv,
        )
        metadata = _forward_metadata(**metadata_kwargs)
        result, softmax_lse = _get_cann_ops().sparse_flash_mla(
            query,
            ori_kv=original_kv,
            cmp_kv=compressed_kv,
            cmp_sparse_indices=compressed_indices,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_ori_kv=cu_seqlens_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp,
            cmp_residual_kv=_residual_tensor(
                residual,
                batch_size,
                query.device,
                has_compressed_kv and compress_ratio != 1,
            ),
            sinks=attention_sink,
            metadata=metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=compress_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=_cmp_mask_mode(has_compressed_kv),
            ori_win_left=window_size - 1,
            ori_win_right=0,
            layout_q=layout,
            layout_kv=layout,
            return_softmax_lse=True,
        )

        saved = [query, original_kv, result, softmax_lse, attention_sink]
        if has_compressed_kv:
            saved.extend((compressed_kv, compressed_indices))
        ctx.save_for_backward(*saved)
        ctx.has_compressed_kv = has_compressed_kv
        ctx.softmax_scale = softmax_scale
        ctx.compress_ratio = compress_ratio
        ctx.window_size = window_size
        ctx.residual = residual
        ctx.metadata_kwargs = metadata_kwargs
        ctx.layout = layout
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_kv = cu_seqlens_kv
        ctx.cu_seqlens_cmp = cu_seqlens_cmp
        ctx.mark_non_differentiable(softmax_lse)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        query, original_kv, result, softmax_lse, attention_sink, *optional = ctx.saved_tensors
        compressed_kv = optional[0] if ctx.has_compressed_kv else None
        compressed_indices = optional[1] if ctx.has_compressed_kv else None
        batch_size = ctx.metadata_kwargs["batch_size"]
        layout = ctx.layout
        cu_seqlens = ctx.cu_seqlens
        cu_seqlens_kv = ctx.cu_seqlens_kv
        cu_seqlens_cmp = ctx.cu_seqlens_cmp
        metadata = _backward_metadata(**ctx.metadata_kwargs)
        gradients = _get_cann_ops().sparse_flash_mla_grad(
            query,
            grad_output.contiguous(),
            result,
            softmax_lse,
            ori_kv=original_kv,
            cmp_kv=compressed_kv,
            ori_sparse_indices=None,
            cmp_sparse_indices=compressed_indices,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_ori_kv=cu_seqlens_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp,
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            cmp_residual_kv=_residual_tensor(
                ctx.residual,
                batch_size,
                query.device,
                # A2/A3 Grad requires a zero ratio-one residual; A5 requires it to remain absent.
                # Remove this workaround once A3 and A5 operator behavior is aligned.
                ctx.has_compressed_kv and (ctx.compress_ratio != 1 or _is_a2_or_a3(query.device.index)),
            ),
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=attention_sink,
            metadata=metadata,
            softmax_scale=ctx.softmax_scale,
            cmp_ratio=ctx.compress_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=_cmp_mask_mode(ctx.has_compressed_kv),
            ori_win_left=ctx.window_size - 1,
            ori_win_right=0,
            layout_q=layout,
            layout_kv=layout,
        )
        grad_query, grad_original_kv, grad_compressed_kv, grad_sinks = gradients[:4]
        return (
            grad_query,
            grad_original_kv,
            grad_compressed_kv if ctx.has_compressed_kv else None,
            None,
            grad_sinks,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _prepare_compressed_indices(indices: torch.Tensor, query: torch.Tensor):
    if indices.ndim not in (2, 3):
        raise ValueError("compressed_topk_indices must have shape [B,S,K] or [T,K]")
    prefix_shape = indices.shape[:-1]
    topk = indices.shape[-1]
    padded_topk = 512 if topk <= 512 else 1024
    if topk > padded_topk:
        raise ValueError("SparseFlashMla supports at most 1024 compressed indices")
    indices = indices.to(dtype=torch.int32)
    if topk < padded_topk:
        indices = torch.cat(
            (
                indices,
                indices.new_full((*prefix_shape, padded_topk - topk), -1),
            ),
            dim=-1,
        )
    # CANN expects [..., N, K]: [B,S,1,K] for BSND and [T,1,K] for TND.
    return indices.unsqueeze(-2).contiguous()


def npu_sparse_flash_mla(
    query: torch.Tensor,
    original_kv: torch.Tensor,
    compressed_kv: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    attention_sink: torch.Tensor,
    softmax_scale: float,
    compress_ratio: int,
    window_size: int,
    layout: str = "BSND",
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_ori_kv: torch.Tensor | None = None,
    cu_seqlens_cmp_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run SparseFlashMla in BSND or packed TND layout.

    ``cu_seqlens_ori_kv`` / ``cu_seqlens_cmp_kv`` decouple the KV prefix from the
    query segment, which packed-TND context parallelism needs: a rank owns a
    query slice but reads each document's whole prefix.
    """
    layout = layout.upper()
    if layout not in ("BSND", "TND"):
        raise ValueError(f"SparseFlashMla layout must be BSND or TND, got {layout!r}")
    if layout == "TND":
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for SparseFlashMla TND layout")
        if query.ndim != 3 or original_kv.ndim != 2:
            raise ValueError("TND expects query [T,N,D] and original_kv [T,D]")
        if compressed_kv is not None and compressed_kv.ndim != 2:
            raise ValueError("TND expects compressed_kv with shape [Tcmp,D]")
        if cu_seqlens.dtype != torch.int32 or cu_seqlens.device != query.device:
            cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.int32)
        if cu_seqlens[0].item() != 0 or cu_seqlens[-1].item() != query.shape[0]:
            raise ValueError("TND cu_seqlens must span all query tokens")
        if not torch.all(cu_seqlens[1:] > cu_seqlens[:-1]).item():
            raise ValueError("TND cu_seqlens must be strictly increasing")
        kv_seqlens = cu_seqlens if cu_seqlens_ori_kv is None else cu_seqlens_ori_kv
        if kv_seqlens.dtype != torch.int32 or kv_seqlens.device != query.device:
            kv_seqlens = kv_seqlens.to(device=query.device, dtype=torch.int32)
        if original_kv.shape[0] != int(kv_seqlens[-1].item()):
            raise ValueError("TND original KV must span every KV cu_seqlens entry")
        if compressed_kv is not None:
            if cu_seqlens_cmp_kv is not None:
                expected_compressed_tokens = int(cu_seqlens_cmp_kv[-1].item())
            else:
                expected_compressed_tokens = int(
                    torch.div(
                        torch.diff(cu_seqlens),
                        compress_ratio,
                        rounding_mode="floor",
                    )
                    .sum()
                    .item()
                )
            if compressed_kv.shape[0] != expected_compressed_tokens:
                raise ValueError(
                    "TND compressed KV length must equal the sum of per-document "
                    f"floor(length / ratio): expected {expected_compressed_tokens}, "
                    f"got {compressed_kv.shape[0]}"
                )
    else:
        if cu_seqlens is not None:
            raise ValueError("cu_seqlens must be None for BSND layout")
    compress_ratio = compress_ratio or 1
    if layout == "BSND":
        _validate_sparse_flash_mla_inputs(
            query,
            original_kv,
            compressed_kv,
            compressed_indices,
            attention_sink,
            compress_ratio,
            window_size,
        )
    if compressed_kv is not None:
        if compressed_indices is None:
            raise ValueError("compressed_topk_indices is required with compressed_key_value")
        compressed_indices = _prepare_compressed_indices(compressed_indices, query)
        compressed_kv = compressed_kv.unsqueeze(2 if layout == "BSND" else 1)
    elif compressed_indices is not None:
        raise ValueError("compressed_key_value is required with compressed_topk_indices")
    if layout == "BSND":
        original_kv = original_kv.unsqueeze(2)
    else:
        original_kv = original_kv.unsqueeze(1)
    return _SparseFlashMlaFunction.apply(
        query,
        original_kv,
        compressed_kv,
        compressed_indices,
        attention_sink.float(),
        softmax_scale,
        compress_ratio,
        window_size,
        layout,
        cu_seqlens,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
    )


@register_op("indexed_sparse_attention", "npu")
def indexed_sparse_attention_npu(
    query: torch.Tensor,
    key_value: torch.Tensor,
    attention_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    attention_mask: torch.Tensor | None = None,
    use_sparse_flash_attn: bool = False,
    compressed_key_value: torch.Tensor | None = None,
    compressed_topk_indices: torch.Tensor | None = None,
    compress_ratio: int = 1,
    window_size: int = 128,
    layout: str = "BSND",
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_ori_kv: torch.Tensor | None = None,
    cu_seqlens_cmp_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use SparseFlashMla when requested, otherwise run semantic PyTorch."""
    if use_sparse_flash_attn:
        return npu_sparse_flash_mla(
            query,
            key_value,
            compressed_key_value,
            compressed_topk_indices,
            attention_sink,
            softmax_scale,
            compress_ratio,
            window_size,
            layout,
            cu_seqlens,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
        )
    return indexed_sparse_attention_torch(
        query,
        key_value,
        attention_sink,
        topk_indices,
        softmax_scale,
        attention_mask,
        layout=layout,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["indexed_sparse_attention_npu", "npu_sparse_flash_mla"]

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""NPU implementations for differentiable indexed sparse attention."""

from functools import lru_cache
import os

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


def _residual_tensor(residual: int, batch_size: int, device, required: bool):
    if not required:
        return None
    return torch.full((batch_size,), residual, dtype=torch.int32, device=device)


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


@lru_cache(maxsize=32)
def _forward_metadata(
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
):
    device = torch.device(device_type, device_index)
    return _get_cann_ops().sparse_flash_mla_metadata(
        num_heads_q=num_query_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        max_seqlen_q=query_length,
        max_seqlen_ori_kv=original_kv_length,
        max_seqlen_cmp_kv=compressed_kv_length,
        cmp_residual_kv=_residual_tensor(
            residual,
            batch_size,
            device,
            has_compressed_kv and compress_ratio != 1,
        ),
        ori_topk=0,
        cmp_topk=topk if has_compressed_kv else 0,
        cmp_ratio=compress_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3 if has_compressed_kv else 0,
        ori_win_left=window_size - 1,
        ori_win_right=0,
        layout_q="BSND",
        layout_kv="BSND",
        has_ori_kv=True,
        has_cmp_kv=has_compressed_kv,
    )


@lru_cache(maxsize=32)
def _backward_metadata(
    batch_size,
    query_length,
    original_kv_length,
    compressed_kv_length,
    num_query_heads,
    num_kv_heads,
    head_dim,
    topk,
    has_compressed_kv,
    compress_ratio,
    residual,
    window_size,
    device_type,
    device_index,
):
    device = torch.device(device_type, device_index)
    return _get_cann_ops().sparse_flash_mla_grad_metadata(
        num_heads_q=num_query_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
        cmp_residual_kv=_residual_tensor(
            residual,
            batch_size,
            device,
            has_compressed_kv and compress_ratio != 1,
        ),
        batch_size=batch_size,
        max_seqlen_q=query_length,
        max_seqlen_ori_kv=original_kv_length,
        max_seqlen_cmp_kv=compressed_kv_length,
        ori_topk=0,
        cmp_topk=topk if has_compressed_kv else 0,
        cmp_ratio=compress_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3 if has_compressed_kv else 0,
        ori_win_left=window_size - 1,
        ori_win_right=0,
        layout_q="BSND",
        layout_kv="BSND",
        has_ori_kv=True,
        has_cmp_kv=has_compressed_kv,
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
    ):
        batch_size, query_length, num_query_heads, head_dim = query.shape
        num_kv_heads = original_kv.shape[2]
        original_kv_length = original_kv.shape[1]
        has_compressed_kv = compressed_kv is not None
        compressed_kv_length = compressed_kv.shape[1] if has_compressed_kv else 0
        topk = compressed_indices.shape[-1] if has_compressed_kv else 0
        residual = original_kv_length % compress_ratio if has_compressed_kv and compress_ratio != 1 else 0
        shape = (
            batch_size,
            query_length,
            original_kv_length,
            compressed_kv_length,
            num_query_heads,
            num_kv_heads,
            head_dim,
            topk,
            query.device.type,
            query.device.index,
        )
        metadata = _forward_metadata(
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
            query.device.type,
            query.device.index,
        )
        result, softmax_lse = _get_cann_ops().sparse_flash_mla(
            query,
            ori_kv=original_kv,
            cmp_kv=compressed_kv,
            cmp_sparse_indices=compressed_indices,
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
            cmp_mask_mode=3 if has_compressed_kv else 0,
            ori_win_left=window_size - 1,
            ori_win_right=0,
            layout_q="BSND",
            layout_kv="BSND",
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
        ctx.shape = shape
        ctx.mark_non_differentiable(softmax_lse)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        query, original_kv, result, softmax_lse, attention_sink, *optional = ctx.saved_tensors
        compressed_kv = optional[0] if ctx.has_compressed_kv else None
        compressed_indices = optional[1] if ctx.has_compressed_kv else None
        (
            batch_size,
            query_length,
            original_kv_length,
            compressed_kv_length,
            num_query_heads,
            num_kv_heads,
            head_dim,
            topk,
            device_type,
            device_index,
        ) = ctx.shape
        metadata = _backward_metadata(
            batch_size,
            query_length,
            original_kv_length,
            compressed_kv_length,
            num_query_heads,
            num_kv_heads,
            head_dim,
            topk,
            ctx.has_compressed_kv,
            ctx.compress_ratio,
            ctx.residual,
            ctx.window_size,
            device_type,
            device_index,
        )
        gradients = _get_cann_ops().sparse_flash_mla_grad(
            query,
            grad_output.contiguous(),
            result,
            softmax_lse,
            ori_kv=original_kv,
            cmp_kv=compressed_kv,
            ori_sparse_indices=None,
            cmp_sparse_indices=compressed_indices,
            cu_seqlens_q=None,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            cmp_residual_kv=_residual_tensor(
                ctx.residual,
                batch_size,
                query.device,
                ctx.has_compressed_kv and ctx.compress_ratio != 1,
            ),
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=attention_sink,
            metadata=metadata,
            softmax_scale=ctx.softmax_scale,
            cmp_ratio=ctx.compress_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3 if ctx.has_compressed_kv else 0,
            ori_win_left=ctx.window_size - 1,
            ori_win_right=0,
            layout_q="BSND",
            layout_kv="BSND",
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
        )


def _prepare_compressed_indices(indices: torch.Tensor, query: torch.Tensor):
    if indices.ndim != 3:
        raise ValueError("compressed_topk_indices must have shape [batch, query, topk]")
    batch_size, query_length = query.shape[:2]
    topk = indices.shape[-1]
    padded_topk = 512 if topk <= 512 else 1024
    if topk > padded_topk:
        raise ValueError("SparseFlashMla supports at most 1024 compressed indices")
    indices = indices.to(dtype=torch.int32)
    if topk < padded_topk:
        indices = torch.cat(
            (
                indices,
                indices.new_full((batch_size, query_length, padded_topk - topk), -1),
            ),
            dim=-1,
        )
    return indices.unsqueeze(2).contiguous()


def npu_sparse_flash_mla(
    query: torch.Tensor,
    original_kv: torch.Tensor,
    compressed_kv: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    attention_sink: torch.Tensor,
    softmax_scale: float,
    compress_ratio: int,
    window_size: int,
) -> torch.Tensor:
    """Run BSND SparseFlashMla forward and SparseFlashMlaGrad backward."""
    compress_ratio = compress_ratio or 1
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
        compressed_kv = compressed_kv.unsqueeze(2)
    elif compressed_indices is not None:
        raise ValueError("compressed_key_value is required with compressed_topk_indices")
    return _SparseFlashMlaFunction.apply(
        query,
        original_kv.unsqueeze(2),
        compressed_kv,
        compressed_indices,
        attention_sink.float(),
        softmax_scale,
        compress_ratio,
        window_size,
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
        )
    return indexed_sparse_attention_torch(
        query,
        key_value,
        attention_sink,
        topk_indices,
        softmax_scale,
        attention_mask,
    )


__all__ = ["indexed_sparse_attention_npu", "npu_sparse_flash_mla"]

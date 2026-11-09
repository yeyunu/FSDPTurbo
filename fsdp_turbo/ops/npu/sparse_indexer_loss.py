# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CANN SparseLightningIndexerKLLossGrad, independent of the attention backend."""

import logging

import torch

from fsdp_turbo.ops.cpu.sparse_indexer_loss import indexer_kl_sum_torch, sparse_indexer_gradient_torch
from fsdp_turbo.ops.npu.sparse_attention import _is_a2_or_a3
from fsdp_turbo.ops.indexer_kl_metric import current_indexer_kl_metrics, indexer_kl_metric
from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.utils.log import log_warning_once


class _IndexerGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, weights, indices, target, ratio):
        ctx.save_for_backward(query, key, weights, indices, target)
        ctx.ratio = ratio
        ctx.metrics = current_indexer_kl_metrics()
        return query.new_zeros((), dtype=torch.float32)

    @staticmethod
    def backward(ctx, scale):
        from cann_ops_transformer import ops

        q, k, w, indices, teacher = ctx.saved_tensors
        batch, length, heads, dim = q.shape
        residual = indices.new_full((batch,), length % ctx.ratio)
        attrs = dict(
            cmp_residual_k=residual,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=3,
            cmp_ratio=ctx.ratio,
        )
        metadata = ops.sparse_lightning_indexer_kl_loss_grad_metadata(
            heads,
            1,
            dim,
            batch_size=batch,
            max_seqlen_q=length,
            max_seqlen_k=k.size(1),
            topk=indices.size(-1),
            **attrs,
        )
        dq, dk, dw, index_probs = ops.sparse_lightning_indexer_kl_loss_grad(
            q,
            k,
            w,
            indices,
            teacher,
            metadata=metadata,
            **attrs,
        )
        if ctx.metrics is not None:
            if index_probs.shape == teacher.squeeze(2).shape:
                index_probs = index_probs.unsqueeze(2)
            if index_probs.shape == teacher.shape:
                metric = indexer_kl_metric(teacher, index_probs, use_fused=ctx.metrics.use_fused)
            else:
                log_warning_once(
                    logging.getLogger(__name__), "Unexpected SLIG probability shape; reporting NaN for Indexer KL."
                )
                metric = teacher.new_full((), float("nan"))
            ctx.metrics.sums.append(metric)
        return dq * scale, dk * scale, dw * scale, None, None, None


@register_op("sparse_indexer_gradient", "npu")
def sparse_indexer_gradient_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    target: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    """Dispatch supported BSND slices to SLIG, retaining device-local PyTorch for other inputs."""
    try:
        from cann_ops_transformer import ops
    except ImportError:
        ops = None
    if (
        not hasattr(ops, "sparse_lightning_indexer_kl_loss_grad")
        or not hasattr(ops, "sparse_lightning_indexer_kl_loss_grad_metadata")
        or query.shape[-2:] != (32, 128)
        or indices.size(-1) > 1024
    ):
        log_warning_once(
            logging.getLogger(__name__),
            "SLIG unavailable for this build or shape; using PyTorch.",
        )
        return sparse_indexer_gradient_torch(query, key, weights, indices, target, ratio)
    # CANN uses y * sum(p) - p; normalize the teacher to match the mean-token KL reference.
    target = torch.nn.functional.normalize(target.detach().float(), p=1, dim=-1)
    # K controls SLIG softmax visibility. Preserve the selected width, including K < 512.
    native_indices = indices.to(torch.int32).unsqueeze(2).contiguous()
    target = target.unsqueeze(2)
    return _IndexerGradient.apply(
        query.bfloat16().contiguous(),
        key.unsqueeze(2).bfloat16().contiguous(),
        weights.float().contiguous(),
        native_indices,
        target.contiguous(),
        ratio,
    )


def _supports_indexer_kl_kernel(device_index: int | None) -> bool:
    if _is_a2_or_a3(device_index):
        return True
    name = torch.accelerator.get_device_name(device_index).upper()
    return name.replace("-", "").replace("_", "").replace(" ", "").startswith("ASCEND950DT")


@register_op("indexer_kl_sum", "npu")
def indexer_kl_sum_npu(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """Use the scalar kernel on A2/A3 and the supported Ascend 950DT build."""
    if (
        _supports_indexer_kl_kernel(teacher.device.index)
        and teacher.shape[-1] <= 8192
        and (teacher.ndim == 2 or teacher.shape[0] <= 512)
    ):
        try:
            from cann_ops_transformer import lightning_indexer_kl_loss
        except ImportError:
            log_warning_once(logging.getLogger(__name__), "KL scalar kernel unavailable; using PyTorch on NPU.")
        else:
            return lightning_indexer_kl_loss(teacher, student, eps=1e-9, weight_type="logits")
    return indexer_kl_sum_torch(teacher, student)

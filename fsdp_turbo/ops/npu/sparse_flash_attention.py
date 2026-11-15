# Copyright © 2026 Huawei Technologies Co., Ltd.
"""Sparse flash attention over the indexer's selected keys, for MLA-absorb DSA.

The dense path expands the top-k indices into a [B, 1, Q_S, KV_S] boolean mask
and hands that to FlashAttentionScore. Building the mask costs more than the
attention it feeds: measured at B=1 S=8192, 209.6 ms of scatter/gather/concat/
reduce against 131 ms of FA. npu_sparse_flash_attention consumes the indices.

The operator only accepts the absorbed MLA layout, verified against the wheel:
  qk_head_dim   must equal 512 (the kv_lora_rank), so the query has to carry
                W_UK already folded in and the key is the compressed latent
  kv_head_num   must be 1, i.e. one shared latent rather than expanded per-head
                K/V -- which also keeps sparse_indices at 67 MB instead of 4.3 GB
  rope          mandatory, and its head dim must be exactly 64
  attention_mode must be 2; the schema's default of 0 is rejected
  sparse_block_size must be 1 for token-level indices; larger values return
                inf/nan
Index order within a row does not matter (checked: 2.8e-07 between a permuted
and an unpermuted call), so the indices are passed through untouched.
"""

import logging

import torch
import torch_npu

from fsdp_turbo.ops.registry import get_op, register_op
from fsdp_turbo.utils.log import log_warning_once

logger = logging.getLogger(__name__)

# Fixed by the operator, not by this model.
SFA_ROPE_DIM = 64
SFA_QK_HEAD_DIM = 512
SFA_SPARSE_BLOCK_SIZE = 1
SFA_ATTENTION_MODE = 2
# rightDownCausal: the causal triangle anchored at the bottom right.
SFA_SPARSE_MODE_CAUSAL = 3

_SFA = getattr(torch_npu, "npu_sparse_flash_attention", None)
_SFA_GRAD = getattr(torch_npu, "npu_sparse_flash_attention_grad", None)
_HAS_TORCH_NPU_SFA = _SFA is not None and _SFA_GRAD is not None

# CANN releases may expose SFA/SFAG solely through TorchNPU. Both bindings
# return softmax max/sum; preserve the same sparse kernel and autograd contract.
try:
    from cann_ops_transformer.ops.attention.sparse_flash_attention import sparse_flash_attention as _CANN_SFA
    from cann_ops_transformer.ops.attention.sparse_flash_attention_grad import sparse_flash_attention_grad as _CANN_SFA_GRAD
except ModuleNotFoundError as error:
    if error.name not in {
        "cann_ops_transformer.ops.attention.sparse_flash_attention",
        "cann_ops_transformer.ops.attention.sparse_flash_attention_grad",
    }:
        raise
    _CANN_SFA = _CANN_SFA_GRAD = None
_HAS_CANN_SFA = _CANN_SFA is not None and _CANN_SFA_GRAD is not None


def make_rope_pair(query, key, rope_head_dim, query_rope=None, key_rope=None, *, check_width=True):
    """Return the (query_rope, key_rope) pair the operator requires.

    A model with real positional Q/K channels passes them through; one without
    gets zeros, which contribute nothing to the q.k score and so leave the
    attention unchanged. Zeroing is only ever applied to the no-rope case -- a
    model that has rope of an unsupported width must fail rather than lose it.
    """
    if rope_head_dim == 0:
        return (query.new_zeros((*query.shape[:-1], SFA_ROPE_DIM)), key.new_zeros((*key.shape[:-1], SFA_ROPE_DIM)))
    if query_rope is None or key_rope is None:
        raise ValueError(f"rope_head_dim={rope_head_dim} but no rope tensors were given.")
    if check_width and (query_rope.shape[-1] != SFA_ROPE_DIM or key_rope.shape[-1] != SFA_ROPE_DIM):
        raise NotImplementedError(
            f"npu_sparse_flash_attention requires rope head dim {SFA_ROPE_DIM}, got "
            f"query {query_rope.shape[-1]} / key {key_rope.shape[-1]}."
        )
    return query_rope.contiguous(), key_rope.contiguous()


class SparseFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        sparse_indices,
        query_rope,
        key_rope,
        scale,
        actual_seq_qlen,
        actual_seq_kvlen,
        sparse_mode,
        layout,
        return_softmax_stats,
    ):
        use_cann = _HAS_CANN_SFA and (layout == "TND" or return_softmax_stats)
        operation = _SFA
        if use_cann:
            operation = _CANN_SFA
        out, softmax_max, softmax_sum, *_ = operation(
            query,
            key,
            value,
            sparse_indices=sparse_indices,
            block_table=None,
            actual_seq_lengths_query=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_kvlen,
            query_rope=query_rope,
            key_rope=key_rope,
            scale_value=float(scale),
            sparse_block_size=SFA_SPARSE_BLOCK_SIZE,
            layout_query=layout,
            layout_kv=layout,
            sparse_mode=int(sparse_mode),
            attention_mode=SFA_ATTENTION_MODE,
            return_softmax_lse=True,
        )
        ctx.save_for_backward(query, key, value, sparse_indices, query_rope, key_rope, out, softmax_max, softmax_sum)
        ctx.scale = scale
        ctx.sparse_mode = sparse_mode
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_kvlen = actual_seq_kvlen
        ctx.layout = layout
        ctx.use_cann = use_cann
        if return_softmax_stats:
            dimension = 1 if layout == "BSND" else 0
            maximum, denominator = softmax_max.squeeze(dimension), softmax_sum.squeeze(dimension)
            ctx.mark_non_differentiable(maximum, denominator)
            return out, maximum, denominator
        return out

    @staticmethod
    def backward(ctx, d_out, *_stats_grads):
        (query, key, value, sparse_indices, query_rope, key_rope, out, softmax_max, softmax_sum) = ctx.saved_tensors
        operation = _SFA_GRAD
        native_args = (float(ctx.scale), SFA_SPARSE_BLOCK_SIZE)
        native_kwargs = {}
        if ctx.use_cann:
            from cann_ops_transformer.ops.attention.sparse_flash_attention_grad import sparse_flash_attention_grad

            operation = sparse_flash_attention_grad
            native_args = ()
            native_kwargs = dict(sinks=None, scale_value=ctx.scale, sparse_block_size=SFA_SPARSE_BLOCK_SIZE)
        dq, dk, dv, dq_rope, dk_rope, *_ = operation(
            query,
            key,
            value,
            sparse_indices,
            d_out.contiguous(),
            out,
            softmax_max,
            softmax_sum,
            *native_args,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=ctx.actual_seq_qlen,
            actual_seq_kvlen=ctx.actual_seq_kvlen,
            layout=ctx.layout,
            sparse_mode=int(ctx.sparse_mode),
            attention_mode=SFA_ATTENTION_MODE,
            **native_kwargs,
        )
        # One gradient per positional argument of apply(). The rope gradients are
        # dropped for a no-rope model because its rope tensors are constants.
        return (
            dq.to(query),
            dk.to(key),
            dv.to(value),
            None,
            dq_rope.to(query_rope),
            dk_rope.to(key_rope),
            None,
            None,
            None,
            None,
            None,
            None,
        )


@register_op("sparse_flash_attention", "npu")
def _sparse_flash_attention_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    scale: float,
    rope_head_dim: int = 0,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    sparse_mode: int = SFA_SPARSE_MODE_CAUSAL,
    *,
    cu_seqlens: torch.Tensor | None = None,
    return_softmax_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention over the indexer's selected keys, with no dense mask.

    query is the absorbed query [B, S, N, kv_lora_rank]; key and value are the
    shared compressed latent [B, S, 1, kv_lora_rank]. topk_indices is the
    indexer's [B, S, topk], with -1 marking unused slots. Packed TND inputs
    and callers requesting stats use CANN SFA/SFAG.
    """
    use_cann = _HAS_CANN_SFA and (cu_seqlens is not None or return_softmax_stats)
    if not use_cann and not _HAS_TORCH_NPU_SFA:
        log_warning_once(
            logger,
            "Operator 'sparse_flash_attention' not available for device 'npu', falling back to CPU implementation",
        )
        return get_op("sparse_flash_attention", "cpu")(
            query,
            key,
            value,
            topk_indices,
            scale,
            rope_head_dim,
            query_rope,
            key_rope,
            sparse_mode,
        )
    if not use_cann and query.shape[-1] != SFA_QK_HEAD_DIM:
        raise NotImplementedError(
            f"The operator only accepts qk_head_dim={SFA_QK_HEAD_DIM} (the absorbed "
            f"MLA layout), got {query.shape[-1]}. W_UK must be folded into the query."
        )
    if key.shape[-2] != 1 or value.shape[-2] != 1:
        raise NotImplementedError(
            f"The operator only accepts kv_head_num=1, got key {key.shape[-2]} / "
            f"value {value.shape[-2]}. Pass the shared latent, not expanded K/V."
        )

    shared_kv = key is value
    query, key = query.contiguous(), key.contiguous()
    value = key if shared_kv else value.contiguous()
    query_rope, key_rope = make_rope_pair(
        query,
        key,
        rope_head_dim,
        query_rope,
        key_rope,
        check_width=not use_cann,
    )

    layout = "BSND"
    actual_seq_qlen = None
    if cu_seqlens is not None:
        layout = "TND"
        actual_seq_qlen = cu_seqlens[1:].contiguous()
    elif not use_cann:
        # The legacy torch_npu kernel needs lengths even for a dense batch.
        actual_seq_qlen = torch.full((query.shape[0],), query.shape[1], dtype=torch.int32, device=query.device)

    return SparseFlashAttentionFunction.apply(
        query,
        key,
        value,
        topk_indices.unsqueeze(-2).to(torch.int32).contiguous(),
        query_rope,
        key_rope,
        scale,
        actual_seq_qlen,
        actual_seq_qlen,
        sparse_mode,
        layout,
        return_softmax_stats,
    )


# GLM uses this export to check availability of the legacy torch_npu path.
# The registered dispatcher also supports CANN-only installations.
sparse_flash_attention_npu = _sparse_flash_attention_npu if _HAS_TORCH_NPU_SFA else None


__all__ = ["sparse_flash_attention_npu"]

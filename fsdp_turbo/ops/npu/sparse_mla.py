"""CANN training implementation of shared-KV sparse MLA."""
import torch
from fsdp_turbo.ops.registry import register_op


class _AscendCSparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, query_rope, key_rope, indices, scale, cu_seqlens):
        from cann_ops_transformer.ops.attention.sparse_flash_attention import sparse_flash_attention

        query, key, query_rope, key_rope, indices = [tensor.contiguous() for tensor in (query, key, query_rope, key_rope, indices)]
        ctx.layout = "BSND" if cu_seqlens is None else "TND"
        ctx.endpoints = None if cu_seqlens is None else cu_seqlens[1:].contiguous()
        sequence_args = {} if ctx.endpoints is None else dict(
            actual_seq_lengths_query=ctx.endpoints, actual_seq_lengths_kv=ctx.endpoints)
        output, maximum, denominator = sparse_flash_attention(
            query, key, key, indices, scale, query_rope=query_rope, key_rope=key_rope,
            sparse_block_size=1, layout_query=ctx.layout, layout_kv=ctx.layout,
            sparse_mode=3, attention_mode=2, return_softmax_lse=True, **sequence_args)
        ctx.save_for_backward(query, key, query_rope, key_rope, indices, output, maximum, denominator)
        ctx.scale = scale
        stats_dimension = 1 if cu_seqlens is None else 0
        maximum, denominator = maximum.squeeze(stats_dimension), denominator.squeeze(stats_dimension)
        ctx.mark_non_differentiable(maximum, denominator)
        return output, maximum, denominator

    @staticmethod
    def backward(ctx, gradient, _maximum_grad, _sum_grad):
        from cann_ops_transformer.ops.attention.sparse_flash_attention_grad import sparse_flash_attention_grad

        query, key, query_rope, key_rope, indices, output, maximum, denominator = ctx.saved_tensors
        sequence_args = {} if ctx.endpoints is None else dict(
            actual_seq_qlen=ctx.endpoints, actual_seq_kvlen=ctx.endpoints)
        query_grad, key_grad, value_grad, query_rope_grad, key_rope_grad, _ = sparse_flash_attention_grad(
            query, key, key, indices, gradient.contiguous(), output, maximum, denominator,
            sinks=None, scale_value=ctx.scale, sparse_block_size=1,
            query_rope=query_rope, key_rope=key_rope,
            layout=ctx.layout, sparse_mode=3, attention_mode=2, **sequence_args)
        # The latent is used as both K and V, so both derivatives belong to it.
        return query_grad, key_grad + value_grad, query_rope_grad, key_rope_grad, None, None, None


@register_op("sparse_mla", "npu")
def sparse_mla_npu(query: torch.Tensor, key: torch.Tensor, query_rope: torch.Tensor,
                   key_rope: torch.Tensor, indices: torch.Tensor, scale: float,
                   cu_seqlens: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared K=V MLA through CANN SFA/SFAG with BSND or unpadded TND inputs.

    Optional int32 ``cu_seqlens=[0,...,T]`` selects a single TND call.
    Detached FP32 statistics are [B,S,H] for BSND and [T,H] for TND.

    Requires matching SFA/SFAG kernels (ops-transformer PRs #14889/#14872
    for the optimized Ascend950 RoPE path). Scale is passed by keyword because
    SFAG has a sinks parameter before it; the dispatcher requires an explicit
    sinks=None even though the Python wrapper supplies a default.

    The index width must equal LI's requested top-k; -1 pads causal rows with
    fewer visible keys, rather than increasing that top-k to an aligned width.
    """
    return _AscendCSparseMLA.apply(query, key, query_rope, key_rope, indices, scale, cu_seqlens)

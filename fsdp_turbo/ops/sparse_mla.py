"""Compatibility API for external Kimi model views predating sparse attention unification."""
from fsdp_turbo.ops.sparse_attention import sparse_flash_attention

def sparse_mla(query, key, query_rope, key_rope, indices, scaling, cu_seqlens=None):
    return sparse_flash_attention(
        query, key, key, indices.squeeze(-2), scaling,
        rope_head_dim=query_rope.shape[-1], query_rope=query_rope, key_rope=key_rope,
        cu_seqlens=cu_seqlens, return_softmax_stats=True,
    )

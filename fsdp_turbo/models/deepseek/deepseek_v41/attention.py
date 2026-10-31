# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Sliding-window and compressed sparse attention for DeepSeek-V4.1."""

from functools import lru_cache

import torch
from torch import nn

from fsdp_turbo.ops.attention_projection import attention_output_projection
from fsdp_turbo.ops.sparse_attention import indexed_sparse_attention
from fsdp_turbo.ops.rope import partial_rope
from fsdp_turbo.utils.device import IS_CANN_OPS_TRANSFORMER_AVAILABLE

from .compressor import Compressor
from .context_parallel import compute_context_parallel_attention
from .indexer import Indexer
from .layers import ColumnParallelLinear, Linear, RMSNorm, RowParallelLinear, apply_rotary_emb, precompute_freqs_cis
from .model_args import ModelArgs
from .runtime import ContextParallelAttentionState, shared_attn
from .utils import _packed_freqs, _packed_sequence_lengths


@lru_cache(8)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int, device: torch.device):
    """Which sliding-window cache slots each query attends to; -1 marks a slot holding nothing.

    The cache is a ring of `window_size` slots. Prefill needs one row per query, each seeing its own
    causal window. A decode step has a single query that sees the whole ring, listed oldest first.
    Order within a row does not matter to `sparse_attn`, which handles every slot independently.
    """
    if start_pos == 0:
        end = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), device=device)
        idxs = torch.where(idxs > end, -1, idxs)  # before the sequence started
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size, device=device), torch.arange(oldest, device=device)])
        idxs = torch.where(idxs > start_pos, -1, idxs)  # ring still filling
    # sparse_attn needs real [b, m, topk] int32 memory, hence the materializing expand
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


class Attention(nn.Module):
    """Latent attention over two KV sources at once, concatenated into one `sparse_attn` call: a
    sliding window of raw KV, plus -- when compress_ratio > 0 -- `index_topk` compressed positions
    reaching further back. Q and the output projection are both low-rank, the latter grouped.

    compress_ratio > 0 does not mean the layer compresses its own KV: only kv_source_layers do,
    the rest read that same cache.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.use_triton_inplace_rope = args.use_triton_inplace_rope
        # Only fuse when the operator is importable, so a config that enables it
        # on a machine without cann_ops_transformer degrades to the reference path.
        self.use_fused_partial_rotary = args.use_fused_partial_rotary and IS_CANN_OPS_TRANSFORMER_AVAILABLE
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.use_sparse_flash_attn = args.use_sparse_flash_attn
        self.use_deepgemm_wo_a = args.use_deepgemm_wo_a
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups, self.n_groups * args.o_lora_rank, dtype=torch.bfloat16
        )
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim**-0.5

        is_backbone = layer_id < args.n_layers
        self.is_kv_source = is_backbone and layer_id in args.kv_source_layers
        self.is_index_source = is_backbone and layer_id in args.index_source_layers
        self.compressor: Compressor | None = None
        self.indexer: Indexer | None = None
        if self.is_kv_source:
            self.compressor = Compressor(args, layer_id)
        if self.is_index_source:
            self.indexer = Indexer(args, layer_id)

        self.window_kv_cache: torch.Tensor
        self.register_buffer(
            "window_kv_cache", torch.zeros(args.max_batch_size, args.window_size, self.head_dim), persistent=False
        )
        if self.is_kv_source:
            self.compress_kv_cache: torch.Tensor
            self.register_buffer(
                "compress_kv_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len // self.compress_ratio, self.head_dim),
                persistent=False,
            )
        if self.compress_ratio:
            original_seq_len, rope_theta = (args.original_seq_len, args.compress_rope_theta)
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.freqs_cis: torch.Tensor
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _window_kv(self, x, freqs_cis, start_pos):
        """Build this layer's sliding-window K and its visible positions."""
        bsz, seqlen, _ = x.size()
        win = self.window_size
        kv = self.kv_norm(self.wkv(x))
        kv = apply_rotary_emb(kv, freqs_cis, use_triton_inplace_rope=self.use_triton_inplace_rope)
        if start_pos == 0:  # prefill: attend over this chunk, seeding the ring buffer for decode
            if not self.training:
                if seqlen <= win:
                    self.window_kv_cache[:bsz, :seqlen] = kv
                else:
                    cutoff = seqlen % win
                    (self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff]) = kv[:, -win:].split(
                        [win - cutoff, cutoff], dim=1
                    )
            window_kv = kv
        else:  # decode: one token into the ring buffer, attend over the whole window
            self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = self.window_kv_cache[:bsz]
        return window_kv, get_window_topk_idxs(win, bsz, seqlen, start_pos, x.device)

    def _compress_topk_idxs(
        self, x, qr, latent, start_pos, offset, compress_len, *, attention_mask=None, cu_seqlens=None
    ):
        """Which compressed positions each query attends to. Index sources run their own indexer;
        the layers in between reuse the result their source published.
        """
        if not self.is_index_source:
            return shared_attn.topk_idxs

        bsz, seqlen, _ = x.size()
        if compress_len == 0:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            assert self.indexer is not None
            if self.indexer.freqs_cis is None:
                self.indexer.freqs_cis = self.freqs_cis
            idxs = self.indexer(x, qr, latent, start_pos, offset, attention_mask=attention_mask, cu_seqlens=cu_seqlens)
        shared_attn.publish("topk_idxs", idxs)
        return idxs

    def _compress_kv(self, x, qr, start_pos, offset, *, attention_mask=None, cu_seqlens=None):
        """The shared compressed KV and the compressed positions every query may attend to. This
        layer compresses its own KV only when it is a source; otherwise it just reads the cache.
        """
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        packed_lengths = _packed_sequence_lengths(seqlen, cu_seqlens) if start_pos == 0 else None
        compress_len = (
            sum(length // ratio for length in packed_lengths)
            if packed_lengths is not None
            else (start_pos + seqlen) // ratio
        )
        if compress_len == 0 and self.training and start_pos == 0:
            # No complete block exists yet, including very short training inputs.
            compress_kv = shared_attn.publish("compress_kv", x.new_empty(bsz, 0, self.head_dim))
            idxs = self._compress_topk_idxs(
                x, qr, None, start_pos, offset, 0, attention_mask=attention_mask, cu_seqlens=cu_seqlens
            )
            return compress_kv, idxs
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos, cu_seqlens=cu_seqlens)
            if not (self.training and start_pos == 0):
                shared_attn.publish("compress_kv", self.compress_kv_cache)
        # the indexer needs the latent before RoPE, so it runs before the cache is written
        idxs = self._compress_topk_idxs(
            x, qr, latent, start_pos, offset, compress_len, attention_mask=attention_mask, cu_seqlens=cu_seqlens
        )
        if latent is not None:
            # a latent stands for the first token of its group, so group j takes position j * ratio
            if packed_lengths is not None:
                freqs = _packed_freqs(self.freqs_cis, packed_lengths, ratio)
            else:
                freqs = (
                    self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                    if start_pos == 0
                    else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
                )
            latent = apply_rotary_emb(latent, freqs, use_triton_inplace_rope=self.use_triton_inplace_rope)
            if self.training and start_pos == 0:
                # Use the write itself: during a replay the slot holds the captured state, so
                # re-reading it would return a different (previous-group) tensor.
                compress_kv = shared_attn.publish("compress_kv", latent)
            else:
                self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent
                compress_kv = shared_attn.compress_kv
        else:
            compress_kv = shared_attn.compress_kv
        # pylint: disable-next=unsubscriptable-object
        return compress_kv[:bsz, :compress_len], idxs

    def _rope_q_o(self, x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
        """RoPE for the large per-layer q/o tensors.

        These two calls dominate: at the V4.1-Flash shapes they carry 128 MiB of
        fp32/complex temporaries each, and their backward costs 5x the fused
        kernel. The operator locates the trailing rotary block itself, so the
        whole tensor is passed, exactly like the reference `apply_rotary_emb`.
        """
        if self.use_fused_partial_rotary:
            return partial_rope(x, freqs_cis.real, freqs_cis.imag, inverse=inverse)
        return apply_rotary_emb(x, freqs_cis, inverse, use_triton_inplace_rope=self.use_triton_inplace_rope)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        *,
        attention_mask: torch.Tensor | None = None,
        sfa_layout: str = "BSND",
        cu_seqlens: torch.Tensor | None = None,
        cp_group: torch.distributed.ProcessGroup | None = None,
        cp_state: ContextParallelAttentionState | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ContextParallelAttentionState]:
        """Attend to local queries, using globally gathered KV when CP is enabled."""
        if cp_group is not None:
            return compute_context_parallel_attention(self, x, start_pos, cp_group, cp_state)
        bsz, seqlen, _ = x.size()
        packed_cu_seqlens = cu_seqlens if sfa_layout == "TND" else None
        packed_lengths = _packed_sequence_lengths(seqlen, packed_cu_seqlens) if start_pos == 0 else None
        freqs_cis = (
            _packed_freqs(self.freqs_cis, packed_lengths)
            if packed_lengths is not None
            else self.freqs_cis[start_pos : start_pos + seqlen]
        )

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = self._rope_q_o(q, freqs_cis)

        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
        compress_kv = None
        compress_idxs = None
        if self.compress_ratio:
            index_offset = 0 if self.use_sparse_flash_attn else kv.size(1)
            compress_kv, compress_idxs = self._compress_kv(
                x, qr, start_pos, index_offset, attention_mask=attention_mask, cu_seqlens=packed_cu_seqlens
            )
            if self.use_sparse_flash_attn and compress_kv.size(1) == 0:
                compress_kv = None
                compress_idxs = None

        if self.use_sparse_flash_attn:
            output_shape = q.shape
            if sfa_layout == "TND":
                # Match MindSpeed-MM: [1,S,N,D]/[1,S,D] -> [T,N,D]/[T,D]
                # immediately before the fused operator, then restore its output.
                q = q.reshape(-1, q.shape[-2], q.shape[-1]).contiguous()
                kv = kv.reshape(-1, kv.shape[-1]).contiguous()
                if compress_kv is not None:
                    compress_kv = compress_kv.reshape(-1, compress_kv.shape[-1]).contiguous()
                    compress_idxs = compress_idxs.reshape(-1, compress_idxs.shape[-1]).contiguous()
            o = indexed_sparse_attention(
                q,
                kv,
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
                attention_mask=attention_mask,
                use_sparse_flash_attn=True,
                compressed_key_value=compress_kv,
                compressed_topk_indices=compress_idxs,
                compress_ratio=self.compress_ratio or 1,
                window_size=self.window_size,
                layout=sfa_layout,
                cu_seqlens=cu_seqlens,
            )
            if sfa_layout == "TND":
                o = o.reshape(output_shape)
        else:
            if compress_kv is not None:
                kv = torch.cat([kv, compress_kv], dim=1)
                topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
            o = indexed_sparse_attention(
                q, kv, self.attn_sink, topk_idxs, self.softmax_scale, attention_mask=attention_mask
            )
        o = o.clone()
        o = self._rope_q_o(o, freqs_cis, True)

        # wo_a is block-diagonal over groups (each projects only its own heads), hence einsum not
        # Linear. convert.py dequantizes it to bf16; an fp8 grouped GEMM would halve the memory.
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = attention_output_projection(o, wo_a, use_deepgemm=self.use_deepgemm_wo_a)
        x = self.wo_b(o.flatten(2))
        return x

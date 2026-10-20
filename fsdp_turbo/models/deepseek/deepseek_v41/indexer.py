"""Sparse compressed-KV indexing and candidate selection for DeepSeek-V4.1."""

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.lightning_indexer import fused_lightning_indexer

from .layers import ColumnParallelLinear, Linear, RMSNorm, apply_rotary_emb
from .model_args import ModelArgs
from .runtime import shared_attn
from .utils import _packed_sequence_lengths, _sequence_slices


def _compute_index_score(query: torch.Tensor, index_key: torch.Tensor, head_weights: torch.Tensor) -> torch.Tensor:
    """Compute index scores in FP32 so top-k decisions are numerically stable."""
    score_per_head = torch.einsum("bshd,btd->bsht", query.float(), index_key.float())
    return (score_per_head.relu_() * head_weights.float().unsqueeze(-1)).sum(dim=2)


class Indexer(torch.nn.Module):
    """Keeps the `index_topk` best compressed positions per query.

    A small side attention: fp4 query heads against one shared key per compressed position, scores
    rectified then combined by `weights_proj`. With a candidate source this is the second of two
    levels; `select_candidate_blocks` is the first.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        # the index keys are derived from the compressor's latent, so only a layer that compresses
        # its own KV can produce them; every other indexer reads them from that layer's cache
        self.owns_k = layer_id in args.kv_source_layers
        self.compress_ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.use_fused_lightning_indexer = args.use_fused_lightning_indexer
        self.lightning_indexer_page_size = args.lightning_indexer_page_size
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.softmax_scale = self.index_head_dim**-0.5
        if self.use_fused_lightning_indexer:
            if (self.n_heads, self.index_head_dim) != (32, 128):
                raise ValueError("Fused DS41 indexers require index_n_heads=32 and index_head_dim=128")
            if not 1 <= self.index_topk <= 8192:
                raise ValueError("Fused DS41 indexers require index_topk in [1,8192]")
            if not 1 <= self.compress_ratio <= 128:
                raise ValueError("Fused DS41 indexers require index-source compress ratios in [1,128]")
            if (self.is_candidate_source or self.uses_candidates) and (
                self.candidate_topk_blocks,
                self.candidate_block_size,
            ) != (2048, 8):
                raise ValueError(
                    "Fused DS41 candidate generation requires candidate_topk_blocks=2048 and candidate_block_size=8"
                )
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.index_head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.freqs_cis: torch.Tensor | None = None
        if self.owns_k:
            self.wk = Linear(args.head_dim, self.index_head_dim, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(self.index_head_dim, args.norm_eps)
            self.k_cache: torch.Tensor
            self.register_buffer(
                "k_cache",
                torch.zeros(args.max_batch_size, args.max_seq_len // self.compress_ratio, args.index_head_dim),
                persistent=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor,
        start_pos: int,
        offset: int,
        *,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ):
        """`latent` is this layer's RoPE-free compressed latent, None when this layer does not
        compress or when its current group is still incomplete. An index-key owner turns it into
        index keys here, which has to happen before Attention overwrites that same storage with
        the RoPE'd, quantized values.
        """
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

        packed_lengths = _packed_sequence_lengths(seqlen, cu_seqlens) if start_pos == 0 else None
        if packed_lengths is not None:
            compressed_lengths = [length // ratio for length in packed_lengths]
            index_parts = []
            key_parts = []
            candidate_parts = []
            full_index_k = shared_attn.index_k
            packed_slices = zip(_sequence_slices(packed_lengths), _sequence_slices(compressed_lengths))
            for document_index, (token_slice, compressed_slice) in enumerate(packed_slices):
                token_start, token_end = token_slice
                compressed_start, compressed_end = compressed_slice
                part_latent = None if latent is None else latent[:, compressed_start:compressed_end]
                if not self.owns_k:
                    shared_attn.index_k = full_index_k[:, compressed_start:compressed_end]
                if self.uses_candidates:
                    if shared_attn.packed_candidates is None:
                        raise RuntimeError("Packed candidate masks were not published by the source layer.")
                    shared_attn.candidates = shared_attn.packed_candidates[document_index]
                part_indices = self(
                    x[:, token_start:token_end],
                    qr[:, token_start:token_end],
                    part_latent,
                    0,
                    offset,
                    attention_mask=attention_mask,
                )
                if part_indices.shape[-1] < self.index_topk:
                    part_indices = F.pad(part_indices, (0, self.index_topk - part_indices.shape[-1]), value=-1)
                index_parts.append(part_indices)
                if self.is_candidate_source:
                    candidate_parts.append(shared_attn.candidates)
                if self.owns_k:
                    key_parts.append(shared_attn.index_k[:, : compressed_end - compressed_start])
            if self.owns_k:
                shared_attn.index_k = torch.cat(key_parts, dim=1)
            else:
                shared_attn.index_k = full_index_k
            if self.is_candidate_source:
                shared_attn.packed_candidates = candidate_parts
            shared_attn.candidates = None
            return torch.cat(index_parts, dim=1)

        # latent is None while a group is still filling up, so there is nothing to publish yet
        if self.owns_k and latent is not None:
            # a latent stands for the first token of its group, so group j takes position j * ratio
            freqs = (
                self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            k = self.k_norm(self.wk(latent))
            apply_rotary_emb(k[..., -rd:], freqs)
            if self.training and start_pos == 0:
                shared_attn.index_k = k
            else:
                self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
                shared_attn.index_k = self.k_cache

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        index_k = shared_attn.index_k[:bsz, : end_pos // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)

        if self.use_fused_lightning_indexer:
            if start_pos != 0:
                raise NotImplementedError(
                    "Fused DS41 lightning indexers are currently integrated for full-sequence prefill/training only"
                )
            candidate_indices = candidate_lengths = None
            if self.uses_candidates:
                candidate_indices = shared_attn.candidate_block_indices
                candidate_lengths = shared_attn.candidate_block_length
                if candidate_indices is None or candidate_lengths is None:
                    raise RuntimeError(
                        "Sparse lightning indexer candidate inputs were not published by the candidate source layer"
                    )
            indexer_kwargs = dict(
                topk=self.index_topk,
                compress_ratio=ratio,
                index_offset=offset,
                attention_mask=attention_mask,
                candidate_topk_blocks=(self.candidate_topk_blocks if self.is_candidate_source else -1),
                candidate_block_size=(
                    self.candidate_block_size if self.is_candidate_source or self.uses_candidates else -1
                ),
                candidate_block_indices=candidate_indices,
                candidate_block_length=candidate_lengths,
                page_size=self.lightning_indexer_page_size,
            )
            idxs, new_candidates, new_candidate_lengths = fused_lightning_indexer(q, index_k, weights, **indexer_kwargs)
            if self.is_candidate_source:
                shared_attn.candidate_block_indices = new_candidates
                shared_attn.candidate_block_length = new_candidate_lengths
            return idxs

        index_score = _compute_index_score(q, index_k, weights)
        # how many compressed positions each query can see: a block becomes visible once the query
        # has passed its last token. One query per decode step, so there it is just a number.
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio

        if self.is_candidate_source:
            shared_attn.candidates = select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            # level two: score with our own weights, but only inside the source's candidate blocks
            index_score = index_score.masked_fill(~shared_attn.candidates, -torch.inf)

        # top-k by score, re-sorted into position order; unreachable -> -1, rest shifted by offset
        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()


def select_candidate_blocks(
    logits: torch.Tensor, compress_lens: torch.Tensor | int, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the `topk_blocks` highest-scoring blocks per query.

    `logits` is [..., n_positions] with positions the query cannot reach already at -inf, which is
    what makes a block score of -inf mean "not reachable yet". `compress_lens` is a plain int during
    decode, or broadcasts against logits' leading dims during prefill. Returns a bool mask shaped
    like `logits`, so the layers consuming it just mask and never think about blocks again.
    """
    width = logits.size(-1)
    # score each block by its best position; -inf pads the last one out to block_size
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # the block with this query's newest position is only partly filled, so pin it in: it holds the
    # most recent tokens but could otherwise be outscored by an older, full block
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # fewer reachable blocks than topk_blocks means leftover picks came back -inf: drop them
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]

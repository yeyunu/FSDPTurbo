"""Compressed KV projection and pooling for DeepSeek-V4.1 attention."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from fsdp_turbo.ops.compressor import fused_compressor

from .layers import Linear, RMSNorm
from .model_args import ModelArgs
from .utils import _packed_sequence_lengths, _sequence_slices


class Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one KV latent with a learned softmax gate.

    Returns the latent before RoPE, or None while a group is still filling up -- so during decode it
    only yields every `compress_ratio` steps, holding the partial group in `kv_state`/`score_state`.
    Pre-RoPE is deliberate: the indexer needs the unrotated form, so Attention rotates afterwards.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        compress_ratio = args.compress_ratios[layer_id]
        head_dim = args.head_dim
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.use_fused_compressor = args.use_fused_compressor and compress_ratio > 1
        self.norm = RMSNorm(head_dim, args.norm_eps)
        # ratio 1 is a plain projection, so it stays in the checkpoint's bf16; the softmax pooling
        # above ratio 1 runs in fp32, so those weights are promoted to fp32 to match
        self.wkv = Linear(args.dim, head_dim, dtype=torch.float32 if compress_ratio > 1 else torch.bfloat16)
        if compress_ratio > 1:
            self.wgate = Linear(args.dim, head_dim, dtype=torch.float32)
            if self.use_fused_compressor:
                state_block_size = min(args.max_seq_len, 1024)
                state_blocks_per_sequence = math.ceil(args.max_seq_len / state_block_size)
                self.fused_state_block_size = state_block_size
                self.register_buffer(
                    "fused_state_cache",
                    torch.zeros(
                        1 + args.max_batch_size * state_blocks_per_sequence,
                        state_block_size,
                        2 * head_dim,
                        dtype=torch.float32,
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    "fused_state_block_table",
                    torch.arange(1, 1 + args.max_batch_size * state_blocks_per_sequence, dtype=torch.int32).reshape(
                        args.max_batch_size, state_blocks_per_sequence
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    "fused_ape", torch.zeros(compress_ratio, head_dim, dtype=torch.float32), persistent=False
                )
            else:
                # Tail of an incomplete group, carried across decode steps.
                state_shape = (args.max_batch_size, compress_ratio, head_dim)
                self.register_buffer("kv_state", torch.zeros(state_shape, dtype=torch.float32), persistent=False)
                self.register_buffer(
                    "score_state", torch.full(state_shape, -torch.inf, dtype=torch.float32), persistent=False
                )

    def forward(
        self, x: torch.Tensor, start_pos: int, *, cu_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype
        packed_lengths = _packed_sequence_lengths(seqlen, cu_seqlens) if start_pos == 0 else None
        if packed_lengths is not None:
            compressed_parts = [self(x[:, start:end], 0) for start, end in _sequence_slices(packed_lengths)]
            if any(part is None for part in compressed_parts):
                raise ValueError(
                    f"Every packed TND document must contain at least one complete compression group for ratio {ratio}."
                )
            return torch.cat(compressed_parts, dim=1)
        if ratio == 1:  # one token per group: nothing to pool, so no gate and no fp32
            return self.norm(self.wkv(x))

        if self.use_fused_compressor:
            positions = torch.full((bsz,), start_pos, dtype=torch.int32, device=x.device)
            used_state_blocks = math.ceil((start_pos + seqlen) / self.fused_state_block_size)
            state_block_table = self.fused_state_block_table[:bsz, :used_state_blocks].contiguous()
            compressed = fused_compressor(
                x,
                self.wkv.weight.to(dtype),
                self.wgate.weight.to(dtype),
                self.fused_state_cache,
                state_block_table,
                self.fused_ape,
                compress_ratio=ratio,
                start_pos=positions,
            )
            if start_pos == 0:
                complete_groups = seqlen // ratio
                if complete_groups == 0:
                    return None
                compressed = compressed[:, :complete_groups]
            elif (start_pos + seqlen) % ratio != 0:
                return None
            return self.norm(compressed.to(dtype))

        # The reference promotes ratio>1 compressor projections and their
        # inputs to FP32 before GEMM. FSDP stores all managed parameters in a
        # uniform BF16 dtype, so recover the reference compute precision with
        # differentiable local casts instead of running these GEMMs in BF16.
        x = x.float()
        kv = F.linear(x, self.wkv.weight.float())
        score = F.linear(x, self.wgate.weight.float())
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:  # trailing partial group waits in the state
                kv, trailing_kv = kv.split([cutoff, remainder], dim=1)
                score, trailing_score = score.split([cutoff, remainder], dim=1)
                if not self.training:
                    self.kv_state[:bsz, :remainder] = trailing_kv
                    self.score_state[:bsz, :remainder] = trailing_score
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:  # one token per step: fill a slot, and pool only when the group just completed
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        return self.norm(kv.to(dtype))

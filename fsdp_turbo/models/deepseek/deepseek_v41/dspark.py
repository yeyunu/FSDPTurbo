"""DSpark draft blocks, heads and sampling for DeepSeek-V4.1."""

# DSpark intentionally specializes the base forward signatures.
# pylint: disable=arguments-differ,arguments-renamed

from functools import lru_cache

import torch
from torch import nn

from fsdp_turbo.ops.sparse_attention import indexed_sparse_attention

from .attention import Attention
from .layers import Block, Linear, ParallelEmbedding, ParallelHead, RMSNorm, apply_rotary_emb
from .model_args import ModelArgs


@lru_cache(1)
def get_dspark_topk_idxs(window_size: int, bsz: int, block_size: int, start_pos: int):
    assert start_pos > 0
    matrix = torch.cat([torch.arange(min(window_size, start_pos + 1)), window_size + torch.arange(block_size)])
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


class DSparkAttention(Attention):
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        main_x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        sfa_layout: str = "BSND",
        cu_seqlens: torch.Tensor | None = None,
    ):
        assert self.compress_ratio == 0
        bsz, seqlen, _ = main_x.size()
        win = self.window_size

        main_freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        main_kv = self.kv_norm(self.wkv(main_x))
        main_kv = apply_rotary_emb(main_kv, main_freqs_cis, use_triton_inplace_rope=self.use_triton_inplace_rope)
        if start_pos == 0:
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = main_kv
            else:
                cutoff = seqlen % win
                (self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff]) = main_kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            return x

        bsz, block_size, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos + seqlen : start_pos + seqlen + block_size]

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = apply_rotary_emb(q, freqs_cis, use_triton_inplace_rope=self.use_triton_inplace_rope)
        kv = self.kv_norm(self.wkv(x))
        kv = apply_rotary_emb(kv, freqs_cis, use_triton_inplace_rope=self.use_triton_inplace_rope)
        topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos)
        self.window_kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)
        kv = torch.cat([self.window_kv_cache[:bsz], kv], dim=1)
        o = indexed_sparse_attention(
            q, kv, self.attn_sink, topk_idxs, self.softmax_scale, attention_mask=attention_mask
        )
        o = apply_rotary_emb(o, freqs_cis, True, use_triton_inplace_rope=self.use_triton_inplace_rope)

        o = o.view(bsz, block_size, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, dspark_markov_rank: int):
        super().__init__()
        self.embed = ParallelEmbedding(vocab_size, dspark_markov_rank)
        self.head = ParallelHead(vocab_size, dspark_markov_rank)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed(token_ids)
        logits = self.head(embed, full_logits=True)
        return logits, embed


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # proj in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for fp32 confidence score.
        self.proj = Linear(input_dim, 1, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor):
        hidden = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(hidden.float()).squeeze(-1)


class DSparkBlock(Block):
    """DSpark stage stored under the mtp.* checkpoint namespace."""

    attention_cls = DSparkAttention

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        stage_id = layer_id - args.n_layers
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        self.temperature = args.temperature
        if stage_id == 0:
            assert len(args.dspark_target_layer_ids) > 0, "DSpark needs target layers"
            self.main_proj = Linear(args.dim * len(args.dspark_target_layer_ids), args.dim)
            self.main_norm = RMSNorm(args.dim, args.norm_eps)
        if stage_id == args.n_mtp_layers - 1:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(args.vocab_size, args.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(args.dim + args.dspark_markov_rank)
        self.embed: ParallelEmbedding | None = None
        self.head: ParallelHead | None = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        pre_mix: torch.Tensor,
        main_x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        sfa_layout: str = "BSND",
        cu_seqlens: torch.Tensor | None = None,
    ):
        if start_pos == 0:
            self.attn(
                x, start_pos, main_x, attention_mask=attention_mask, sfa_layout=sfa_layout, cu_seqlens=cu_seqlens
            )  # prefill only seeds the window KV cache
            return x, pre_mix
        return super().forward(
            x,
            start_pos,
            pre_mix,
            None,
            main_x,
            attention_mask=attention_mask,
            sfa_layout=sfa_layout,
            cu_seqlens=cu_seqlens,
        )  # drafts are text: no VL bias

    def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor):
        assert self.embed is not None
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_input_ids = input_ids.new_full([input_ids.size(0), self.block_size], self.noise_token_id)
        draft_input_ids[:, 0] = input_ids
        x = self.embed(draft_input_ids)  # pylint: disable=not-callable
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x

    def forward_head(
        self, x: torch.Tensor, pre_mix: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.head is not None
        x = self.hc_pre(x, pre_mix)
        logits = self.head(self.norm(x), full_logits=True)  # pylint: disable=not-callable
        output_ids = input_ids.new_empty(input_ids.size(0), self.block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        for i in range(self.block_size):
            logits_bias, markov_embed = self.markov_head(output_ids[:, i])
            logits[:, i].add_(logits_bias)
            markov_embeds.append(markov_embed)
            output_ids[:, i + 1] = sample(logits[:, i], self.temperature)
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = self.confidence_head(x, markov_embed)
        return output_ids, logits, confidence


def sample(logits, temperature: float = 1.0):
    """Gumbel-max trick: equivalent to multinomial sampling but faster on GPU,
    since it avoids the GPU-to-CPU sync in torch.multinomial.
    """
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

"""Base layers, MoE and hyper-connection decoder layers for DeepSeek-V4.1."""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from fsdp_turbo.ops.hf32_linear import hf32_linear
from fsdp_turbo.ops.mhc import fused_mhc_post, fused_mhc_pre_sinkhorn_premix
from fsdp_turbo.ops.rms_norm import rms_norm
from fsdp_turbo.ops.rope import inplace_rope
from fsdp_turbo.ops.sinkhorn import hc_split_sinkhorn

from .activation import limited_swiglu, silu_mul
from .engram import Engram, EngramLayout
from .experts import DeepSeekV41Experts
from .model_args import ModelArgs
from .runtime import ContextParallelAttentionState
from .utils import set_dtype


class ParallelEmbedding(nn.Module):
    """Full-vocabulary embedding; FSDPTurbo owns distributed sharding."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Apply the BF16 training GEMM."""
    return F.linear(x, weight, bias)


class Linear(nn.Linear):
    """Linear layer used by the BF16-only training model."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or torch.bfloat16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """Named linear layer for FSDPTurbo's column-parallel plans."""


class RowParallelLinear(Linear):
    """Linear layer retaining separate bias addition; FSDPTurbo owns parallelism."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        if x.device.type == "npu":
            return rms_norm(x, self.weight, self.eps)
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Rotary frequencies as complex exponentials, one row per position.

    With original_seq_len > 0 this applies YaRN: dimensions whose wavelength already fits inside the
    training context keep their frequency, those far beyond it are divided by `factor`, and the
    `beta_fast`..`beta_slow` band in between is faded across with a linear ramp.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        # the dim whose wavelength completes `rotations` turns over the training context
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
    *,
    use_triton_inplace_rope: bool = False,
) -> torch.Tensor:
    """Rotate the trailing rotary block of `x` in place. Accepts [b, s, d] and [b, s, h, d];
    `inverse` conjugates the rotation, which is how the attention output gets the query's rotation
    removed again so the cache can stay in one shared rotated form.
    """
    return inplace_rope(
        x,
        freqs_cis.real,
        freqs_cis.imag,
        interleaved=True,
        inverse=inverse,
        inplace=True,
        device_type="cpu" if not use_triton_inplace_rope else x.device.type,
    )


class ParallelHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        # FSDP may normalize parameter storage; projection precision is independent.
        self.compute_dtype = torch.float32
        self.use_hf32 = False
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, full_logits=False):
        """x: [b, s, d]. Generation only needs the last position, so that is the default."""
        if not full_logits:
            x = x[:, -1]
        if self.use_hf32:
            logits = hf32_linear(x, self.weight)
        else:
            logits = F.linear(x.to(self.compute_dtype), self.weight.to(self.compute_dtype))
        return logits


class Gate(nn.Module):
    """MoE gating. The correction bias steers expert selection only; the routing weights come from the
    unbiased scores. Image-span tokens use a separate bias (training `noaux_tc_for_vl`).
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        n_routed_experts, n_activated_experts = args.get_moe_config(layer_id)
        self.dim = args.dim
        self.topk = n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
        self.bias_vl = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32)) if args.vision_enabled else None

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [n, dim]; image_mask: [n] bool, True for tokens inside an image span."""
        scores = linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        # the bias picks experts but does not scale them: weights come from the raw scores
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights /= weights.sum(dim=-1, keepdim=True) + 1e-20  # not norm_eps, matches training
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """One SwiGLU FFN. The clamps come straight from training, where they keep fp8/fp4 activations in
    range: the up branch is clamped on both sides, the gate branch only from above.
    """

    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit=0.0, use_fused_clipped_swiglu=False):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)
        self.w2 = Linear(inter_dim, dim, dtype=dtype)
        self.w3 = Linear(dim, inter_dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit
        self.use_fused_clipped_swiglu = use_fused_clipped_swiglu

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.use_fused_clipped_swiglu and self.swiglu_limit > 0:
            x = limited_swiglu(torch.cat((gate, up), dim=-1), self.swiglu_limit, fused=True)
        else:
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            x = silu_mul(gate, up)
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        n_routed_experts, n_activated_experts = args.get_moe_config(layer_id)
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.gate = Gate(layer_id, args)
        self.experts = DeepSeekV41Experts(
            num_experts=n_routed_experts,
            hidden_size=args.dim,
            intermediate_size=args.moe_inter_dim,
            dtype=None,
            swiglu_limit=args.swiglu_limit,
            use_fused_clipped_swiglu=args.use_fused_clipped_swiglu,
        )
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(
            args.dim,
            args.moe_inter_dim,
            swiglu_limit=args.swiglu_limit,
            use_fused_clipped_swiglu=args.use_fused_clipped_swiglu,
        )

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        routed_output = self.experts(x, indices, weights)
        # Match the reference accumulation: routed experts are accumulated in
        # FP32, the shared expert is added in FP32, and only the complete MoE
        # result is converted back to the residual-stream dtype.
        output = routed_output.float() + self.shared_experts(x).float()
        return output.type_as(x).view(shape)


class Block(nn.Module):
    """A block whose residual stream is `hc_mult` parallel copies (Hyper-Connections).

    Attention and FFN each sit between `hc_pre` (collapse the copies into one sublayer input) and
    `hc_post` (expand back out, mixing the residual in through `comb`). `hc_mixes` derives all three
    coefficient sets from the stream itself, `comb` made doubly stochastic by Sinkhorn.

    The coefficients a sublayer computes are used by the *next* one -- see `forward`.
    """

    attention_cls: type[nn.Module] | None = None

    def __init__(self, layer_id: int, args: ModelArgs, engram_layout: EngramLayout | None = None):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        attention_cls = self.attention_cls
        if attention_cls is None:
            # Attention imports the base layers; resolve the default after module initialization.
            from .attention import Attention as attention_cls  # pylint: disable=cyclic-import
        self.attn = attention_cls(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.engram = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            if args.use_fused_engram:
                embedding_dim = (engram_layout.max_ngram_size - 1) * engram_layout.n_heads * engram_layout.head_dim
                if args.hc_mult != 4:
                    raise ValueError("Fused Engram requires hc_mult=4.")
                if args.dim not in (1024, 1280, 1536, 2048, 2560, 5120):
                    raise ValueError(f"Fused Engram has no TilingKey specialization for dim={args.dim}.")
                if embedding_dim % 128 != 0:
                    raise ValueError(
                        f"Fused Engram requires the flattened memory dim to be a multiple of 128, got {embedding_dim}."
                    )
                from .fused_engram import FusedEngram

                self.engram = FusedEngram(args, layer_id, engram_layout)
            else:
                self.engram = Engram(args, layer_id, engram_layout)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.use_fused_mhc = args.use_fused_mhc
        if self.use_fused_mhc and (self.hc_mult != 4 or self.hc_sinkhorn_iters != 20):
            raise ValueError("Fused MHC requires hc_mult=4 and hc_sinkhorn_iters=20")
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_mixes(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """x: [b,s,hc,d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc]. Returns the
        pre / post / comb coefficients, split out of one projection of the flattened stream.
        """
        # normalized over the whole flattened hc*d stream, one statistic per token
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn.float()) * rsqrt
        return hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
            use_eager=True,
        )

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor):
        """Collapse the hc copies into one, weighted by pre_mix. [b,s,hc,d] x [b,s,hc] -> [b,s,d]"""
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_pre_sinkhorn_premix(
        self, x: torch.Tensor, pre_mix: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor
    ):
        """Calculate this sub-block's MHC coefficients and collapse its input."""
        if self.use_fused_mhc:
            return fused_mhc_pre_sinkhorn_premix(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                pre_mix,
                hc_mult=self.hc_mult,
                num_iters=self.hc_sinkhorn_iters,
                hc_eps=self.hc_eps,
                norm_eps=self.norm_eps,
            )
        pre, post, comb = self.hc_mixes(x, hc_fn, hc_scale, hc_base)
        return self.hc_pre(x, pre_mix), post, comb, pre

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        """Expand the sublayer output back to hc copies and mix the residual in through `comb`.
        x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc] -> [b,s,hc,d]
        """
        if self.use_fused_mhc:
            return fused_mhc_post(x, residual, post, comb)
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        pre_mix: torch.Tensor,
        image_mask: torch.Tensor | None,
        *attn_args: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sfa_layout: str = "BSND",
        cu_seqlens: torch.Tensor | None = None,
        engram_hash_ids: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        cp_group: torch.distributed.ProcessGroup | None = None,
        cp_state: ContextParallelAttentionState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, ContextParallelAttentionState]:
        """`pre_mix` collapses the hc_mult copies down to one input for this block's attention. Each
        sub-block's own `hc_mixes` produces the mix for the *next* one, so attention uses what the
        previous layer's FFN produced and the FFN uses what this attention produced.

        image_mask: [b, s] bool, True inside image spans (selects the VL routing bias).
        attn_args: Additional tensor inputs consumed by specialized attention subclasses.
        cp_group: Explicit KV-AllGather group for this block's attention, or None.
        cp_state: CP tensors from the preceding layer; returned alongside hidden
            states so activation checkpointing retains the correct dependencies.
        """
        # Run inside Block.forward so its FSDP pre-hook has materialized the
        # small Engram projection/gates. The embedding itself stays sharded.
        if engram_hash_ids is not None:
            x = self.engram(x, engram_hash_ids, engram_mask)
        residual = x
        x, attn_post, attn_comb, attn_pre = self.hc_pre_sinkhorn_premix(
            x, pre_mix, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(
            x,
            start_pos,
            *attn_args,
            attention_mask=attention_mask,
            sfa_layout=sfa_layout,
            cu_seqlens=cu_seqlens,
            **({"cp_group": cp_group, "cp_state": cp_state} if cp_group is not None else {}),
        )
        if cp_group is not None:
            x, cp_state = x
        x = self.hc_post(x, residual, attn_post, attn_comb)

        residual = x
        x, ffn_post, ffn_comb, ffn_pre = self.hc_pre_sinkhorn_premix(
            x, attn_pre, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, image_mask)
        x = self.hc_post(x, residual, ffn_post, ffn_comb)
        return (x, ffn_pre, cp_state) if cp_group is not None else (x, ffn_pre)


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """initial one-hot mix"""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix

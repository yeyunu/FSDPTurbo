from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from fsdp_turbo.ops.vision_flash_attention import fused_vision_flash_attention
from fsdp_turbo.ops.rope import rope
from fsdp_turbo.ops.rms_norm import rms_norm
from .activation import packed_swiglu


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # [tokens, heads, dim] -> [batch, tokens, heads, dim].
    return rope(
        x.unsqueeze(0),
        cos.unsqueeze(0),
        sin.unsqueeze(0),
        interleaved=False,
    ).squeeze(0)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "npu":
            return rms_norm(x, self.weight, self.eps)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = nn.Linear(3 * args.vision_patch_size**2, args.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.vision_n_heads
        self.head_dim = args.vision_dim // args.vision_n_heads
        self.use_fused_vision_flash_attn = args.use_fused_vision_flash_attn
        self.wqkv = nn.Linear(args.vision_dim, 3 * args.vision_dim)
        self.wo = nn.Linear(args.vision_dim, args.vision_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sequence_lengths: list[int] | None = None,
    ) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        def attend(q_part, k_part, v_part):
            if self.use_fused_vision_flash_attn:
                return fused_vision_flash_attention(
                    q_part.unsqueeze(0),
                    k_part.unsqueeze(0),
                    v_part.unsqueeze(0),
                ).squeeze(0)
            return F.scaled_dot_product_attention(
                q_part.transpose(0, 1),
                k_part.transpose(0, 1),
                v_part.transpose(0, 1),
            ).transpose(0, 1)

        if sequence_lengths is None:
            o = attend(q, k, v)
        else:
            # QKV projections run once so every FSDP-wrapped vision block has
            # one collective sequence per training step. Attention itself is
            # split locally to keep images fully isolated from one another.
            q_parts = q.split(sequence_lengths, dim=0)
            k_parts = k.split(sequence_lengths, dim=0)
            v_parts = v.split(sequence_lengths, dim=0)
            o = torch.cat(
                [attend(q_part, k_part, v_part) for q_part, k_part, v_part in zip(q_parts, k_parts, v_parts)],
                dim=0,
            )
        return self.wo(o.reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(args.vision_inter_dim, args.vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.w1(x)
        if gate_up.dtype == torch.bfloat16:
            # Preserve the reference's BF16 rounding after SiLU. The fused
            # activation can change it, causing output drift after w2.
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = F.silu(gate) * up
        else:
            intermediate = packed_swiglu(gate_up)
        return self.w2(intermediate)


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = RMSNorm(args.vision_dim)
        self.attn = Attention(args)
        self.norm2 = RMSNorm(args.vision_dim)
        self.mlp = MLP(args)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sequence_lengths: list[int] | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, sequence_lengths)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, args):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_size = args.vision_patch_size
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.vision_n_layers)])
        self.norm = RMSNorm(args.vision_dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos = cos.to(x.device)
        sin = sin.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward_packed(
        self,
        patches: list[torch.Tensor],
        grid_sizes: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        """Encode multiple images with one invocation of every FSDP module."""
        sequence_lengths = [patch.shape[0] for patch in patches]
        x = self.patch_embed(torch.cat(patches, dim=0))
        rope_coefficients = [get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta) for n_h, n_w in grid_sizes]
        cos = torch.cat([item[0] for item in rope_coefficients], dim=0).to(x.device)
        sin = torch.cat([item[1] for item in rope_coefficients], dim=0).to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin, sequence_lengths)
        return list(self.norm(x).split(sequence_lengths, dim=0))


class Aligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, args.dim)
        self.w2 = nn.Linear(args.dim, args.dim)

    def forward(self, x, n_h, n_w=None):
        if isinstance(x, (list, tuple)):
            return self._forward_packed(x, n_h)
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))

    def _forward_packed(
        self,
        features: list[torch.Tensor],
        grid_sizes: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        """Align multiple isolated image grids with one FSDP module call."""
        packed = []
        output_lengths = []
        r = self.downsample_ratio
        for x, (n_h, n_w) in zip(features, grid_sizes):
            x = x.view(n_h, n_w, -1).permute(2, 0, 1)
            x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
            x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
            packed.append(x)
            output_lengths.append(x.shape[0])
        output = self.w2(F.gelu(self.w1(torch.cat(packed, dim=0))))
        return list(output.split(output_lengths, dim=0))

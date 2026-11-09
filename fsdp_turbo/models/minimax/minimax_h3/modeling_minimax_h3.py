# Derived from DiffSynth-Studio 0ad850953156c028c6334ea8c17c50032a1ef169.
# Adapted from MindSpeed-MM 6e99966fa25168ae393e3f02451d6a3e4d6de8f1.
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from fsdp_turbo.distributed.context_parallel.batch import cal_split_sizes
from fsdp_turbo.distributed.context_parallel.communication import (
    gather_forward_split_backward,
    split_forward_gather_backward,
)
from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_attention import ulysses_context_parallel_attention
from fsdp_turbo.ops.flash_attn import flash_attention_forward
from fsdp_turbo.ops.moe import swiglu
from fsdp_turbo.ops.rms_norm import rms_norm
from fsdp_turbo.ops.rotary_mul import rotary_mul

from .constants import AUDIO_COND_NOISE_AUG, IMGVID_COND_NOISE_AUG
from .flow_match import FlowMatchScheduler


MINIMAX_H3_ADALN_MODALITY_NUM = 3
_PATCH_T, _PATCH_H, _PATCH_W = (1, 2, 2)


def patchify_video(latent: torch.Tensor) -> torch.Tensor:
    """Compute the H3 component output with reference tensor layout."""
    b, c, ft, fh, fw = (int(x) for x in latent.shape)
    t, h, w = (ft // _PATCH_T, fh // _PATCH_H, fw // _PATCH_W)
    packed = latent.reshape(b, c, t, _PATCH_T, h, _PATCH_H, w, _PATCH_W)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(b * t * h * w, c * _PATCH_T * _PATCH_H * _PATCH_W).contiguous()


def unpatchify_video(rows: torch.Tensor, t: int, h: int, w: int, channel: int = 24) -> torch.Tensor:
    """Compute the H3 component output with reference tensor layout."""
    packed = rows.reshape(-1, t, h, w, channel, _PATCH_T, _PATCH_H, _PATCH_W)
    latent = torch.einsum("nthwcrpq->nctrhpwq", packed)
    return latent.reshape(-1, channel, t * _PATCH_T, h * _PATCH_H, w * _PATCH_W).contiguous()


def pack_audio(latent: torch.Tensor) -> torch.Tensor:
    """Compute the H3 component output with reference tensor layout."""
    ac, ld, steps = (int(x) for x in latent.shape)
    return latent.permute(0, 2, 1).reshape(ac * steps, ld).contiguous()


def unpack_audio(rows: torch.Tensor, audio_channel: int, steps: int, latent_dim: int = 32) -> torch.Tensor:
    """Compute the H3 component output with reference tensor layout."""
    return rows.reshape(audio_channel, steps, latent_dim).permute(0, 2, 1).contiguous()


class _MiniMaxH3RMSNorm(nn.RMSNorm):
    """_MiniMaxH3RMSNorm: H3 structure preserving reference parameter names."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the H3 component output with reference tensor layout."""
        return rms_norm(x, self.weight, self.eps, use_eager=x.device.type != "npu")


def _norm(size: int, *, eps: float) -> nn.RMSNorm:
    return _MiniMaxH3RMSNorm(size, eps=eps)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    rot_dim = freqs.shape[-1]
    x_rot, x_pass = (x[..., :rot_dim], x[..., rot_dim:])
    cos = torch.cos(freqs).to(x.dtype)
    sin = torch.sin(freqs).to(x.dtype)
    x_rot = rotary_mul(x_rot.unsqueeze(0), cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)).squeeze(0)
    return torch.cat((x_rot, x_pass), dim=-1)


def _modulate_scale_shift(x, shift, scale, indices):
    return (x * (1.0 + scale.float().index_select(0, indices)) + shift.float().index_select(0, indices)).to(x.dtype)


def _modulate_gate(x, gate, other, indices):
    return (x + gate.float().index_select(0, indices) * other).to(x.dtype)


def _sdpa_varlen_attention(q, k, v, cu_seqlens, softmax_scale, module, total_seq_len=None, skip_ulysses=False):
    if cu_seqlens.numel() > 3:
        raise ValueError("H3 FL2VA currently supports one sample plus alignment padding.")

    def attention_fn(query, key, value, **_):
        # Isolate valid and alignment-padding segments after the shared CP exchange.
        # Return the full padded BSND output for the inverse all-to-all; calling
        # bare non-causal FA on the full sequence would mix padding into valid tokens.
        boundaries = cu_seqlens.tolist()
        segments = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if start == stop:
                continue
            query_segment, key_segment, value_segment = [tensor[:, :, start:stop] for tensor in (query, key, value)]
            if module.config._attn_implementation == "eager":
                output = F.scaled_dot_product_attention(
                    query_segment, key_segment, value_segment, scale=softmax_scale
                ).transpose(1, 2)
            else:
                output, _ = flash_attention_forward(
                    module, query_segment, key_segment, value_segment, scaling=softmax_scale, is_causal=False
                )
            if stop > start > 0:
                output = torch.zeros_like(output)
            segments.append(output)
        return torch.cat(segments, dim=1)

    query, key, value = [tensor.transpose(0, 1).unsqueeze(0) for tensor in (q, k, v)]
    if module.ulysses_group is not None and not skip_ulysses:
        output = ulysses_context_parallel_attention(
            attention_fn,
            query,
            key,
            value,
            ulysses_group=module.ulysses_group,
            total_seq_len=total_seq_len,
            is_causal=False,
        )
    else:
        output = attention_fn(query, key, value)
    return output.squeeze(0)


class MiniMaxH3Rope(nn.Module):
    """MiniMaxH3Rope: H3 structure preserving reference parameter names."""

    def __init__(self, inv_freq_len: int) -> None:
        """Initialize the reference H3 component."""
        super().__init__()
        self.inv_freq_len = inv_freq_len
        self.inv_freq = nn.Parameter(self._build_inv_freq())

    def _build_inv_freq(self, device=None) -> torch.Tensor:
        steps = torch.arange(0, self.inv_freq_len, dtype=torch.float32, device=device)
        return 1.0 / 10000.0 ** (steps / self.inv_freq_len)

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        """Compute the H3 component output with reference tensor layout."""
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(f"img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}")
        pos = img_position_ids[0].to(torch.float32)
        inv_freq = self._build_inv_freq(pos.device)
        per_axis = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        return torch.cat((half, half), dim=-1)


class MiniMaxH3TimeEmbedder(nn.Module):
    """MiniMaxH3TimeEmbedder: H3 structure preserving reference parameter names."""

    def __init__(self, timestep_input_dim: int, time_embed_hidden_size: int, time_embed_dim: int) -> None:
        """Initialize the reference H3 component."""
        super().__init__()
        self.frequency_embedding_size = timestep_input_dim
        self.proj_in = nn.Linear(timestep_input_dim, time_embed_hidden_size, bias=True)
        self.proj_out = nn.Linear(time_embed_hidden_size, time_embed_dim, bias=True)

    def forward(self, t: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        """Compute the H3 component output with reference tensor layout."""
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        hidden = self.proj_in(t_freq.to(dtype))
        hidden = nn.functional.silu(hidden)
        return self.proj_out(hidden)


class MiniMaxH3Attention(nn.Module):
    """MiniMaxH3Attention: H3 structure preserving reference parameter names."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm_eps: float,
        attn_implementation: str,
    ) -> None:
        """Initialize the reference H3 component."""
        super().__init__()
        self.ulysses_group = None
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        inner_dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim ** (-0.5)
        self.config = SimpleNamespace(_attn_implementation=attn_implementation)
        self.is_causal = False
        self.qkv_proj = nn.Linear(hidden_size, inner_dim * 3, bias=False)
        self.q_norm = _norm(attention_head_dim, eps=qk_norm_eps)
        self.k_norm = _norm(attention_head_dim, eps=qk_norm_eps)
        self.out_proj = nn.Linear(inner_dim, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_freqs: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
        total_seq_len: int | None = None,
        skip_ulysses: bool = False,
    ) -> torch.Tensor:
        """Compute the H3 component output with reference tensor layout."""
        total = x.shape[0]
        qkv = self.qkv_proj(x)
        qkv = qkv.view(total, self.num_heads, 3, self.head_dim)
        q = qkv[:, :, 0, :]
        k = qkv[:, :, 1, :]
        v = qkv[:, :, 2, :]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_freqs is not None:
            q = _apply_rope(q, rope_freqs)
            k = _apply_rope(k, rope_freqs)
        out = _sdpa_varlen_attention(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            softmax_scale=self.softmax_scale,
            module=self,
            total_seq_len=total_seq_len,
            skip_ulysses=skip_ulysses,
        )
        out = out.reshape(total, self.num_heads * self.head_dim)
        return self.out_proj(out)


class MiniMaxH3MLP(nn.Module):
    """MiniMaxH3MLP: H3 structure preserving reference parameter names."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        """Initialize the reference H3 component."""
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the H3 component output with reference tensor layout."""
        hidden = self.fc1(x)
        hidden = swiglu(hidden, dim=-1, fused=x.device.type == "npu")
        return self.fc2(hidden)


class MiniMaxH3AdalnProj(nn.Module):
    """MiniMaxH3AdalnProj: H3 structure preserving reference parameter names."""

    def __init__(
        self, hidden_size: int, time_embed_dim: int, out_features: int, *, expand_ratio: int, modality_num: int
    ) -> None:
        """Initialize the reference H3 component."""
        super().__init__()
        if out_features != expand_ratio * hidden_size * modality_num:
            raise ValueError(
                f"adaln out_features mismatch: {out_features} != {expand_ratio}*{hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = hidden_size
        self.linear = nn.Linear(time_embed_dim, out_features, bias=True)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Compute the H3 component output with reference tensor layout."""
        x = nn.functional.silu(t_emb)
        x = self.linear(x)
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """MiniMaxH3TokenRefinerBlock: H3 structure preserving reference parameter names."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_hidden_size: int,
        norm_eps: float,
        qk_norm_eps: float,
        attn_implementation: str,
    ) -> None:
        """Initialize the reference H3 component."""
        super().__init__()
        self.norm1 = _norm(hidden_size, eps=norm_eps)
        self.norm2 = _norm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps, attn_implementation
        )
        self.mlp = MiniMaxH3MLP(hidden_size, ffn_hidden_size)

    def forward(self, x: torch.Tensor, *, cu_seqlens: torch.Tensor, max_seqlen: int | None) -> torch.Tensor:
        """Compute the H3 component output with reference tensor layout."""
        x = x + self.attn(
            self.norm1(x), rope_freqs=None, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, skip_ulysses=True
        )
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    """MiniMaxH3TokenRefiner: H3 structure preserving reference parameter names."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_hidden_size: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
        attn_implementation: str,
    ) -> None:
        """Initialize the reference H3 component."""
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_hidden_size,
                    norm_eps,
                    qk_norm_eps,
                    attn_implementation,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = _norm(hidden_size, eps=final_norm_eps)

    def forward(self, x: torch.Tensor, *, cu_seqlens: torch.Tensor, max_seqlen: int | None) -> torch.Tensor:
        """Compute the H3 component output with reference tensor layout."""
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.final_norm(x)


class MiniMaxH3DiTBlock(nn.Module):
    """MiniMaxH3DiTBlock: H3 structure preserving reference parameter names."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_hidden_size: int,
        time_embed_dim: int,
        adaln_out_features: int,
        norm_eps: float,
        qk_norm_eps: float,
        attn_implementation: str,
    ) -> None:
        """Initialize the reference H3 component."""
        super().__init__()
        self.norm1 = _norm(hidden_size, eps=norm_eps)
        self.norm2 = _norm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps, attn_implementation
        )
        self.mlp = MiniMaxH3MLP(hidden_size, ffn_hidden_size)
        self.adaln_proj = MiniMaxH3AdalnProj(
            hidden_size, time_embed_dim, adaln_out_features, expand_ratio=6, modality_num=MINIMAX_H3_ADALN_MODALITY_NUM
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_freqs: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None,
        total_seq_len: int | None = None,
        past_key_values: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute the H3 component output with reference tensor layout."""
        del past_key_values
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        residual = x
        h = self.norm1(x)
        h = _modulate_scale_shift(h, shift_msa, scale_msa, combined_indices)
        h = self.attn(
            h, rope_freqs=rope_freqs, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, total_seq_len=total_seq_len
        )
        x = _modulate_gate(residual, gate_msa, h, combined_indices)
        residual = x
        h = self.norm2(x)
        h = _modulate_scale_shift(h, shift_mlp, scale_mlp, combined_indices)
        h = self.mlp(h)
        return _modulate_gate(residual, gate_mlp, h, combined_indices)


class MiniMaxH3FinalLayer(nn.Module):
    """MiniMaxH3FinalLayer: H3 structure preserving reference parameter names."""

    def __init__(
        self,
        hidden_size: int,
        time_embed_dim: int,
        final_adaln_out_features: int,
        latents_dim: int,
        audio_latents_dim: int,
        patch_size: int,
        final_norm_eps: float,
    ) -> None:
        """Initialize the reference H3 component."""
        super().__init__()
        video_patch_dim = latents_dim * patch_size[0] * patch_size[1] * patch_size[2]
        self.norm = _norm(hidden_size, eps=final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            hidden_size, time_embed_dim, final_adaln_out_features, expand_ratio=2, modality_num=1
        )
        self.video_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_out = nn.Linear(hidden_size, audio_latents_dim, bias=True)

    def forward(
        self, x: torch.Tensor, *, t_emb: torch.Tensor, inverse_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the H3 component output with reference tensor layout."""
        shift, scale = self.adaln_proj(t_emb)
        h = self.norm(x)
        h = _modulate_scale_shift(h, shift, scale, inverse_indices)
        video = self.video_out(h)
        audio = self.audio_out(h)
        return (video, audio)


class MiniMaxH3DiT(nn.Module):
    """MiniMaxH3DiT: H3 structure preserving reference parameter names."""

    cp_group = None
    ulysses_group = None

    def configure_context_parallel(self, *, cp_group: dist.ProcessGroup, ulysses_group: dist.ProcessGroup) -> None:
        """Bind Training-created groups through the same interface as Qwen3.5.

        Only Ulysses is supported. Group sizes and attention heads must match.
        """
        if cp_group is None or ulysses_group is None:
            raise ValueError("H3 CP requires explicit CP and Ulysses process groups.")
        size = dist.get_world_size(cp_group)
        if dist.get_world_size(ulysses_group) != size:
            raise ValueError("H3 supports Ulysses only: CP and Ulysses group sizes must match.")
        if self.attn_implementation != "flash_attention_2":
            raise NotImplementedError("H3 CP requires flash_attention_2.")
        if self.num_attention_heads % size:
            raise ValueError("H3 attention heads must be divisible by the Ulysses group size.")
        self.cp_group = cp_group
        self.ulysses_group = ulysses_group
        for module in self.modules():
            if isinstance(module, MiniMaxH3Attention):
                module.ulysses_group = ulysses_group

    _repeated_blocks = ["MiniMaxH3DiTBlock"]

    def __init__(
        self,
        num_layers: int = 50,
        token_refiner_num_layers: int = 2,
        hidden_size: int = 5376,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        ffn_hidden_size: int = 14336,
        latents_dim: int = 24,
        audio_latents_dim: int = 32,
        patch_size: tuple = (1, 2, 2),
        text_dim: int = 5120,
        timestep_input_dim: int = 256,
        time_embed_hidden_size: int = 5376,
        time_embed_dim: int = 2688,
        adaln_out_features: int = 96768,
        final_adaln_out_features: int = 10752,
        rope_inv_freq_len: int = 16,
        norm_eps: float = 1e-05,
        qk_norm_eps: float = 1e-05,
        final_norm_eps: float = 1e-05,
        attn_implementation: str = "eager",
        **kwargs: object,
    ) -> None:
        """Initialize the reference H3 component."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_channels_latents = latents_dim
        self.attn_implementation = attn_implementation
        video_patch_dim = latents_dim * patch_size[0] * patch_size[1] * patch_size[2]
        self.video_patch_proj = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(audio_latents_dim, hidden_size, bias=True)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True)
        self.time_embedder = MiniMaxH3TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim)
        self.rope = MiniMaxH3Rope(rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(
            token_refiner_num_layers,
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            ffn_hidden_size,
            norm_eps,
            qk_norm_eps,
            final_norm_eps,
            attn_implementation,
        )
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3DiTBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_hidden_size,
                    time_embed_dim,
                    adaln_out_features,
                    norm_eps,
                    qk_norm_eps,
                    attn_implementation,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = MiniMaxH3FinalLayer(
            hidden_size,
            time_embed_dim,
            final_adaln_out_features,
            latents_dim,
            audio_latents_dim,
            patch_size,
            final_norm_eps,
        )

    def _embed(
        self,
        *,
        x,
        audio_x,
        text_embeddings_selected,
        unique_timesteps,
        img_pos,
        audio_pos,
        text_pos,
        refiner_cu_seqlens,
        refiner_max_seqlen,
        seq_len,
        device,
    ):
        dtype = text_embeddings_selected.dtype
        x_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos).to(dtype)
        video_embed = self.video_patch_proj(x_rows)
        audio_rows = audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos).to(dtype)
        audio_embed = self.audio_patch_proj(audio_rows)
        text_rows = text_embeddings_selected.to(device=device)
        text_embed = self.condition_proj(text_rows)
        text_embed = self.token_refiner(text_embed, cu_seqlens=refiner_cu_seqlens, max_seqlen=refiner_max_seqlen)
        embeddings = torch.zeros((seq_len, self.hidden_size), device=device, dtype=dtype)
        embeddings.index_add_(0, text_pos, text_embed.to(dtype)[: text_pos.shape[0]])
        embeddings.index_add_(0, img_pos, video_embed.to(dtype)[: img_pos.shape[0]])
        embeddings.index_add_(0, audio_pos, audio_embed.to(dtype)[: audio_pos.shape[0]])
        t_emb = self.time_embedder(unique_timesteps, dtype=dtype)
        return (embeddings, t_emb)

    def forward(
        self,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        img_position_ids: torch.Tensor,
        unique_timesteps: torch.Tensor,
        inverse_indices: torch.Tensor,
        update_mask: torch.Tensor,
        token_tags: torch.Tensor,
        prompt_embeds: torch.Tensor,
        img_pos_info: dict[str, torch.Tensor],
        audio_pos_info: dict[str, torch.Tensor],
        text_pos_info: dict[str, torch.Tensor],
        img_pos_for_infer_output_info: dict[str, torch.Tensor],
        packed_seq_params: dict[str, torch.Tensor | int],
        refiner_packed_seq_params: dict[str, torch.Tensor | int],
        update_audio_mask: torch.Tensor | None = None,
        skip_mask_out_condition: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the H3 component output with reference tensor layout."""
        inverse_indices = inverse_indices.view(-1).to(torch.long)
        token_tags = token_tags.view(-1).to(torch.long)
        text_selected = prompt_embeds
        img_pos = img_pos_info["position_ids"].view(-1).to(torch.long)
        audio_pos = audio_pos_info["position_ids"].view(-1).to(torch.long)
        text_pos = text_pos_info["position_ids"].view(-1).to(torch.long)
        infer_out_pos = img_pos_for_infer_output_info["position_ids"].view(-1).to(torch.long)
        cu_seqlens = packed_seq_params["cu_seqlens_q"].to(torch.int32)
        max_seqlen = int(packed_seq_params["max_seqlen_q"])
        refiner_cu = refiner_packed_seq_params["cu_seqlens_q"].to(torch.int32)
        refiner_max = int(refiner_packed_seq_params["max_seqlen_q"])
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        device = x.device
        use_ulysses = self.ulysses_group is not None
        rope_freqs = self.rope(img_position_ids).to(device)
        decoder_input, t_emb = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos.to(device),
            audio_pos=audio_pos.to(device),
            text_pos=text_pos.to(device),
            refiner_cu_seqlens=refiner_cu.to(device),
            refiner_max_seqlen=refiner_max,
            seq_len=seq_len,
            device=device,
        )
        combined_indices = (inverse_indices * MINIMAX_H3_ADALN_MODALITY_NUM + token_tags.clamp(min=0)).to(device)
        inverse_indices = inverse_indices.to(device)
        hidden = decoder_input
        if use_ulysses:
            split_sizes = cal_split_sizes(seq_len, dist.get_world_size(self.ulysses_group))
            hidden = split_forward_gather_backward(hidden, self.ulysses_group, dim=0, split_sizes=split_sizes)
            rope_freqs = split_forward_gather_backward(rope_freqs, self.ulysses_group, dim=0, split_sizes=split_sizes)
            combined_indices = split_forward_gather_backward(
                combined_indices, self.ulysses_group, dim=0, split_sizes=split_sizes
            )
        cu_seqlens = cu_seqlens.to(device)
        for block in self.blocks:
            hidden = block(
                hidden,
                t_emb=t_emb,
                combined_indices=combined_indices,
                rope_freqs=rope_freqs,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                total_seq_len=seq_len,
            )
        if use_ulysses:
            hidden = gather_forward_split_backward(
                hidden,
                self.ulysses_group,
                dim=0,
                gather_sizes=cal_split_sizes(seq_len, dist.get_world_size(self.ulysses_group)),
            )
        video_logits, audio_logits = self.final_layer(hidden, t_emb=t_emb, inverse_indices=inverse_indices)
        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(f"update_mask length mismatch: {update_mask.shape[0]} != {video_logits.shape[0]}")
            video_logits = video_logits * update_mask.unsqueeze(-1)
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return (video_logits, audio_logits)


@dataclass
class MiniMaxH3TrainingOutput:
    """Combined video/audio Flow Matching loss returned to the trainer."""

    loss: torch.Tensor


class MiniMaxH3ForTraining(MiniMaxH3DiT):
    """Preserve the DiffSynth DiT module tree and add the SFT loss."""

    def __init__(
        self,
        *args: object,
        loss_replay_seed: int | None = None,
        loss_replay_data_parallel_rank: int = 0,
        loss_replay_data_parallel_size: int = 1,
        **kwargs: object,
    ) -> None:
        """Initialize video and audio flow-matching schedules."""
        super().__init__(*args, **kwargs)
        self.loss_replay_seed = loss_replay_seed
        self.loss_replay_step = 0
        self.loss_replay_data_parallel_rank = loss_replay_data_parallel_rank
        self.loss_replay_data_parallel_size = loss_replay_data_parallel_size
        self.scheduler = FlowMatchScheduler("MiniMax-H3")
        self.scheduler_audio = FlowMatchScheduler("MiniMax-H3")
        self.scheduler.set_timesteps(1000, training=True, shift=12.0)
        self.scheduler_audio.set_timesteps(1000, training=True, shift=3.0)

    @staticmethod
    def _restore_position_ids(position_ids_bits):
        return position_ids_bits.view(torch.float64)

    def _model_fn(
        self,
        video_latents,
        audio_latents,
        packed,
        prompt_embeds,
        t_video,
        t_audio,
        keyframe_cond_anchor,
        ref_visual_anchor,
        ref_audio_anchor,
    ):
        device = video_latents.device
        dtype = video_latents.dtype
        video_rows = patchify_video(video_latents)
        audio_rows = pack_audio(audio_latents)
        seq_len = packed["seq_len"]
        img_pos = packed["img_pos"]
        audio_pos = packed["audio_pos"]
        text_pos = packed["text_pos"]
        text_len = packed["text_len"]
        cond_rows = packed["cond_rows"]
        ref_audio_rows = packed["ref_audio_rows"]
        x = torch.zeros(1, seq_len, 96, dtype=dtype, device=device)
        audio_x = torch.zeros(1, seq_len, 32, dtype=dtype, device=device)
        if ref_visual_anchor is not None:
            x[0].index_copy_(0, img_pos[:cond_rows], ref_visual_anchor)
            x[0].index_copy_(0, img_pos[cond_rows:], video_rows)
        elif keyframe_cond_anchor is not None:
            x[0].index_copy_(0, img_pos[:cond_rows], keyframe_cond_anchor)
            x[0].index_copy_(0, img_pos[cond_rows:], video_rows)
        else:
            x[0].index_copy_(0, img_pos, video_rows)
        if ref_audio_anchor is not None:
            audio_x[0].index_copy_(0, audio_pos[:ref_audio_rows], ref_audio_anchor)
            audio_x[0].index_copy_(0, audio_pos[ref_audio_rows:], audio_rows)
        else:
            audio_x[0].index_copy_(0, audio_pos, audio_rows)
        timesteps = torch.full((seq_len,), float(t_video), dtype=torch.float32, device=device)
        timesteps[audio_pos] = float(t_audio)
        if cond_rows:
            timesteps[img_pos[:cond_rows]] = max(float(t_video), IMGVID_COND_NOISE_AUG)
        if ref_audio_rows:
            timesteps[audio_pos[:ref_audio_rows]] = max(float(t_audio), AUDIO_COND_NOISE_AUG)
        unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)
        refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device)
        video_rows_pred, audio_rows_pred = super().forward(
            x=x,
            audio_x=audio_x,
            img_position_ids=packed["img_position_ids"],
            unique_timesteps=unique_timesteps,
            inverse_indices=inverse_indices,
            update_mask=packed["update_mask"],
            token_tags=packed["token_tags"],
            prompt_embeds=prompt_embeds,
            img_pos_info={"position_ids": img_pos},
            audio_pos_info={"position_ids": audio_pos},
            text_pos_info={"position_ids": text_pos},
            img_pos_for_infer_output_info={"position_ids": img_pos},
            packed_seq_params={"cu_seqlens_q": packed["cu_seqlens"], "max_seqlen_q": int(packed["cu_seqlens"][1])},
            refiner_packed_seq_params={"cu_seqlens_q": refiner_cu, "max_seqlen_q": text_len},
            skip_mask_out_condition=False,
        )
        if cond_rows:
            video_rows_pred = video_rows_pred[cond_rows:]
        if ref_audio_rows:
            audio_rows_pred = audio_rows_pred[ref_audio_rows:]
        video_pred = unpatchify_video(
            video_rows_pred, packed["latent_t"], packed["latent_h_patched"], packed["latent_w_patched"]
        )
        audio_pred = unpack_audio(audio_rows_pred, packed["audio_channel"], packed["audio_t"])
        return (-video_pred, -audio_pred)

    # Training receives cache fields and delegates the DiT call in _model_fn.
    # pylint: disable-next=arguments-differ
    def forward(
        self,
        input_latents: torch.Tensor,
        audio_input_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        packed_seq_len: int,
        packed_img_pos: torch.Tensor,
        packed_audio_pos: torch.Tensor,
        packed_text_pos: torch.Tensor,
        packed_update_mask: torch.Tensor,
        packed_img_position_ids_bits: torch.Tensor,
        packed_token_tags: torch.Tensor,
        packed_cu_seqlens: torch.Tensor,
        packed_text_len: int,
        packed_audio_channel: int,
        packed_audio_t: int,
        packed_latent_t: int,
        packed_latent_h_patched: int,
        packed_latent_w_patched: int,
        packed_cond_rows: int = 0,
        packed_ref_audio_rows: int = 0,
        keyframe_cond_anchor: torch.Tensor | None = None,
        ref_visual_anchor: torch.Tensor | None = None,
        ref_audio_anchor: torch.Tensor | None = None,
        use_cache: bool = False,
        **_: object,
    ) -> MiniMaxH3TrainingOutput:
        """Compute video and audio flow-matching loss from a precomputed cache sample."""
        del use_cache
        packed = {
            "seq_len": packed_seq_len,
            "img_pos": packed_img_pos,
            "audio_pos": packed_audio_pos,
            "text_pos": packed_text_pos,
            "update_mask": packed_update_mask,
            "img_position_ids": self._restore_position_ids(packed_img_position_ids_bits),
            "token_tags": packed_token_tags,
            "cu_seqlens": packed_cu_seqlens,
            "text_len": packed_text_len,
            "audio_channel": packed_audio_channel,
            "audio_t": packed_audio_t,
            "latent_t": packed_latent_t,
            "latent_h_patched": packed_latent_h_patched,
            "latent_w_patched": packed_latent_w_patched,
            "cond_rows": packed_cond_rows,
            "ref_audio_rows": packed_ref_audio_rows,
        }
        generator = None
        if self.loss_replay_seed is not None:
            data_parallel_rank = self.loss_replay_data_parallel_rank
            data_parallel_size = self.loss_replay_data_parallel_size
            seed = self.loss_replay_seed + self.loss_replay_step * data_parallel_size + data_parallel_rank
            self.loss_replay_step += 1
            generator = torch.Generator("cpu").manual_seed(seed)
        timestep_id = torch.randint(0, len(self.scheduler.timesteps), (1,), generator=generator)
        timestep_video = self.scheduler.timesteps[timestep_id].to(dtype=input_latents.dtype)
        timestep_audio = self.scheduler_audio.timesteps[timestep_id].to(dtype=input_latents.dtype)
        if generator is None:
            video_noise = torch.randn_like(input_latents)
            audio_noise = torch.randn_like(audio_input_latents)
        else:
            video_noise = torch.randn(input_latents.shape, generator=generator, dtype=input_latents.dtype).to(
                input_latents.device
            )
            audio_noise = torch.randn(
                audio_input_latents.shape, generator=generator, dtype=audio_input_latents.dtype
            ).to(audio_input_latents.device)
        video_latents = self.scheduler.add_noise(input_latents, video_noise, timestep_video)
        video_target = self.scheduler.training_target(input_latents, video_noise, timestep_video)
        audio_latents = self.scheduler_audio.add_noise(audio_input_latents, audio_noise, timestep_audio)
        audio_target = self.scheduler_audio.training_target(audio_input_latents, audio_noise, timestep_audio)
        video_pred, audio_pred = self._model_fn(
            video_latents,
            audio_latents,
            packed,
            prompt_embeds,
            1.0 - float(timestep_video) / self.scheduler.num_train_timesteps,
            1.0 - float(timestep_audio) / self.scheduler_audio.num_train_timesteps,
            keyframe_cond_anchor,
            ref_visual_anchor,
            ref_audio_anchor,
        )
        video_loss = F.mse_loss(video_pred.float(), video_target.float())
        video_loss *= self.scheduler.training_weight(timestep_video)
        audio_loss = F.mse_loss(audio_pred.float(), audio_target.float())
        audio_loss *= self.scheduler_audio.training_weight(timestep_audio)
        return MiniMaxH3TrainingOutput(loss=video_loss + audio_loss)

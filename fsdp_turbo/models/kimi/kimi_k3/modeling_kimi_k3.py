# coding=utf-8
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2025-2026 The Moonshot AI Team and HuggingFace Inc. team. All rights reserved.
# Copyright c) 2026 Tenstorrent AI
# 2025-2026 The Moonshot AI Team DeepSeek-AI and HuggingFace Inc. team. All rights reserved
# 2026 Tenstorrent AI ULC
#
# The code is based on llava (llava/modeling_llava.py), but modified for Kimi-K3.
#
# Licensing Information:
# - Code derived from llava (llava/modeling_llava.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# pylint: disable=relative-beyond-top-level,consider-using-from-import,no-name-in-module,too-many-lines,useless-parent-delegation,consider-merging-isinstance,redefined-outer-name,attribute-defined-outside-init

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import activations

try:
    from transformers.activations import PytorchGELUTanh
except ImportError:
    from transformers.activations import GELUTanh

    activations.PytorchGELUTanh = GELUTanh
    PytorchGELUTanh = GELUTanh
from transformers.activations import PytorchGELUTanh
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from transformers.utils import is_flash_attn_2_available

from .configuration_kimi_k3 import KimiK3Config
from .modeling_kimi_linear import KimiLinearForCausalLM

# Flash attention imports
if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None
from fsdp_turbo.memory.chunk_loss.chunk_loss import calculate_lm_loss, chunk_loss
from .kimi_flash_attention import context_parallel_active
from fsdp_turbo.utils.device import get_accelerator_name
from ...utils import register_model

from fsdp_turbo.distributed.context_parallel.batch import shard_fused_sequence_inputs
from fsdp_turbo.distributed.context_parallel.loss import cp_compute_lm_loss
from fsdp_turbo.distributed.context_parallel.hybrid_cp import HybridCPRuntime
from fsdp_turbo.distributed.context_parallel.kda_hybrid.p2p import (
    KdaCPContext,
    exchange_causal_history,
)
from fsdp_turbo.distributed.context_parallel.ulysses.utils import ulysses_all_to_all
from fsdp_turbo.distributed.context_parallel.kv_allgather.utils import kv_allgather
from fsdp_turbo.ops.kda_p2p import chunk_kda_p2p
from fsdp_turbo.ops.flash_attn import flash_attention_forward, vision_attention
from fsdp_turbo.ops.rope import rope

from .kimi_moe_patch import apply_kimi_k3_moe_patch

_SIN = None
_COS = None

apply_kimi_k3_moe_patch()


def set_global_param(param_name: str = None, param: Optional[torch.Tensor] = None) -> None:
    if param_name == "sin":
        global _SIN
        _SIN = param
    elif param_name == "cos":
        global _COS
        _COS = param
    else:
        raise ValueError(f"Invalid param type: '{param_name}'.")


def get_global_param(param_name: str = None) -> Optional[torch.Tensor]:
    if param_name == "sin":
        return _SIN
    elif param_name == "cos":
        return _COS
    else:
        raise ValueError(f"Invalid param type: '{param_name}'.")


def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    deterministic: bool = False,
):
    """Multi-head attention using flash attention 2.

    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        q_cu_seqlens (torch.Tensor): cumulative sequence lengths of q.
            The first element should be 0 and the last element should be q.shape[0].
        k_cu_seqlens (torch.Tensor): cumulative sequence lengths of k.
            The first element should be 0 and the last element should be k.shape[0].

    Returns:
        output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,
            where dim = num_heads * head_dim
    """
    if get_accelerator_name() == "npu":
        attn_out = vision_attention(q, k, v, q_cu_seqlens, k_cu_seqlens)
    else:
        attn_out = flash_attn_varlen_func(
            q,
            k,
            v,
            q_cu_seqlens,
            k_cu_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            causal=False,
            deterministic=deterministic,
        )
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]

    attn_out = attn_out.flatten(start_dim=-2)

    return attn_out


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = q.shape[0]
    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight = attn_weight.masked_fill(~attention_mask, torch.finfo(attn_weight.dtype).min)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": multihead_attention,
    "eager": eager_attention,
}


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def get_rope_shape_decorate(func):
    _get_rope_shape_first_call_flag = set()

    def wrapper(org, interpolation_mode, shape):
        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)
        if key not in _get_rope_shape_first_call_flag:
            _get_rope_shape_first_call_flag.add(key)
            _ = func(org, interpolation_mode, shape=(64, 64))
        return func(org, interpolation_mode, shape)

    return wrapper


@get_rope_shape_decorate
def get_rope_shape(org, interpolation_mode, shape):
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    if get_accelerator_name() == "npu":
        # Modification start
        cos = get_global_param("cos")
        sin = get_global_param("sin")
        xq_out = rope(xq.float().unsqueeze(0), cos[..., ::2], sin[..., ::2], interleaved=True).squeeze(0)
        xk_out = rope(xk.float().unsqueeze(0), cos[..., ::2], sin[..., ::2], interleaved=True).squeeze(0)
        # Modification end
    else:
        freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
        # ..., num_heads, head_dim/2
        xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    From:
    https://github.com/OpenGVLab/InternVideo/blob/421f6d2361fc8f61a3394244571f2601a4e99e29/InternVideo2/multi_modality/models/backbones/internvideo2/pos_embed.py#L86
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    def __init__(self, height: int, width: int, num_frames: int, dim: int, interpolation_mode: str = 'bicubic') -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            'time_weight',
            torch.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames)).float().unsqueeze(1),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames, f't:{t} > self.num_frames:{self.num_frames}'
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape(
                    self.weight,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + torch.cat(pos_embs)
        return out


class MoonVision3dPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = 'divided_fixed',
        patch_embed_proj_bias: bool = True,
        pos_emb_interpolation_mode: str = 'bicubic',
    ):
        super().__init__()
        assert isinstance(patch_size, int | Sequence), f'Invalid patch_size type: {type(patch_size)}'
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2, f'Expected patch_size to be a tuple of 2, got {patch_size}'
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size, bias=patch_embed_proj_bias)

        if pos_emb_type == 'divided_fixed':
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
                interpolation_mode=pos_emb_interpolation_mode,
            )
        else:
            raise NotImplementedError(f'Not support pos_emb_type: {pos_emb_type}')

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hws (N, 3): temporal, height and width

        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_thws)
        return x


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding with multi-resolution support.

    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.

    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py

    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, 'dim must be divisible by 4'
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def extra_repr(self):
        return f'dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}'

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.

        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_thws: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid time, height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if not hasattr(self, 'freqs_cis'):
            self.register_buffer('freqs_cis', self._precompute_freqs_cis(device), persistent=False)

        shapes = grid_thws.tolist()
        assert all(1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1) for t, h, w in shapes],
            dim=0,
        )
        return freqs_cis


class MLP2(nn.Module):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


class MoonViTEncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        qkv_hidden_size: int | None = None,
        norm_type: str = 'layernorm',
        mlp_type: str = 'mlp2',
        *,
        attn_implementation: str = 'flash_attention_2',
        activation=F.gelu,
        attn_bias: bool = False,
        linear_bias: bool = True,
        use_deterministic_attn: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.qkv_hidden_size = hidden_dim if qkv_hidden_size is None else qkv_hidden_size
        self.hidden_size_per_attention_head = self.qkv_hidden_size // self.num_heads
        self.attn_implementation = attn_implementation
        self.use_deterministic_attn = use_deterministic_attn

        if norm_type == "layernorm":
            self.norm0 = nn.LayerNorm(hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
        elif norm_type == "rmsnorm":
            self.norm0 = nn.RMSNorm(hidden_dim)
            self.norm1 = nn.RMSNorm(hidden_dim)
        else:
            raise NotImplementedError(f"Not support norm_type: {norm_type}")

        if mlp_type == "mlp2":
            self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation, bias=linear_bias)
        else:
            raise NotImplementedError(f"Not support mlp_type: {mlp_type}")

        self.wqkv = nn.Linear(hidden_dim, self.qkv_hidden_size * 3, bias=attn_bias)
        self.wo = nn.Linear(self.qkv_hidden_size, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        """
        Args:
            x (torch.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (torch.Tensor):
        """
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        # CP runs need the fused NPU kernel; single-rank runs keep the
        # upstream config-driven implementation.
        if context_parallel_active():
            attn_func = multihead_attention
        else:
            attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_out = attn_func(
            xq,
            xk,
            xv,
            q_cu_seqlens=cu_seqlens,
            k_cu_seqlens=cu_seqlens,
            max_seqlen_k=max_seqlen,
            max_seqlen_q=max_seqlen,
            deterministic=self.use_deterministic_attn,
        )

        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor | None = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        hidden_states = self.attention_qkvpacked(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MoonViT3dEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, block_cfg: dict, use_deterministic_attn: bool = False) -> None:
        super().__init__()
        self.use_deterministic_attn = use_deterministic_attn

        qkv_hidden_size = (
            block_cfg['hidden_dim'] if block_cfg.get('qkv_hidden_size') is None else block_cfg['qkv_hidden_size']
        )
        self.rope_2d = Rope2DPosEmbRepeated(qkv_hidden_size // block_cfg['num_heads'], 512, 512)
        self.blocks = nn.ModuleList(
            [
                MoonViTEncoderLayer(**block_cfg, use_deterministic_attn=self.use_deterministic_attn)
                for _ in range(num_layers)
            ]
        )
        norm_type = block_cfg.get('norm_type', 'layernorm')
        if norm_type == "layernorm":
            self.final_layernorm = nn.LayerNorm(hidden_dim)
        elif norm_type == "rmsnorm":
            self.final_layernorm = nn.RMSNorm(hidden_dim)
        else:
            raise NotImplementedError(f"Not support norm_type: {norm_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws=grid_thws, device=hidden_states.device)

        lengths = torch.cat(
            (
                torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )

        max_seqlen = lengths.max()
        cu_seqlens = lengths.to(hidden_states.device).cumsum(dim=0, dtype=torch.int32)

        if get_accelerator_name() == "npu":
            cu_seqlens = tuple(cu_seqlens[1:].cpu().numpy().tolist())

        cos = rope_freqs_cis.unsqueeze(-2).real.to(torch.float32).repeat_interleave(2, dim=-1).contiguous()
        sin = rope_freqs_cis.unsqueeze(-2).imag.to(torch.float32).repeat_interleave(2, dim=-1).contiguous()
        set_global_param("cos", cos)
        set_global_param("sin", sin)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


def tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    d_model = x.size(-1)

    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        # Get the current sequence
        seq = x[pre_sum : pre_sum + t * h * w]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(t, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)  # temporal pooling
        padded_seq = reshaped_seq.view(new_height * new_width, kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w

    return outputs


class MoonViT3dPretrainedModel(PreTrainedModel):
    config_class = None
    model_type = 'moonvit3d'
    _no_split_modules = ['MoonViTEncoderLayer']
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config = deepcopy(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
            patch_embed_proj_bias=getattr(config, 'patch_embed_proj_bias', True),
            pos_emb_interpolation_mode=getattr(config, 'pos_emb_interpolation_mode', 'bicubic'),
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                'num_heads': config.num_attention_heads,
                'hidden_dim': config.hidden_size,
                'qkv_hidden_size': getattr(config, 'qkv_hidden_size', None),
                'mlp_dim': config.intermediate_size,
                'norm_type': getattr(config, 'norm_type', 'layernorm'),
                'mlp_type': getattr(config, 'mlp_type', 'mlp2'),
                'activation': PytorchGELUTanh(),
                'attn_bias': getattr(config, 'attn_bias', True),
                'linear_bias': getattr(config, 'linear_bias', True),
                'attn_implementation': config._attn_implementation,
            },
            use_deterministic_attn=getattr(self, 'use_deterministic_attn', False),
        )

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_thws (torch.Tensor): Temporal, height and width.

        Returns:
            torch.Tensor: The output tokens.
        """
        # grid_thws = grid_thws.to('cpu')
        assert grid_thws.ndim == 2, f'grid_thws should be 2D, got {grid_thws.ndim}'
        assert grid_thws.size(1) == 3, f'No support for thw: {grid_thws}'
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        if self.merge_type == 'sd2_tpool':  # spatial downsampling 2x with temporal pooling all
            hidden_states = tpool_patch_merger(hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size)
        else:
            raise NotImplementedError(f'Not support {self.merge_type}')

        return hidden_states


# ============================================================================
# MM Projector Helper Classes (from mm_projector/modeling_mm_projectors.py)
# ============================================================================


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO, use faster LayerNorm
        self.pre_norm = nn.LayerNorm(config.mm_hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        assert isinstance(x, list | tuple), f'x is not a list or tuple: {type(x)}'
        lengths = [item.shape[0] for item in x]
        x = torch.cat(x, dim=0)
        x = self.pre_norm(x)
        x = self.proj(x)
        x = torch.split(x, lengths, dim=0)

        return x


class PatchMergerMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        eps = config.projector_ln_eps
        self.hidden_size = config.mm_hidden_size * (config.merge_kernel_size[0] * config.merge_kernel_size[1])
        self.pre_norm = nn.LayerNorm(config.mm_hidden_size, eps=eps)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list) or isinstance(x, tuple):
            x = [self.proj(self.pre_norm(item).view(item.shape[0], -1)) for item in x]
        else:
            # B, N, N_k, C = x.shape
            B = x.shape[0]
            x = self.proj(self.pre_norm(x).view(B, -1, self.hidden_size))
        return x


class PatchMergerMLPV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        eps = config.projector_ln_eps
        self.hidden_size = config.mm_hidden_size * (config.merge_kernel_size[0] * config.merge_kernel_size[1])
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.hidden_size, bias=False),
        )
        self.post_norm = nn.RMSNorm(config.hidden_size, eps=eps)
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list) or isinstance(x, tuple):
            lengths = [item.shape[0] for item in x]
            x = torch.concat([item.view(item.shape[0], -1) for item in x], dim=0)
            x = self.post_norm(self.proj(x))
            x = torch.split(x, lengths, dim=0)
        else:
            # B, N, N_k, C = x.shape
            B = x.shape[0]
            x = self.proj(x.view(B, -1, self.hidden_size))
            x = self.post_norm(x)
        return x


class KimiK3PreTrainedModel(PreTrainedModel):
    config_class = KimiK3Config
    base_model_prefix = "model"
    _no_split_modules = [
        "MoonViT3dPretrainedModel",
        "MoonViTEncoderLayer",
        "KimiDecoderLayer",
        "PatchMergerMLP",
        "PatchMergerMLPV2",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = False

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class VisionTowerConfig(PretrainedConfig):
    model_type = 'moonvit3d'

    def __init__(self, config: KimiK3Config, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = config.patch_size
        self.init_pos_emb_height = config.init_pos_emb_height
        self.init_pos_emb_width = config.init_pos_emb_width
        self.init_pos_emb_time = config.init_pos_emb_time
        self.pos_emb_type = config.pos_emb_type
        self.num_attention_heads = config.vt_num_attention_heads
        self.num_hidden_layers = config.vt_num_hidden_layers
        self.hidden_size = config.vt_hidden_size
        self.intermediate_size = config.vt_intermediate_size
        self.merge_kernel_size = config.merge_kernel_size
        self.merge_type = config.merge_type
        self._attn_implementation = config._attn_implementation
        self.qkv_hidden_size = getattr(config, 'qkv_hidden_size', None)
        self.norm_type = getattr(config, 'norm_type', 'layernorm')
        self.attn_bias = getattr(config, 'attn_bias', True)
        self.patch_embed_proj_bias = getattr(config, 'patch_embed_proj_bias', True)
        self.mlp_type = getattr(config, 'mlp_type', 'mlp2')
        self.linear_bias = getattr(config, 'linear_bias', True)
        self.pos_emb_interpolation_mode = getattr(config, 'pos_emb_interpolation_mode', 'bilinear')


class ProjectorConfig:
    def __init__(self, config: KimiK3Config):
        self.mm_projector_type = config.mm_projector_type
        self.mm_hidden_size = config.mm_hidden_size
        self.hidden_size = config.text_hidden_size
        self.merge_kernel_size = config.merge_kernel_size
        self.projector_hidden_act = config.projector_hidden_act
        self.projector_ln_eps = config.projector_ln_eps


# ref https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/llava/modeling_llava.py#L240
@register_model("kimi_k3")
class KimiK3ForConditionalGeneration(KimiK3PreTrainedModel):
    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def __init__(self, config: KimiK3Config):
        super().__init__(config)

        vt_config = VisionTowerConfig(config.vision_config)
        self.vision_tower = MoonViT3dPretrainedModel(vt_config)

        proj_config = ProjectorConfig(config.vision_config)
        if proj_config.mm_projector_type == 'identity':
            self.mm_projector = IdentityMap()
        elif proj_config.mm_projector_type == 'mlp':
            self.mm_projector = MLP(proj_config)
        elif proj_config.mm_projector_type == 'patchmerger':
            self.mm_projector = PatchMergerMLP(proj_config)
        elif proj_config.mm_projector_type == 'patchmergerv2':
            self.mm_projector = PatchMergerMLPV2(proj_config)
        else:
            raise ValueError(f"Unsupported mm_projector_type: {proj_config.mm_projector_type}")

        from .kimi_moe_patch import apply_kimi_k3_moe_patch

        apply_kimi_k3_moe_patch()

        self.language_model = KimiLinearForCausalLM(config.text_config)
        self.post_init()

        if hasattr(self.language_model, 'dtype'):
            target_dtype = self.language_model.dtype
            self.vision_tower = self.vision_tower.to(dtype=target_dtype)
            self.mm_projector = self.mm_projector.to(dtype=target_dtype)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Export Kimi packed experts with the original checkpoint/vLLM ABI.

        Before FSDP-Turbo parallelizes the model, Kimi experts retain their
        physical 3-D keys for the adapter's initialization snapshot.  Once EP
        or FSDP restoration has activated external export, rewrite any packed
        keys collected by FSDP2 to the standard per-expert Kimi names.
        """
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        from .kimi_moe_patch import rewrite_kimi_k3_state_dict_for_external_export

        return rewrite_kimi_k3_state_dict_for_external_export(self, state_dict, prefix)

    def tie_weights(self, *args, **kwargs):
        """Delegate weight tying while preserving the Transformers API."""
        return self.language_model.tie_weights(*args, **kwargs)

    def resize_token_embeddings(self, new_num_tokens: int | None = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(
        self,
        image_features: list[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        Args:
            image_features (:obj:`torch.Tensor` of shape :obj:`(num_image_tokens, embed_dim)`):
                The image features to merge with the input embeddings.
            inputs_embeds (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length, embed_dim)`):
                The input embeddings.
            input_ids (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                The input ids.
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                The attention mask.
            labels (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, *optional*):
                The labels.
        """
        image_token_index: int = self.config.media_placeholder_token_id
        pad_token_id: int = self.config.pad_token_id
        ignore_index: int = self.config.ignore_index

        # if self.training:
        #     pad_mask = input_ids == pad_token_id
        #     image_mask = input_ids == image_token_index
        #     image_features = torch.cat(image_features, dim=0)

        #     n_image_tokens = image_mask.sum()
        #     if image_features is not None and n_image_tokens != image_features.shape[0]:
        #         raise ValueError(
        #             f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
        #         )

        #     inputs_embeds[image_mask] = image_features
        #     inputs_embeds *= ~pad_mask.unsqueeze(-1)
        #     position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_(attention_mask == 0, 1)

        #     return inputs_embeds, attention_mask, labels, position_ids

        _, embed_dim = image_features[0].shape
        feature_lengths = [x.shape[0] for x in image_features]
        image_features = torch.cat(image_features, dim=0)
        image_token_mask = input_ids == image_token_index
        image_token_count = image_token_mask.sum().item()
        if image_token_count != len(feature_lengths):
            raise ValueError(
                "Kimi-K3 expects one media placeholder token per image before "
                f"model-side expansion, but got {image_token_count} placeholders "
                f"for {len(feature_lengths)} images."
            )

        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))

        # 1. Create a mask to know where special image tokens are
        _token_occupation_table = torch.ones_like(input_ids.flatten())
        _token_occupation_table[input_ids.flatten() == image_token_index] = torch.tensor(
            feature_lengths, dtype=torch.long, device=input_ids.device
        )
        _token_occupation_table = _token_occupation_table.reshape(input_ids.shape)

        max_embed_dim = _token_occupation_table.sum(-1).max().item()
        assert max_embed_dim >= sequence_length, (
            f"The maximum embedding dimension ({max_embed_dim}) is less than the sequence length ({sequence_length})"
        )
        batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        new_token_positions = torch.cumsum(_token_occupation_table, -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                ignore_index,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask.
        # The single-shot RHS read materializes a full copy of every text
        # embedding (14 GiB at 1M tokens, hidden 7168 bf16), so long multimodal
        # runs raise memory.mm_merge_chunk_tokens, which the training entry
        # stamps onto the model config before load. 0 (absent) keeps the
        # upstream path; a positive value moves the text embeddings that many
        # tokens at a time. Pure per-row data movement with no arithmetic, so
        # the batched writes are bitwise identical to the single-shot
        # assignment.
        mm_merge_chunk = getattr(self.config, "mm_merge_chunk_tokens", 0)
        if mm_merge_chunk > 0:
            for start in range(0, batch_indices.shape[0], mm_merge_chunk):
                stop = start + mm_merge_chunk
                final_embedding[batch_indices[start:stop], text_to_overwrite[start:stop]] = inputs_embeds[
                    batch_indices[start:stop], non_image_indices[start:stop]
                ]
        else:
            final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {image_to_overwrite.sum()} while"
                f" the number of image features given to the model is {image_features.shape[:-1].numel()}. "
                "This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        image_token_attention = attention_mask[image_token_mask].to(target_device)
        image_token_attention = torch.repeat_interleave(
            image_token_attention,
            torch.tensor(feature_lengths, device=target_device),
        )
        final_attention_mask[image_to_overwrite] = image_token_attention.to(final_attention_mask.dtype)
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def _extract_image_features(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            pixel_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_channels, height, width)`):
                The pixel values of the images processed by image processor.
            grid_thws (:obj:`torch.Tensor` of shape :obj:`(batch_size, 3)`):
                The grid, height, width of the images.

        Returns:
            selected_image_feature (:obj:`torch.FloatTensor` of shape :obj:`(num_image_tokens, embed_dim)`):
                The selected image features to use as input to the projector head.

        """

        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(target_dtype)

        image_features = self.vision_tower(pixel_values, grid_thws)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | list[torch.FloatTensor] | None = None,
        grid_thws: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | LlavaCausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        ```
        """
        assert self.vision_tower is not None, "vision_tower is not loaded"
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Chunk loss needs the full hidden-state sequence; force it on if enabled.
        enable_chunk_loss = getattr(self, "enable_chunk_loss", False)
        if enable_chunk_loss:
            output_hidden_states = True

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and len(pixel_values) > 0 and input_ids.shape[1] != 1:
                image_features = self._extract_image_features(pixel_values, grid_thws)
                if self.mm_projector:
                    image_features = self.mm_projector(image_features)

                inputs_embeds = inputs_embeds.to(image_features[0].dtype)  # num_tokens, embed_dim
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    labels,
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Modification start: chunk loss adaptation, following Kimi-K2.5 FSDP.
        # The language model returns hidden_states; the VLM top-level applies
        # lm_head so that the chunk loss path can be used.
        hidden_states = outputs.hidden_states

        if enable_chunk_loss:
            logits = None
            chunk_size = getattr(self, "chunk_size", 1024)
            if labels is None:
                raise ValueError("labels must be provided when chunk loss is enabled")

            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if attention_mask is not None:
                shift_labels = shift_labels.masked_fill(
                    attention_mask[..., 1:].to(shift_labels.device) == 0,
                    -100,
                )

            # Normalize every chunk by the total number of valid tokens so that
            # their accumulated loss matches the regular mean cross-entropy.
            num_valid_tokens = shift_labels.ne(-100).sum().clamp_min(1)
            loss_kwargs_chunks = [
                {
                    "shift_labels": labels_chunk,
                    "alpha": num_valid_tokens,
                    "reduction": "sum",
                }
                for labels_chunk in torch.split(shift_labels, chunk_size, dim=1)
            ]
            lm_head = self.language_model.lm_head
            loss = chunk_loss(
                shift_hidden_states,
                lm_head.weight,
                lm_head.bias,
                calculate_lm_loss,
                loss_kwargs_chunks,
                chunk_size,
            )
        else:
            logits = self.language_model.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                if attention_mask is not None:
                    shift_attention_mask = attention_mask[..., 1:]
                    shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                    shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1).to(shift_logits.device),
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        grid_thws=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                cache_length = past_key_values.get_seq_length()
                past_length = getattr(past_key_values, 'seen_tokens', cache_length)
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.media_placeholder_token_id in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)


# Hybrid CP model selection and subclass composition.


def build_kimi_k3_model_class(
    model_base: type[PreTrainedModel],
    kda_base: type[nn.Module],
    mla_base: type[nn.Module],
    *,
    kda_backend: str = "kcp",
    mla_backend: str = "ulysses",
) -> type[PreTrainedModel]:
    """Select a model class before weights are loaded, using one reference family.

    Backends must come from the validated CP configuration. Legacy modes return
    model_base; hybrid_p2p constructs its Hybrid CP subclasses. This function performs
    no checkpoint I/O, registration, config mutation or process-group setup.
    The application owns reference imports and weight loading in every mode.
    """
    if kda_backend == "hybrid_p2p":
        return build_kimi_k3_hybrid_cp_class(model_base, kda_base, mla_base, mla_backend=mla_backend)
    return model_base


def build_kimi_k3_hybrid_cp_class(
    model_base: type[PreTrainedModel],
    kda_base: type[nn.Module],
    mla_base: type[nn.Module],
    *,
    mla_backend: str = "ulysses",
) -> type[PreTrainedModel]:
    """Subclass application-supplied Transformers/remote-code Kimi classes.

    The caller owns reference-code loading. Construct this class *before* loading
    weights so component subclasses retain the original checkpoint parameter names.
    ``mla_backend`` comes from the validated CP configuration: Ulysses selects
    contiguous input shards; KV-AllGather selects front/back shards. The same
    choice is shared by model input, attention and loss. Build a class for the
    selected backend before loading weights. No imports from sibling model
    directories and no mutation of reference classes.
    """

    zigzag = mla_backend != "ulysses"

    class KimiHybridCPDeltaAttention(kda_base):
        """KDA subclass whose configured path exchanges heads and affine states."""

        def configure_hybrid_cp(self, runtime: HybridCPRuntime) -> str:
            """Validate head splitting before any model collective and bind groups."""
            size = dist.get_world_size(runtime.ulysses_group)
            if (
                self.num_heads % size
                or self.head_dim != self.head_k_dim
                or getattr(self, "num_k_heads", self.num_heads) != self.num_heads
            ):
                raise ValueError("Ulysses size must divide KDA num_heads; K/V head dimensions must agree.")
            self.cp_runtime = runtime
            return "kda"

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            cache_params: object | None = None,
            **kwargs: object,
        ) -> torch.Tensor:
            """Run CP training; cache and pre-packed sequences are excluded."""
            if cache_params is not None or kwargs.get("cu_seqlens") is not None or kwargs.get("use_cache"):
                raise ValueError("Hybrid P2P KDA supports training without cache or packed cu_seqlens.")
            attention_mask = kwargs.get("cp_padding_mask", attention_mask)
            runtime = self.cp_runtime
            if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
                raise ValueError("Hybrid P2P KDA requires [1, local_sequence, hidden].")
            if attention_mask is not None and attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("KDA attention_mask must be the local [batch, sequence] padding mask.")
            original = hidden_states
            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask[..., None]
            local_length = hidden_states.shape[1]
            partition = runtime.layout(local_length, zigzag=zigzag)
            ulysses_size = dist.get_world_size(runtime.ulysses_group)
            ulysses_rank = dist.get_rank(runtime.ulysses_group)
            local_heads = self.num_heads // ulysses_size
            head_start, head_end = ulysses_rank * local_heads, (ulysses_rank + 1) * local_heads
            state_size = partition.cp_size // ulysses_size
            state_rank = partition.rank // ulysses_size
            if partition.zigzag and local_length % 2:
                raise ValueError("Zigzag local sequence must contain two equal intervals; pad before sharding.")
            state_length = local_length * ulysses_size
            context = KdaCPContext(
                runtime.state_group,
                state_rank,
                state_size,
                state_length // (2 if partition.zigzag else 1),
                partition.zigzag,
            )
            query, key, value = (projection(hidden_states) for projection in (self.q_proj, self.k_proj, self.v_proj))
            decay = self.f_b_proj(self.f_a_proj(hidden_states))
            beta = self.b_proj(hidden_states).float().sigmoid()
            query, key, value, decay, beta = (
                partition.reorder_ulysses(ulysses_all_to_all(tensor, runtime.ulysses_group, 2, 1), ulysses_size)
                for tensor in (query, key, value, decay, beta)
            )
            query, key, value = (
                self._cp_convolution(
                    tensor,
                    conv,
                    context,
                    head_start * self.head_dim,
                    head_end * self.head_dim,
                )
                for tensor, conv in (
                    (query, self.q_conv1d),
                    (key, self.k_conv1d),
                    (value, self.v_conv1d),
                )
            )
            shape = (1, state_length, local_heads, self.head_dim)
            output = chunk_kda_p2p(
                query.reshape(shape),
                key.reshape(shape),
                value.reshape(shape),
                decay.reshape(shape),
                beta,
                self.A_log[head_start:head_end],
                self.dt_bias.reshape(self.num_heads, -1)[head_start:head_end].reshape(-1),
                state_context=context,
                lower_bound=self.gate_lower_bound,
            ).flatten(2)
            output = partition.reorder_ulysses(output, ulysses_size, inverse=True)
            output = ulysses_all_to_all(output, runtime.ulysses_group, 1, 2).reshape(
                1, local_length, self.num_heads, self.head_dim
            )
            gate = (
                self.g_proj(hidden_states) if self.use_full_rank_gate else self.g_b_proj(self.g_a_proj(hidden_states))
            )
            output = self.o_proj(self.o_norm(output, gate.reshape_as(output)).flatten(2))
            return output if attention_mask is None else output * attention_mask[..., None].to(original.dtype)

        def _cp_convolution(
            self,
            projected: torch.Tensor,
            conv: nn.Module,
            context: KdaCPContext,
            start: int,
            end: int,
        ) -> torch.Tensor:
            """Apply this model convolution to intervals with exchanged history."""
            weight = conv.weight[start:end].squeeze(1)
            bias = None if conv.bias is None else conv.bias[start:end]
            parts = torch.stack(projected[0].chunk(len(context.slots)))
            prefix = exchange_causal_history(parts, weight.shape[-1] - 1, context)
            inputs = torch.cat((prefix, parts), dim=1) if prefix.shape[1] else parts
            output = F.conv1d(inputs.transpose(1, 2), weight.unsqueeze(1), bias, groups=weight.shape[0]).transpose(1, 2)
            if conv.activation in ("silu", "swish"):
                output = F.silu(output)
            return output.flatten(0, 1).unsqueeze(0)

    class KimiHybridCPMLAAttention(mla_base):
        """MLA subclass using the complete CP domain and shared token coordinates."""

        def configure_hybrid_cp(self, runtime: HybridCPRuntime) -> str:
            """Bind the selected implemented MLA backend."""
            if mla_backend == "ulysses" and self.num_heads % dist.get_world_size(runtime.cp_group):
                raise ValueError("MLA Ulysses requires num_attention_heads divisible by the complete CP size.")
            self.cp_runtime = runtime
            return "mla"

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: object | None = None,
            **kwargs: object,
        ) -> torch.Tensor:
            """Compute global causal MLA attention from locally projected Q/K/V."""
            if past_key_values is not None:
                raise ValueError("Hybrid P2P MLA does not support cached decoding.")
            padding_mask = kwargs.get("cp_padding_mask")
            runtime = self.cp_runtime
            batch, length = hidden_states.shape[:2]
            partition = runtime.layout(length, zigzag=zigzag)
            query = (
                self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
                if self.q_lora_rank is not None
                else self.q_proj(hidden_states)
            )
            query = query.reshape(batch, length, self.num_heads, self.q_head_dim).transpose(1, 2)
            compressed = self.kv_a_proj_with_mqa(hidden_states)
            compressed, rotary_key = compressed.split((self.kv_lora_rank, self.qk_rope_head_dim), -1)
            expanded = (
                self.kv_b_proj(self.kv_a_layernorm(compressed))
                .reshape(batch, length, self.num_heads, -1)
                .transpose(1, 2)
            )
            key, value = expanded.split((self.qk_nope_head_dim, self.v_head_dim), -1)
            key = torch.cat((key, rotary_key[:, None].expand(-1, self.num_heads, -1, -1)), -1)
            if padding_mask is None:
                padding_mask = torch.ones((batch, length), device=hidden_states.device, dtype=torch.bool)
            masks = [torch.empty_like(padding_mask) for _ in range(partition.cp_size)]
            dist.all_gather(masks, padding_mask.contiguous(), group=runtime.cp_group)
            global_valid = partition.restore_gathered(torch.cat(masks, 1), 1).bool()
            if mla_backend == "ulysses":
                query, key, value = (
                    ulysses_all_to_all(tensor, runtime.cp_group, 1, 2) for tensor in (query, key, value)
                )
                query_positions = torch.arange(partition.padded_length, device=hidden_states.device)
            else:
                key, value = (
                    partition.restore_gathered(kv_allgather(tensor, runtime.cp_group, 2), 2) for tensor in (key, value)
                )
                query_positions = torch.cat(
                    [torch.arange(lo, hi, device=hidden_states.device) for lo, hi in partition.owned_intervals]
                )
            causal_fast_path = mla_backend == "ulysses" and bool(global_valid.all())
            mask = None
            if not causal_fast_path:
                key_positions = torch.arange(partition.padded_length, device=hidden_states.device)
                allowed = (query_positions[:, None] >= key_positions[None, :])[None, None] & global_valid[
                    :, None, None, :
                ]
                mask = torch.zeros(allowed.shape, device=query.device, dtype=query.dtype).masked_fill(
                    ~allowed, torch.finfo(query.dtype).min
                )
            original_value_dim = value.shape[-1]
            if value.shape[-1] < query.shape[-1]:
                value = F.pad(value, (0, query.shape[-1] - value.shape[-1]))
            output = flash_attention_forward(
                self,
                query,
                key,
                value,
                mask,
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
                is_causal=causal_fast_path,
            )[0][..., :original_value_dim]
            if mla_backend == "ulysses":
                output = ulysses_all_to_all(output, runtime.cp_group, 1, 2)
            output = output.reshape(batch, length, -1)
            if self.use_output_gate:
                output = output * self.g_proj(hidden_states).sigmoid()
            return self.o_proj(output) * padding_mask[..., None]

    class KimiK3HybridCP(model_base):
        """Fuse globally, shard once, and run KDA/MLA subclasses with CP loss."""

        def __init__(self, config: object) -> None:
            super().__init__(config)
            for layer in self.language_model.model.layers:
                attention = layer.self_attn
                adapter = KimiHybridCPDeltaAttention if isinstance(attention, kda_base) else KimiHybridCPMLAAttention
                layer.self_attn = adapter(config.text_config, attention.layer_idx)

        def configure_hybrid_cp(self, runtime: HybridCPRuntime) -> str:
            """Bind post-fusion input sharding and locally normalized model loss."""
            if getattr(self, "enable_chunk_loss", False):
                raise ValueError("Hybrid P2P Kimi K3 does not support chunk loss.")
            self.cp_runtime = runtime
            return "model"

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            pixel_values: torch.Tensor | list[torch.Tensor] | None = None,
            grid_thws: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            past_key_values: object | None = None,
            inputs_embeds: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            use_cache: bool | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            cache_position: torch.Tensor | None = None,
            num_items_in_batch: torch.Tensor | None = None,
            **kwargs: object,
        ) -> LlavaCausalLMOutputWithPast | tuple:
            """Fuse images before CP; labels stay global until shifted in the loss."""
            if past_key_values is not None or cache_position is not None or use_cache:
                raise ValueError("Hybrid P2P Kimi K3 supports training without cache inputs.")
            if kwargs.get("cu_seqlens") is not None:
                raise ValueError("Packed input sequences are not supported by this adapter.")
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
                if pixel_values is not None and len(pixel_values):
                    features = self._extract_image_features(pixel_values, grid_thws)
                    if self.mm_projector:
                        features = self.mm_projector(features)
                    inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                        features,
                        inputs_embeds.to(features[0].dtype),
                        input_ids,
                        attention_mask,
                        labels,
                    )
            elif pixel_values is not None and len(pixel_values):
                raise ValueError("pixel_values cannot be merged with supplied inputs_embeds.")
            if inputs_embeds.shape[0] != 1:
                raise ValueError("Hybrid P2P Kimi K3 requires batch size 1.")
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )
            valid_mask = attention_mask[0].bool()
            valid_segments = valid_mask[0].int() + (valid_mask[1:] & ~valid_mask[:-1]).sum()
            if valid_segments > 1:
                raise ValueError(
                    "KDA P2P supports a single valid token interval with left/right padding, not interior holes."
                )
            if labels is not None:
                labels = labels.masked_fill(attention_mask.to(labels.device) == 0, self.config.ignore_index)
            if position_ids is None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
            runtime = self.cp_runtime
            inputs_embeds, position_ids, attention_mask = shard_fused_sequence_inputs(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                group=runtime.cp_group,
                zigzag=zigzag,
            )
            # Call the backbone directly: the reference LM wrapper drops **kwargs.
            outputs = self.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cp_padding_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            logits = self.language_model.lm_head(outputs.last_hidden_state).float()
            loss = (
                None
                if labels is None
                else cp_compute_lm_loss(
                    logits,
                    labels,
                    runtime.cp_group,
                    num_items_in_batch=num_items_in_batch,
                    ignore_index=self.config.ignore_index,
                    zigzag=zigzag,
                )
            )
            if return_dict is False:
                result = (
                    logits,
                    outputs.past_key_values,
                    outputs.hidden_states,
                    outputs.attentions,
                )
                return (loss,) + result if loss is not None else result
            return LlavaCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    return KimiK3HybridCP

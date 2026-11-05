# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
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


"""Multi-Token Prediction (MTP) block.

Generic across models: the decoder layer, norm and hyper-connection mixer
classes are injected by the caller. Context-parallel awareness is likewise
injected — the caller passes the Ulysses group context (``ulysses_context``,
the ``(group, rank, size)`` tuple from ``get_ulysses_context()``) and the CP
group (``cp_context``, a ``(group, size)`` tuple) via forward kwargs, keeping
this module free of ``fsdp_turbo.distributed`` imports:

- The MindSpeed-MM CP communication primitives
  (``split_forward_gather_backward_with_cp`` / ``gather_forward_split_backward``)
  are reimplemented on top of plain ``torch.distributed`` so MTP keeps working
  when Ulysses CP is on, without pulling in MindSpeed-MM's communication module.
"""

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


def _cal_split_sizes(dim_size: int, world_size: int):
    """Even split sizes with the remainder distributed to the first ranks."""
    split_size = dim_size // world_size
    remainder = dim_size % world_size
    return [split_size + (1 if i < remainder else 0) for i in range(world_size)]


def _split_forward_gather_backward(input_, group, dim, split_sizes=None):
    """Forward: take this rank's shard. Backward: all-gather gradients.

    A thin autograd wrapper around ``ulysses_all_to_all`` that splits on ``dim``
    in the forward pass and all-gathers in the backward pass, mirroring
    MindSpeed-MM's ``split_forward_gather_backward``.
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    class _SplitGather(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, group, dim, split_sizes):
            ctx.group = group
            ctx.dim = dim
            ctx.split_sizes = split_sizes
            if split_sizes is None:
                split_sizes = _cal_split_sizes(x.size(dim), world_size)
            rank = dist.get_rank(group)
            start = sum(split_sizes[:rank])
            end = start + split_sizes[rank]
            return x.narrow(dim, start, end - start).contiguous()

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = torch.empty(
                *grad_output.shape[:dim],
                sum(ctx.split_sizes) if ctx.split_sizes else grad_output.size(dim) * world_size,
                *grad_output.shape[dim + 1 :],
                device=grad_output.device,
                dtype=grad_output.dtype,
            )
            dist.all_gather_into_tensor(grad_input, grad_output.contiguous(), group=ctx.group)
            return grad_input, None, None, None

    return _SplitGather.apply(input_, group, dim, split_sizes)


def _gather_forward_split_backward(input_, group, dim, gather_sizes=None, grad_scale="up"):
    """Forward: all-gather. Backward: take this rank's shard (optionally scaled).

    Mirrors MindSpeed-MM's ``gather_forward_split_backward``. ``grad_scale`` is
    accepted for API parity but only ``"up"`` (scale by world_size) is used here,
    matching the MTP caller's convention.
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    class _GatherSplit(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, group, dim, gather_sizes):
            ctx.group = group
            ctx.dim = dim
            ctx.gather_sizes = gather_sizes
            ctx.grad_scale = grad_scale
            if gather_sizes is None:
                total = x.size(dim) * world_size
                gather_sizes = _cal_split_sizes(total, world_size)
            output = torch.empty(
                *x.shape[:dim],
                sum(gather_sizes),
                *x.shape[dim + 1 :],
                device=x.device,
                dtype=x.dtype,
            )
            dist.all_gather_into_tensor(output, x.contiguous(), group=group)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.gather_sizes is None:
                gather_sizes = _cal_split_sizes(grad_output.size(dim), world_size)
            else:
                gather_sizes = ctx.gather_sizes
            rank = dist.get_rank(ctx.group)
            start = sum(gather_sizes[:rank])
            end = start + gather_sizes[rank]
            grad_input = grad_output.narrow(ctx.dim, start, end - start).contiguous()
            if ctx.grad_scale == "up":
                grad_input = grad_input * world_size
            return grad_input, None, None, None

    return _GatherSplit.apply(input_, group, dim, gather_sizes)


def shift_tensor(tensor, shifts=-1, dims=-1, fill_value=0, cu_seqlens=None):
    """Shift a tensor and clear positions that would cross packed boundaries."""
    if cu_seqlens is not None:
        shifted = torch.roll(tensor.reshape(-1), shifts=shifts, dims=0)
        shifted.index_fill_(0, cu_seqlens[1:] - 1, fill_value)
        return shifted.view_as(tensor)

    shifted = torch.roll(tensor, shifts=shifts, dims=dims)
    shifted.select(dims, shifts).fill_(fill_value)
    return shifted


class _MTPPerTokenLoss(torch.autograd.Function):
    """Return the MTP mean loss while applying Megatron's per-token gradient scale."""

    @staticmethod
    def forward(ctx, loss_sum, original_num_tokens, mtp_num_tokens, global_avg_num_tokens):
        ctx.save_for_backward(original_num_tokens, mtp_num_tokens, global_avg_num_tokens)
        return loss_sum / mtp_num_tokens.to(loss_sum.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (original_num_tokens, mtp_num_tokens, global_avg_num_tokens) = ctx.saved_tensors
        grad_scale = (
            original_num_tokens.to(grad_output.dtype)
            / mtp_num_tokens.to(grad_output.dtype)
            / global_avg_num_tokens.to(grad_output.dtype)
        )
        return grad_output * grad_scale, None, None, None


class MultiTokenPredictionBlock(nn.Module):
    def __init__(self, config, layer_cls, norm_cls, hyper_connection_mixer_cls=None):
        super().__init__()
        self.config = config
        num_physical_layers = int(config.mtp_num_layers > 0)
        self.layers = nn.ModuleList(
            [layer_cls(config, i, is_mtp=True) for i in range(num_physical_layers)]
            if "is_mtp" in layer_cls.__init__.__code__.co_varnames
            else [layer_cls(config, i) for i in range(num_physical_layers)]
        )
        self.pre_fc_norm_embedding = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

        # Qwen4Exp (Qwen3.8-Flash-Next) uses a hyper-connection scheme where
        # decoder layers operate on `hc_count * hidden_size` features. The
        # stock `Qwen4ExpTextModel.forward` expands hidden_states via
        # `.repeat(1, 1, hc_count)` before the decoder stack and collapses
        # them back via `hyper_connection_mixer` afterwards. MTP calls the
        # decoder layer directly, so it must replicate this expand/collapse
        # or the layer's `Qwen4ExpTextGatedResidual` raises:
        #   "Expected {hc_count*hidden_size} hyper-connection features, got {hidden_size}"
        # Models without `hc_count` (e.g. Qwen3.5-MoE) skip this entirely.
        #
        # The mixer is created here as an independent module (NOT a shared
        # reference to the language model's mixer). Under FSDP2, a shared
        # reference would remain governed by the `language_model` FSDP unit:
        # MTP calls it outside that unit's forward, so FSDP2 never
        # all-gathers its parameters, leaving them as DTensor shards and
        # causing "mixed torch.Tensor and DTensor" errors. Owning the module
        # makes it part of the `mtp` FSDP unit, whose forward triggers the
        # all-gather correctly. MTP weights are all newly initialized
        # (missing from checkpoint), so an independently initialized mixer
        # is consistent with the rest of the MTP block.
        self.hc_count = getattr(config, "hc_count", None)
        if self.hc_count is not None and self.hc_count <= 1:
            self.hc_count = None
        if self.hc_count is not None and hyper_connection_mixer_cls is not None:
            self.hyper_connection_mixer = hyper_connection_mixer_cls(config, use_combine=False)
        else:
            self.hyper_connection_mixer = None
            if self.hc_count is not None:
                raise RuntimeError(
                    "hc_count is set but no hyper_connection_mixer_cls provided; "
                    "cannot collapse hyper-connection features without a mixer."
                )

    def _prepare_position_ids(
        self,
        position_ids: torch.LongTensor | None,
        inputs_embeds: torch.Tensor,
        past_key_values: Cache | None,
        cache_position: torch.LongTensor | None,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        return position_ids, text_position_ids, cache_position

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        embed_tokens: nn.Module | None = None,
        rotary_emb: nn.Module | None = None,
        output_layer: nn.Module | None = None,
        loss_function: Callable | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        ulysses_context: tuple | None = None,
        cp_context: tuple | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if not self.layers:
            return None
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        all_mtp_loss = None
        calculate_per_token_loss = hasattr(loss_function, "avg_per_step_token_num")
        if calculate_per_token_loss:
            avg_per_step_token_num = getattr(loss_function, "avg_per_step_token_num", None)
            if avg_per_step_token_num is None:
                raise KeyError("per_token_loss must use PrefetchGradAccDataLoader")
            original_num_tokens = (labels > -1).sum()
            global_avg_num_tokens = avg_per_step_token_num.detach().clone()
            torch.distributed.all_reduce(global_avg_num_tokens, op=torch.distributed.ReduceOp.AVG)

        # Parallel context is injected by the caller (see the module docstring):
        # ulysses_context is the (group, rank, size) tuple from
        # get_ulysses_context(); cp_context is the (group, size) tuple of the
        # CP group. Both default to disabled.
        ulysses_group, _ulysses_rank, ulysses_size = ulysses_context or (None, 0, 1)
        use_ulysses = ulysses_size > 1
        use_packing = "cu_seqlens" in kwargs and kwargs["cu_seqlens"] is not None
        cu_seqlens = None
        if use_packing:
            text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
            if use_ulysses:
                cu_seqlens = _generate_ulysses_cu_seqlen_params(text_position_ids)
            else:
                cu_seqlens = None
            if cu_seqlens is not None:
                kwargs["cu_seqlens"] = cu_seqlens.squeeze(0).to(device=input_ids.device, dtype=torch.long)

        for _ in range(self.config.mtp_num_layers):
            decoder_layer = self.layers[0]
            input_ids = shift_tensor(input_ids, cu_seqlens=cu_seqlens)
            labels = shift_tensor(labels, fill_value=-100, cu_seqlens=cu_seqlens)
            if cu_seqlens is not None:
                # The shared causal loss shifts once more; mask starts to protect the preceding sequence.
                labels.view(-1).index_fill_(0, cu_seqlens[1:-1], -100)
            inputs_embeds = embed_tokens(input_ids)

            position_ids, text_position_ids, cache_position = self._prepare_position_ids(
                position_ids, inputs_embeds, past_key_values, cache_position
            )

            if use_ulysses:
                pad_len = (ulysses_size - inputs_embeds.size(1) % ulysses_size) % ulysses_size
                if pad_len > 0:
                    inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, pad_len), value=0)
                    if attention_mask is not None:
                        attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
                    if text_position_ids is not None:
                        text_position_ids = F.pad(text_position_ids, (0, pad_len), value=0)
                    if position_ids is not None:
                        position_ids = F.pad(position_ids, (0, pad_len), value=0)
                    if labels is not None:
                        labels = F.pad(labels, (0, pad_len), value=-100)

            if use_packing:
                causal_mask = None
            else:
                causal_mask = create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                    # Same rule as Qwen4Exp.create_masks_for_generate: the QSA
                    # indexer inside each decoder layer needs a materialized
                    # mask, so SDPA's is_causal shortcut must not skip it.
                    allow_is_causal_skip=False,
                )

            if use_ulysses:
                if not use_packing:
                    generated = _generate_ulysses_cu_seqlen_params(text_position_ids)
                    if generated is not None:
                        kwargs["cu_seqlens"] = generated.squeeze(0).to(device=input_ids.device, dtype=torch.long)
                split_sizes = _cal_split_sizes(inputs_embeds.size(1), ulysses_size)
                position_ids = _split_forward_gather_backward(
                    position_ids, ulysses_group, dim=2, split_sizes=split_sizes
                )
                text_position_ids = _split_forward_gather_backward(
                    text_position_ids, ulysses_group, dim=1, split_sizes=split_sizes
                )
                inputs_embeds = _split_forward_gather_backward(
                    inputs_embeds, ulysses_group, dim=1, split_sizes=split_sizes
                )

            position_embeddings = rotary_emb(hidden_states, position_ids)

            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))

            # Qwen4Exp hyper-connection: decoder layers expect
            # `hc_count * hidden_size` features. Expand before the layer
            # (mirrors Qwen4ExpTextModel.forward line 1417) and collapse
            # afterwards via the shared hyper_connection_mixer (mirrors
            # line 1430). No-op for models without hyper-connections.
            if self.hc_count is not None:
                hidden_states = hidden_states.repeat(1, 1, self.hc_count)

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            if self.hc_count is not None:
                if self.hyper_connection_mixer is None:
                    raise RuntimeError(
                        f"MTP block has hc_count={self.hc_count} but hyper_connection_mixer is None; "
                        "pass hyper_connection_mixer_cls so hyper-connection features can be collapsed."
                    )
                hidden_states = self.hyper_connection_mixer(hidden_states)

            hidden_states = self.norm(hidden_states)

            if getattr(self.config, "enable_chunk_loss", False):
                mtp_loss = output_layer(hidden_states[:, slice_indices, :], loss_function, labels)
            else:
                mtp_logits = output_layer(hidden_states[:, slice_indices, :])
                if labels is not None:
                    mtp_loss = loss_function(mtp_logits, labels, vocab_size=self.config.vocab_size)
                else:
                    mtp_loss = None

            if all_mtp_loss is None:
                all_mtp_loss = []
            if calculate_per_token_loss:
                # The MTP loss function returns the local loss sum. Match
                # Megatron's N/M correction, then use the main loss denominator D.
                mtp_labels = shift_tensor(labels, shifts=-1, dims=-1, fill_value=-100)
                mtp_num_tokens = (mtp_labels > -1).sum().clamp(min=1)
                mtp_loss = _MTPPerTokenLoss.apply(
                    mtp_loss,
                    original_num_tokens,
                    mtp_num_tokens,
                    global_avg_num_tokens.clamp(min=1),
                )
            all_mtp_loss.append(mtp_loss)

            if use_ulysses:
                position_ids = _gather_forward_split_backward(
                    position_ids,
                    ulysses_group,
                    dim=2,
                    gather_sizes=None,
                    grad_scale="up",
                )
                text_position_ids = _gather_forward_split_backward(
                    text_position_ids,
                    ulysses_group,
                    dim=1,
                    gather_sizes=None,
                    grad_scale="up",
                )

        if all_mtp_loss is not None and cp_context is not None:
            cp_group, cp_group_size = cp_context
            for i in range(len(all_mtp_loss)):
                all_mtp_loss[i] = _gather_forward_split_backward(all_mtp_loss[i].unsqueeze(0), cp_group, dim=0)
                all_mtp_loss[i] = all_mtp_loss[i].sum() / cp_group_size
        return all_mtp_loss


def _generate_ulysses_cu_seqlen_params(position_ids, need_cpu_tensor=True):
    """Generate cu_seqlens from 2D position_ids for packed sequences.

    Ported from MindSpeed-MM ``context_parallel/utils.py``; returns ``None`` when
    position_ids is not 2D (no packing).
    """
    if position_ids is None or position_ids.dim() != 2:
        return None
    # position_ids: [batch, seq] — build cu_seqlens per batch row.
    batch = position_ids.size(0)
    seq = position_ids.size(1)
    # Treat each row as one segment; cu_seqlens is the prefix sum of segment lengths.
    lengths = torch.full((batch,), seq, device=position_ids.device, dtype=torch.long)
    cu_seqlens = torch.zeros(batch + 1, device=position_ids.device, dtype=torch.long)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    return cu_seqlens.unsqueeze(0)

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Attention wrappers for Ulysses context parallelism."""

from functools import wraps
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fsdp_turbo.distributed.context_parallel.communication import all_to_all
from fsdp_turbo.distributed.context_parallel.utils import _PREPARED_COMPRESSED_KV_KEY
from fsdp_turbo.distributed.context_parallel.kv_allgather.sparse_gather import sparse_gather_rows
from fsdp_turbo.distributed.context_parallel.kv_allgather.utils import (
    get_flattened_cp_context,
    get_kv_allgather_context,
    kv_allgather,
)
from fsdp_turbo.ops.qsa import (
    apply_query_rope,
    qsa_compress_block_keys,
    qsa_selected_token_mask,
)
from fsdp_turbo.distributed.context_parallel.ulysses.utils import ulysses_all_to_all


def _to_additive_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if attention_mask.dtype != torch.bool:
        return attention_mask
    additive_mask = torch.zeros_like(attention_mask, dtype=dtype)
    return additive_mask.masked_fill(~attention_mask, torch.finfo(dtype).min)


def _full_causal_mask_from_local(attention_mask: torch.Tensor | None, query, group, seq_dim: int):
    batch_size = query.shape[0] if attention_mask is None else attention_mask.shape[0]
    full_seq_len = query.shape[seq_dim] * dist.get_world_size(group)
    positions = torch.arange(full_seq_len, device=query.device)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
    mask = torch.zeros(
        (batch_size, 1, full_seq_len, full_seq_len),
        device=query.device,
        dtype=query.dtype,
    )
    return mask.masked_fill(~causal.view(1, 1, full_seq_len, full_seq_len), torch.finfo(query.dtype).min)


def ulysses_context_parallel_attention(
    attention_fn,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *args,
    ulysses_group,
    scatter_dim: int = 1,
    gather_dim: int = 2,
    flash_causal: bool = False,
    flash_attention: bool = False,
    total_seq_len: int | None = None,
    **kwargs,
):
    """Run full attention after exchanging sequence and head dimensions."""
    ulysses_size = dist.get_world_size(ulysses_group)
    is_multi_dimensional = query.dim() > 3

    if flash_causal:
        # Flash kernels carry causality natively (sparse_mode=3 / is_causal);
        # materializing the [batch, 1, S, S] dense mask instead would cost
        # ~2 TiB at 1M tokens. Right padding stays excluded from the loss via
        # ignore_index, matching the dense-mask path's semantics.
        kwargs["attention_mask"] = None
        kwargs["is_causal"] = True
    elif flash_attention:
        attention_mask = kwargs.get("attention_mask")
        # HF FA2 consumes BHSD Q/K/V, a 2D padding mask and returns BSHD.
        # All ranks must gather a mask even if HF elided an all-valid local mask.
        # Visual varlen attention uses global cu_seqlens instead of a padding
        # mask; supplying a mask would override those frame boundaries in FA2.
        if kwargs.get("cu_seq_lens_q") is None:
            if attention_mask is None:
                attention_mask = torch.ones(
                    query.shape[0], query.shape[gather_dim], device=query.device, dtype=torch.bool
                )
            attention_mask = attention_mask.to(dtype=torch.bool)
            gathered_masks = [torch.empty_like(attention_mask) for _ in range(ulysses_size)]
            dist.all_gather(gathered_masks, attention_mask.contiguous(), group=ulysses_group)
            kwargs["attention_mask"] = torch.cat(gathered_masks, dim=-1)
        if key.shape[scatter_dim] % ulysses_size:
            repeats = query.shape[scatter_dim] // key.shape[scatter_dim]
            key = key.repeat_interleave(repeats, dim=scatter_dim)
            value = value.repeat_interleave(repeats, dim=scatter_dim)
    else:
        attention_mask = kwargs.get("attention_mask")
        if is_multi_dimensional and attention_mask is None and kwargs.get("is_causal", True) is not False:
            attention_mask = _full_causal_mask_from_local(None, query, ulysses_group, gather_dim)
            kwargs["attention_mask"] = attention_mask
        if (
            attention_mask is not None
            and attention_mask.dim() > 2
            and attention_mask.shape[-2] == query.shape[gather_dim]
            and attention_mask.shape[-1] == key.shape[gather_dim]
        ):
            attention_mask = _full_causal_mask_from_local(attention_mask, query, ulysses_group, gather_dim)
            kwargs["attention_mask"] = attention_mask
        if (
            attention_mask is not None
            and attention_mask.dim() > 2
            and attention_mask.shape[-2] == query.shape[gather_dim]
        ):
            gathered_masks = [torch.empty_like(attention_mask) for _ in range(ulysses_size)]
            dist.all_gather(gathered_masks, attention_mask.contiguous(), group=ulysses_group)
            kwargs["attention_mask"] = torch.cat(gathered_masks, dim=-2).contiguous()

    query = all_to_all(
        query,
        ulysses_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
        gather_size=total_seq_len,
    )
    key = all_to_all(
        key,
        ulysses_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
        gather_size=total_seq_len,
    )
    value = all_to_all(
        value,
        ulysses_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
        gather_size=total_seq_len,
    )

    if is_multi_dimensional and not flash_causal and not flash_attention:
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None and attention_mask.dim() > 2:
            kwargs["attention_mask"] = _to_additive_attention_mask(attention_mask, query.dtype)
            kwargs["is_causal"] = False

    if flash_causal:
        # Signals the flash kernel that the Ulysses all-to-all already ran.
        kwargs["_fsdp_turbo_cp_exchanged"] = True
    output = attention_fn(query, key, value, *args, **kwargs)
    if isinstance(output, tuple):
        attn_output, *rest = output
    else:
        attn_output, rest = output, []

    attn_output = all_to_all(
        attn_output,
        ulysses_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
        gather_size=None,
    )
    return (attn_output, *rest) if isinstance(output, tuple) else attn_output


def get_ulysses_attention_fn(fn: Callable, group: dist.ProcessGroup):
    """Wrap a standard Transformers attention callable with Ulysses."""

    @wraps(fn)
    def wrapper(  # pylint: disable=keyword-arg-before-vararg
        module, query, key, value, attention_mask=None, *args, **kwargs
    ):
        config = module.config
        num_key_value_groups = getattr(config, "num_key_value_groups", None)

        # Expand GQA KV heads when they cannot be split across the Ulysses group;
        # neutralize the downstream repeat_kv (module.num_key_value_groups = 1 is a no-op).
        kv_heads = getattr(config, "num_key_value_heads", None)
        ulysses_size = dist.get_world_size(group)
        saved_kv_groups = None
        if kv_heads is not None and kv_heads % ulysses_size != 0:
            kv_expand_repeats = int(config.num_attention_heads / kv_heads)
            key = torch.repeat_interleave(key, dim=1, repeats=kv_expand_repeats)
            value = torch.repeat_interleave(value, dim=1, repeats=kv_expand_repeats)
            if hasattr(module, "num_key_value_groups"):
                saved_kv_groups = module.num_key_value_groups
                module.num_key_value_groups = 1

        def attention_fn(q, k, v, *inner_args, **inner_kwargs):
            return fn(module, q, k, v, *inner_args, **inner_kwargs)

        # Flash kernels tagged _fsdp_turbo_flash_causal take is_causal and skip
        # the dense-mask reconstruction above (the Kimi dispatcher carries the
        # marker because it always selects the flash kernel under CP). The
        # marker and the CP_EXCHANGED_KWARG receipt are defined in
        # fsdp_turbo.models.kimi.kimi_flash_attention; they are matched by
        # their literal names here because distributed code must not import
        # the model package.
        flash_causal = getattr(fn, "_fsdp_turbo_flash_causal", False)

        try:
            return ulysses_context_parallel_attention(
                attention_fn,
                query,
                key,
                value,
                *args,
                attention_mask=attention_mask,
                ulysses_group=group,
                num_key_value_groups=num_key_value_groups,
                flash_causal=flash_causal,
                **kwargs,
            )
        finally:
            if saved_kv_groups is not None:
                module.num_key_value_groups = saved_kv_groups

    return wrapper


def get_ulysses_compressor_attention_fn(
    fn: Callable,
    group: dist.ProcessGroup,
    attn_output_transposed: bool = False,
):
    """Wrap compressor attention whose KV contains sliding and compressed parts."""

    @wraps(fn)
    def wrapper(module, query, key, value, *args, **kwargs):
        ulysses_size = dist.get_world_size(group)
        shared_kv = key is value
        has_prepared_compressed_kv = _PREPARED_COMPRESSED_KV_KEY in kwargs
        prepared_compressed_kv = kwargs.pop(_PREPARED_COMPRESSED_KV_KEY, None)
        num_groups = int(module.config.num_attention_heads / module.config.num_key_value_heads)
        saved_kv_groups = None
        saved_sinks = None
        sinks_overridden = False
        if num_groups > 1 or ulysses_size > module.config.num_key_value_heads:
            key = torch.repeat_interleave(key, dim=1, repeats=num_groups)
            value = key if shared_kv else torch.repeat_interleave(value, dim=1, repeats=num_groups)
            if hasattr(module, "num_key_value_groups"):
                saved_kv_groups = module.num_key_value_groups
                module.num_key_value_groups = 1

        try:
            if ulysses_size > 1:
                query = ulysses_all_to_all(query, group, scatter_dim=1, gather_dim=2)
                if not has_prepared_compressed_kv:
                    sliding_len = query.shape[2] // ulysses_size
                    compressed_len = key.shape[2] - sliding_len
                    sliding_key, compressed_key = key.split([sliding_len, compressed_len], dim=2)
                    sliding_key = ulysses_all_to_all(sliding_key, group, scatter_dim=1, gather_dim=2)
                    compressed_key = ulysses_all_to_all(compressed_key, group, scatter_dim=1, gather_dim=2)
                    key = torch.cat([sliding_key, compressed_key], dim=2)

                    if shared_kv:
                        value = key
                    else:
                        sliding_value, compressed_value = value.split([sliding_len, compressed_len], dim=2)
                        sliding_value = ulysses_all_to_all(sliding_value, group, scatter_dim=1, gather_dim=2)
                        compressed_value = ulysses_all_to_all(compressed_value, group, scatter_dim=1, gather_dim=2)
                        value = torch.cat([sliding_value, compressed_value], dim=2)
                else:
                    key = ulysses_all_to_all(key, group, scatter_dim=1, gather_dim=2)
                    value = key if shared_kv else ulysses_all_to_all(value, group, scatter_dim=1, gather_dim=2)

            if has_prepared_compressed_kv and prepared_compressed_kv is not None:
                if prepared_compressed_kv.shape[1] != key.shape[1]:
                    if prepared_compressed_kv.shape[1] != 1:
                        raise ValueError("Prepared DeepSeek-V4 compressed KV must have one head.")
                    prepared_compressed_kv = torch.repeat_interleave(
                        prepared_compressed_kv, dim=1, repeats=key.shape[1]
                    )
                key = torch.cat([key, prepared_compressed_kv], dim=2)
                value = key if shared_kv else torch.cat([value, prepared_compressed_kv], dim=2)

            s_aux = kwargs.get("s_aux", getattr(module, "sinks", None))
            if ulysses_size > 1 and s_aux is not None:
                head_start = dist.get_rank(group) * query.shape[1]
                kwargs["s_aux"] = s_aux.narrow(0, head_start, query.shape[1])
                if getattr(module, "sinks", None) is not None:
                    saved_sinks = module.sinks
                    delattr(module, "sinks")
                    module.sinks = kwargs["s_aux"]
                    sinks_overridden = True

            attn_output, attn_weights = fn(module, query, key, value, *args, **kwargs)
            if ulysses_size > 1:
                if attn_output_transposed:
                    attn_output = ulysses_all_to_all(attn_output, group, scatter_dim=1, gather_dim=2)
                else:
                    attn_output = ulysses_all_to_all(attn_output, group, scatter_dim=2, gather_dim=1)
            return attn_output, attn_weights
        finally:
            if sinks_overridden:
                delattr(module, "sinks")
                module.sinks = saved_sinks
            if saved_kv_groups is not None:
                module.num_key_value_groups = saved_kv_groups

    return wrapper


def get_ulysses_transposed_compressor_attention_fn(
    fn: Callable,
    group: dist.ProcessGroup,
):
    """Wrap compressor attention returning sequence before heads."""
    return get_ulysses_compressor_attention_fn(fn, group=group, attn_output_transposed=True)


def _gather_sequence(tensor, group):
    # Only non-differentiable indexer inputs / RoPE are gathered here. Q/K/V
    # use the autograd-capable exchanges below.
    shards = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(shards, tensor.contiguous(), group=group)
    return torch.cat(shards, dim=1)


def _causal_prefix_mask(validity, start, end):
    # Global-coordinate causal mask for query rows [start, end) over the
    # key prefix [0, end), intersected with the full padded validity mask.
    keys = torch.arange(end, device=validity.device)
    queries = torch.arange(start, end, device=validity.device)
    causal = keys[None, :] <= queries[:, None]
    return causal[None, None, :, :] & validity[:, None, None, :end].bool()


def get_qwen4_exp_ulysses_forward(original: Callable, group: dist.ProcessGroup) -> Callable:
    """Wrap Qwen4ExpTextAttention before its indexer loses global history.




    Requires a full, padded [B,S] validity mask supplied by the post-fusion
    model forward. Cache inputs and arbitrary 4-D/packed masks are unsupported.
    """

    @wraps(original)
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        kv_group, kv_rank, kv_size = get_kv_allgather_context()
        cp_group, _cp_rank, cp_size = get_flattened_cp_context()
        ulysses_size = dist.get_world_size(group)
        if cp_size == 1:
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
        if kv_size * ulysses_size != cp_size:
            raise RuntimeError(
                f"CP topology mismatch: kvallgather ({kv_size}) x ulysses ({ulysses_size}) != cp ({cp_size})."
            )
        if past_key_values is not None:
            raise ValueError("Qwen4 QSA CP supports training without a KV cache only.")
        if self.config._attn_implementation != "sdpa":
            raise ValueError("Qwen4 QSA CP requires SDPA.")
        batch, local_length, _ = hidden_states.shape
        if batch != 1:
            raise ValueError("Qwen4 QSA CP currently requires batch size 1.")
        full_length = local_length * cp_size
        if attention_mask is None or attention_mask.shape != (batch, full_length):
            raise ValueError("Qwen4 QSA CP requires the full padded [batch, sequence] validity mask.")
        validity = attention_mask.bool()
        # Block keys are pooled per shard, so a 0 -> 1 transition inside the
        # validity mask would break the prefix-visible assumption of the
        # scorer (right padding only).
        if (validity[0, 1:] & ~validity[0, :-1]).any():
            raise ValueError("Qwen4 QSA CP requires a right-padded validity mask (no holes).")

        heads = self.config.num_attention_heads
        kv_heads = self.config.num_key_value_heads
        padded_heads = ((heads + ulysses_size - 1) // ulysses_size) * ulysses_size
        chunk_size = getattr(self.config, "qsa_cp_query_chunk_size", 128) or 128
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("qsa_cp_query_chunk_size must be a positive integer.")
        sparse_gather = bool(getattr(self.config, "qsa_sparse_gather", True))
        indexer = self.indexer
        if indexer.index_kv_heads != 1:
            raise ValueError("Qwen4 QSA CP currently requires one indexer KV head.")
        rate = indexer.compress_ratio
        if local_length % rate != 0:
            raise ValueError(
                f"Local sequence length ({local_length}) must be divisible by the QSA compress "
                f"ratio ({rate}); pad the sequence to a multiple of cp_size * compress_ratio."
            )
        # Query chunks stay block-aligned so the scorer's key prefix is exact.
        chunk_size = ((chunk_size + rate - 1) // rate) * rate
        from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

        seg_len = local_length * ulysses_size
        seg_start = kv_rank * seg_len
        seg_end = seg_start + seg_len
        ulysses_rank = dist.get_rank(group)
        local_heads = padded_heads // ulysses_size
        # Global (padded) head ids held by this rank's Ulysses shard, mapped to
        # their GQA KV head. Dummy padding heads clamp onto the last KV head;
        # their outputs are discarded before the gate and o_proj.
        head_map = (
            torch.arange(ulysses_rank * local_heads, (ulysses_rank + 1) * local_heads, device=hidden_states.device)
            // (heads // kv_heads)
        ).clamp_(max=kv_heads - 1)

        query, gate = self.q_proj(hidden_states).view(batch, local_length, heads, self.head_dim * 2).chunk(2, -1)
        gate = gate.reshape(batch, local_length, -1)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(batch, local_length, kv_heads, self.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, local_length, kv_heads, self.head_dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if padded_heads != heads:
            query = F.pad(query, (0, 0, 0, 0, 0, padded_heads - heads))
        # Only Q takes the head all-to-all; K/V keep their compact GQA form on
        # the originating CP shard and are gathered on demand below.
        query = ulysses_all_to_all(query, group, scatter_dim=1, gather_dim=2)

        with torch.no_grad():
            # Segment inputs for the indexer query side (== this rank's query
            # rows; the whole sequence when kvallgather is disabled).
            seg_hidden = _gather_sequence(hidden_states.detach(), group)
            seg_cos = _gather_sequence(position_embeddings[0].detach(), group)
            seg_sin = _gather_sequence(position_embeddings[1].detach(), group)
            # Compressed keys: pool locally, gather S/ratio rows per rank.
            query_width = indexer.index_n_heads * indexer.index_head_dim
            raw_keys = F.linear(hidden_states.detach(), indexer.index_qk_proj.weight[query_width:])
            block_keys = _gather_sequence(
                qsa_compress_block_keys(
                    raw_keys,
                    position_embeddings[0].detach(),
                    position_embeddings[1].detach(),
                    indexer.k_layernorm.weight,
                    rms_norm_eps=indexer.k_layernorm.eps,
                    compress_ratio=rate,
                ),
                cp_group,
            )

        key_full = value_full = None
        if not sparse_gather:
            key_full = kv_allgather(key, cp_group, gather_dim=2)
            value_full = kv_allgather(value, cp_group, gather_dim=2)

        def indexer_queries(start, end):
            rows = end - start
            projected = F.linear(
                seg_hidden[:, start - seg_start : end - seg_start],
                indexer.index_qk_proj.weight[:query_width],
            )
            index_q = projected.view(batch, rows, indexer.index_n_heads, indexer.index_head_dim)
            return apply_query_rope(
                indexer.q_layernorm(index_q),
                seg_cos[:, start - seg_start : end - seg_start],
                seg_sin[:, start - seg_start : end - seg_start],
            )

        def select(start, end, return_ids):
            return qsa_selected_token_mask(
                indexer_queries(start, end),
                None,
                None,
                None,
                _causal_prefix_mask(attention_mask, start, end),
                None,
                rms_norm_eps=indexer.k_layernorm.eps,
                token_budget=indexer.token_budget,
                compress_ratio=rate,
                block_keys=block_keys[:, : end // rate],
                return_token_ids=return_ids,
            )

        def attend_sparse(q, k, v, start, end):
            # Selection is a no-grad constant; only the selected-token gather
            # and SDPA participate in autograd (replayed under checkpoint).
            with torch.no_grad():
                sel_ids = select(start, end, return_ids=True)[0]
                width = sel_ids.shape[-1]
                flat = sel_ids.reshape(-1).to(torch.int64)
                keep = flat >= 0
                union_ids = torch.unique(flat[keep])
            n_union = union_ids.numel()
            if n_union == 0:
                return q.new_zeros(q.shape[0], q.shape[1], q.shape[2], v.shape[-1])
            k_sel, v_sel = sparse_gather_rows((k, v), union_ids, local_length, cp_group)
            k_sel = k_sel.index_select(1, head_map)
            v_sel = v_sel.index_select(1, head_map)
            with torch.no_grad():
                # Position of every selected token inside the gathered set.
                pos = torch.searchsorted(union_ids, flat.clamp(min=0))
                row = torch.arange(flat.shape[0], device=flat.device) // width
                rows = end - start
                chunk_mask = torch.zeros(rows, n_union + 1, dtype=torch.bool, device=q.device)
                chunk_mask.view(-1).index_fill_(0, (row * (n_union + 1) + pos)[keep], True)
                chunk_mask = chunk_mask[:, :n_union].unsqueeze(0).unsqueeze(0)
            return F.scaled_dot_product_attention(
                q,
                k_sel,
                v_sel,
                attn_mask=chunk_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
                scale=self.scaling,
            )

        def attend_dense(q, _k, _v, start, end):
            with torch.no_grad():
                mask = _causal_prefix_mask(attention_mask, start, end) & select(start, end, return_ids=False)
            k_chunk = key_full[:, :, :end].index_select(1, head_map)
            v_chunk = value_full[:, :, :end].index_select(1, head_map)
            return F.scaled_dot_product_attention(
                q,
                k_chunk,
                v_chunk,
                attn_mask=mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
                scale=self.scaling,
            )

        attend = attend_sparse if sparse_gather else attend_dense

        chunks = []
        for start in range(seg_start, seg_end, chunk_size):
            end = min(start + chunk_size, seg_end)

            # Bind offsets now: backward recomputation happens after this loop.
            def run_chunk(q, k, v, _start=start, _end=end):
                return attend(q, k, v, _start, _end)

            q_chunk = query[:, :, start - seg_start : end - seg_start]
            if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (query, key, value)):
                # Recreate the selection and replay its collectives during
                # backward instead of retaining gathered KV sets across all
                # chunks. Every rank replays the same deterministic sequence.
                output = checkpoint(run_chunk, q_chunk, key, value, use_reentrant=False)
            else:
                output = run_chunk(q_chunk, key, value)
            chunks.append(output)
        output = torch.cat(chunks, dim=2).transpose(1, 2).contiguous()
        output = ulysses_all_to_all(output, group, scatter_dim=1, gather_dim=2)
        output = output[:, :, :heads, :]
        output = output.reshape(batch, local_length, -1) * torch.sigmoid(gate)
        return self.o_proj(output), None

    forward._qwen4_qsa_ulysses = True
    return forward

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Attention wrappers for Ulysses context parallelism."""

from functools import wraps
from typing import Callable

import torch
import torch.distributed as dist

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
    **kwargs,
):
    """Run full attention after exchanging sequence and head dimensions."""
    ulysses_size = dist.get_world_size(ulysses_group)
    is_multi_dimensional = query.dim() > 3

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
    if attention_mask is not None and attention_mask.dim() > 2 and attention_mask.shape[-2] == query.shape[gather_dim]:
        gathered_masks = [torch.empty_like(attention_mask) for _ in range(ulysses_size)]
        dist.all_gather(gathered_masks, attention_mask.contiguous(), group=ulysses_group)
        kwargs["attention_mask"] = torch.cat(gathered_masks, dim=-2).contiguous()

    query = ulysses_all_to_all(
        query,
        ulysses_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
    )
    key = ulysses_all_to_all(
        key,
        ulysses_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
    )
    value = ulysses_all_to_all(
        value,
        ulysses_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
    )

    if is_multi_dimensional:
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None and attention_mask.dim() > 2:
            kwargs["attention_mask"] = _to_additive_attention_mask(attention_mask, query.dtype)
            kwargs["is_causal"] = False

    output = attention_fn(query, key, value, *args, **kwargs)
    if isinstance(output, tuple):
        attn_output, *rest = output
    else:
        attn_output, rest = output, []

    attn_output = ulysses_all_to_all(
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
        num_groups = int(module.config.num_attention_heads / module.config.num_key_value_heads)
        if num_groups > 1 or ulysses_size > module.config.num_key_value_heads:
            key = torch.repeat_interleave(key, dim=1, repeats=num_groups)
            value = key if shared_kv else torch.repeat_interleave(value, dim=1, repeats=num_groups)

        if ulysses_size > 1:
            sliding_len = query.shape[2]
            compressed_len = key.shape[2] - sliding_len
            sliding_key, compressed_key = key.split([sliding_len, compressed_len], dim=2)
            query = ulysses_all_to_all(query, group, scatter_dim=1, gather_dim=2)
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

        attn_output, attn_weights = fn(module, query, key, value, *args, **kwargs)
        if ulysses_size > 1:
            if attn_output_transposed:
                attn_output = ulysses_all_to_all(attn_output, group, scatter_dim=1, gather_dim=2)
            else:
                attn_output = ulysses_all_to_all(attn_output, group, scatter_dim=2, gather_dim=1)
        return attn_output, attn_weights

    return wrapper


def get_ulysses_transposed_compressor_attention_fn(
    fn: Callable,
    group: dist.ProcessGroup,
):
    """Wrap compressor attention returning sequence before heads."""
    return get_ulysses_compressor_attention_fn(fn, group=group, attn_output_transposed=True)

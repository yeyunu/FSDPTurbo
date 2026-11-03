# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""NPU flash attention via ``npu_fusion_attention``.

Query/key/value arrive in **BNSD** layout ``[B, H, S, D]``, or **1TND**
layout ``[1, T, H, D]`` when ``input_layout="1TND"`` is supplied.

The 1TND path follows MindSpeed-MM: single sequences use BSND, while packed
sequences use TND with cumulative sequence boundaries.

Mask strategy aligns with MindSpeed-MM:
  - ``is_causal=True``: ``sparse_mode=3`` (built-in causal).  When a 4-D
    ``attention_mask`` is supplied it is converted to a bool mask and passed
    alongside ``sparse_mode=3`` so packed-segment boundaries and padding are
    honoured; otherwise a cached 2-D upper-triangular placeholder is used.
  - ``is_causal=False`` with ``cu_seq_lens``: split into chunks, each
    processed with ``sparse_mode=0`` (bidirectional, no mask).
  - ``is_causal=False`` with a 4-D ``attention_mask``: ``sparse_mode=0`` with
    the converted bool mask.
"""

import torch

from fsdp_turbo.ops.registry import register_op

_ATTEN_MASK_CACHE: dict = {}


def _get_attn_mask(device, seq_len=2048):
    """Return a cached 2-D bool upper-triangular mask (True = ignore)."""
    if device not in _ATTEN_MASK_CACHE:
        _ATTEN_MASK_CACHE[device] = torch.triu(torch.ones([seq_len, seq_len], device=device), diagonal=1).bool()
    return _ATTEN_MASK_CACHE[device]


def _mask_to_bool(attention_mask):
    """Convert a 4-D additive-float mask to a bool mask (True = ignore).

    For packed sequences (``B == 1``) returns a 2-D ``[S, S]`` mask; otherwise
    returns the 4-D ``[B, 1, S, S]`` mask so multi-batch padding is honoured
    instead of being silently dropped.
    """
    mask = attention_mask
    if mask.dtype != torch.bool:
        mask = mask < 0
    else:
        mask = ~mask
    squeezed = mask.squeeze()
    if squeezed.dim() == 2 and squeezed.shape[0] == squeezed.shape[1]:
        return squeezed
    return mask


def _npu_fa_bnsd_raw(query, key, value, head_num, scaling, dropout, atten_mask, sparse_mode):
    """Single NPU ``npu_fusion_attention`` call in BNSD, keeps the BNSD layout."""
    import torch_npu

    return torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        head_num,
        "BNSD",
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
        scale=scaling,
        keep_prob=1.0 - dropout,
        inner_precise=0,
        sparse_mode=sparse_mode,
    )[0]


def _npu_fa_bnsd(query, key, value, head_num, scaling, dropout, atten_mask, sparse_mode):
    """Single NPU ``npu_fusion_attention`` call in BNSD, returns BSHD ``[B, S, H, D]``."""
    return _npu_fa_bnsd_raw(query, key, value, head_num, scaling, dropout, atten_mask, sparse_mode).transpose(1, 2)


@register_op("causal_flash_attention", "npu")
def causal_flash_attention_npu(query, key, value, *, scaling=None, dropout=0.0, is_causal=True):
    """NPU ``npu_fusion_attention`` with the kernel's native causal mode.

    ``sparse_mode=3`` is the kernel's bottom-right causal mode, so a query
    block against a longer K/V prefix stays correctly aligned without a dense
    mask. The mask content is ignored under ``sparse_mode=3``, so the cached
    2048x2048 bool placeholder keeps mask cost independent of the sequence
    length. Returns the output in BNSD layout.
    """
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    atten_mask = _get_attn_mask(query.device) if is_causal else None
    return _npu_fa_bnsd_raw(query, key, value, query.shape[1], scaling, dropout, atten_mask, 3 if is_causal else 0)


@register_op("flash_attention", "npu")
def flash_attention_npu(
    module,
    query,
    key,
    value,
    attention_mask=None,
    *,
    scaling=None,
    dropout=0.0,
    is_causal=True,
    **kwargs,
):
    """NPU ``npu_fusion_attention`` path for BNSD and 1TND inputs.

    Returns output in BSHD ``[B, S, H, D]``.
    """
    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    if kwargs.get("input_layout") == "1TND":
        import torch_npu

        cu_q = kwargs.get("cu_seq_lens_q")
        cu_k = kwargs.get("cu_seq_lens_k", cu_q)
        cu_q = cu_q.tolist() if isinstance(cu_q, torch.Tensor) else cu_q
        cu_k = cu_k.tolist() if isinstance(cu_k, torch.Tensor) else cu_k
        head_num = query.shape[2]
        layout = "TND"
        if (cu_q is None or len(cu_q) == 2) and (cu_k is None or len(cu_k) == 2):
            layout = "BSND"
            cu_q = cu_k = None
        else:
            query, key, value = (tensor.squeeze(0) for tensor in (query, key, value))

        atten_mask = attention_mask
        if atten_mask is None or atten_mask.ndim != 2 or atten_mask.shape[0] != atten_mask.shape[1]:
            atten_mask = _get_attn_mask(query.device) if is_causal else None
        attn_output = torch_npu.npu_fusion_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            head_num,
            layout,
            pse=None,
            padding_mask=None,
            atten_mask=atten_mask,
            actual_seq_qlen=cu_q,
            actual_seq_kvlen=cu_k,
            scale=scaling,
            keep_prob=1.0 - dropout,
            inner_precise=0,
            sparse_mode=3 if is_causal else 0,
        )[0]
        return (attn_output.unsqueeze(0) if layout == "TND" else attn_output), None

    head_num = query.shape[1]
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q")

    if cu_seq_lens_q is not None and not is_causal:
        if isinstance(cu_seq_lens_q, torch.Tensor):
            cu_q = cu_seq_lens_q.tolist()
        else:
            cu_q = list(cu_seq_lens_q)

        chunk_lens = [cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1)]

        if sum(chunk_lens) == query.shape[2]:
            q_chunks = torch.split(query, chunk_lens, dim=2)
            k_chunks = torch.split(key, chunk_lens, dim=2)
            v_chunks = torch.split(value, chunk_lens, dim=2)
            attn_output = torch.cat(
                [
                    _npu_fa_bnsd(qc, kc, vc, head_num, scaling, dropout, None, 0)
                    for qc, kc, vc in zip(q_chunks, k_chunks, v_chunks)
                ],
                dim=1,
            )
        else:
            attn_output = _npu_fa_bnsd(query, key, value, head_num, scaling, dropout, None, 0)
        return attn_output, None

    atten_mask = None
    sparse_mode = 3 if is_causal else 0

    if attention_mask is not None and attention_mask.dim() == 4:
        # Honour the caller's mask (packed-segment boundaries or padding)
        # alongside the built-in causal mask instead of silently dropping it.
        atten_mask = _mask_to_bool(attention_mask)

    if atten_mask is None and is_causal:
        atten_mask = _get_attn_mask(query.device)

    attn_output = _npu_fa_bnsd(query, key, value, head_num, scaling, dropout, atten_mask, sparse_mode)
    return attn_output, None

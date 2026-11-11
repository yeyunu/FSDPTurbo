# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Packed-TND attention over a context-parallel KV prefix.

The query is this rank's segment ``[q_lo, q_hi)``; the KV prefix is assembled
across the CP group by the injected ``TNDRuntime``. Compressed rows and index
keys are exchanged pre-RoPE, like the raw KV stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fsdp_turbo.ops.lightning_indexer import fused_lightning_indexer
from fsdp_turbo.ops.sparse_attention import indexed_sparse_attention

from .attention import Attention
from .indexer import (
    _compute_index_score,
    canonical_index_order,
    select_candidate_blocks,
)
from .layers import apply_rotary_emb


@dataclass
class _DocumentProjection:
    """Per-document tensors this rank owns before anything is exchanged."""

    segment: object  # distributed DocumentSegment (duck-typed)
    xs: torch.Tensor  # local token slice feeding the projections
    qr: torch.Tensor  # q LoRA output, reused by the indexer
    query: torch.Tensor  # [1, length, heads, head_dim], already rotated
    freqs_q: torch.Tensor  # this document's RoPE coefficients
    kv_pre: torch.Tensor  # pre-RoPE KV rows produced locally


class CPAttention(Attention):
    """Attention that reads the KV prefix from an injected TND runtime."""

    cp_runtime = None

    def forward(self, x: torch.Tensor, start_pos: int, **kwargs):
        if self.cp_runtime is None:
            return super().forward(x, start_pos, **kwargs)
        if self.cp_runtime.plan is None or not self.cp_runtime.plan.docs:
            # Missing plan means the trainer did not split; treat it as a config error.
            raise RuntimeError(
                "CPAttention received a batch without a packed-TND plan; set "
                "data.sfa_layout='TND' and keep batch_size=1."
            )
        return self.forward_tnd(x)

    # --- Pipeline ---

    def forward_tnd(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 0:
            return x
        plan = self.cp_runtime.plan
        projections = self._project_documents(x, plan)
        raw_by_doc, raw_stream, row_ranges = self._exchange_raw(projections, plan)
        latent_by_doc, latent_stream = self._exchange_latent(projections, plan)
        operands = self._assemble_documents(
            projections, plan, raw_by_doc, raw_stream, latent_by_doc, latent_stream, row_ranges
        )
        return self._apply_attention(projections, *operands)

    def _project_documents(self, x: torch.Tensor, plan) -> list[_DocumentProjection]:
        rope_dim = self.rope_head_dim
        projections = []
        for segment in plan.docs:
            doc_start = plan.cu_ends[segment.doc_id]
            local_lo = doc_start + segment.q_lo - plan.local_start
            length = segment.q_hi - segment.q_lo
            xs = x[:, local_lo : local_lo + length]
            qr = self.q_norm(self.wq_a(xs))
            query = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
            freqs_q = self.freqs_cis[segment.q_lo : segment.q_hi]
            apply_rotary_emb(query[..., -rope_dim:], freqs_q)
            projections.append(
                _DocumentProjection(
                    segment=segment,
                    xs=xs,
                    qr=qr,
                    query=query,
                    freqs_q=freqs_q,
                    kv_pre=self.kv_norm(self.wkv(xs)),
                )
            )
        return projections

    def _exchange_raw(self, projections, plan):
        """Assemble the pre-RoPE raw KV rows every local document needs."""
        runtime = self.cp_runtime
        if plan.exchange_style != "halo_prefix":
            return None, runtime.gather_stream(torch.cat([p.kv_pre for p in projections], dim=1)), None
        row_ranges = plan.row_ranges(1)
        raw_by_doc = runtime.gather_document_rows({p.segment.doc_id: p.kv_pre for p in projections}, 1, "raw")
        return raw_by_doc, None, row_ranges

    def _exchange_latent(self, projections, plan):
        """Assemble the compressed rows this rank reads, pre-RoPE as well."""
        ratio = self.compress_ratio
        if not ratio or not self.is_kv_source:
            return None, None
        runtime = self.cp_runtime
        if plan.exchange_style != "halo_prefix":
            local_latent = [self.compressor(p.xs, 0) for p in projections]
            return None, runtime.gather_stream(torch.cat(local_latent, dim=1), ratio=ratio)
        latent_by_doc = runtime.gather_document_rows(
            {p.segment.doc_id: self.compressor(p.xs, 0) for p in projections}, ratio, "cmp"
        )
        return latent_by_doc, None

    def _assemble_documents(self, projections, plan, raw_by_doc, raw_stream, latent_by_doc, latent_stream, row_ranges):
        """Per-document attention operands plus the packed operator metadata."""
        rope_dim = self.rope_head_dim
        window = self.window_size
        ratio = self.compress_ratio
        runtime = self.cp_runtime
        device = projections[0].xs.device

        query_parts, raw_parts, comp_parts, cmp_parts = [], [], [], []
        for projection in projections:
            segment = projection.segment
            buffer_start = 0 if plan.exchange_style == "full_prefix" else max(0, segment.q_lo - (window - 1))
            if raw_by_doc is not None:
                row_lo = row_ranges[segment.doc_id][0]
                raw = raw_by_doc[segment.doc_id][:, buffer_start - row_lo : segment.q_hi - row_lo]
            else:
                doc_start = plan.cu_ends[segment.doc_id]
                raw = raw_stream[:, doc_start + buffer_start : doc_start + segment.q_hi]
            # apply_rotary_emb rotates in place, so the gathered stream must be
            # copied first: slicing it would rotate every receiver's copy.
            raw = raw.clone()
            apply_rotary_emb(raw[..., -rope_dim:], self.freqs_cis[buffer_start : segment.q_hi])

            query_parts.append(projection.query)
            raw_parts.append(raw)
            if not ratio:
                continue
            compressed, indices = self._compressed_rows(projection, plan, latent_by_doc, latent_stream)
            comp_parts.append(compressed)
            cmp_parts.append(indices)

        cu_q, cu_kv, cu_cmp = runtime.cu_seqlens(ratio, device)
        positions = runtime.window_positions(plan, device).reshape(-1, window)
        return (
            torch.cat(query_parts, dim=1).reshape(-1, self.n_local_heads, self.head_dim),
            torch.cat(raw_parts, dim=1).reshape(-1, self.head_dim),
            torch.cat(comp_parts, dim=1).reshape(-1, self.head_dim) if ratio else None,
            torch.cat(cmp_parts, dim=1).reshape(-1, cmp_parts[0].shape[-1]) if ratio else None,
            positions,
            (cu_q, cu_kv, cu_cmp),
        )

    def _compressed_rows(self, projection, plan, latent_by_doc, latent_stream):
        """Compressed KV and index top-k for one document."""
        runtime = self.cp_runtime
        segment = projection.segment
        ratio = self.compress_ratio
        rope_dim = self.rope_head_dim
        device = projection.xs.device
        rows = segment.q_hi // ratio
        compress_state = runtime.state_for("compress", ratio)
        index_key_state = runtime.state_for("index_k", ratio)
        topk_state = runtime.state_for("topk", ratio)
        candidate_state = runtime.state_for("candidates")

        if self.is_kv_source:
            if latent_by_doc is not None:
                latent = latent_by_doc[segment.doc_id][:, :rows]
            else:
                base = plan.prefix_start(segment.doc_id, ratio)
                latent = latent_stream[:, base : base + rows]
            compressed = latent.clone()
            apply_rotary_emb(compressed[..., -rope_dim:], self.freqs_cis[: rows * ratio : ratio])
            compress_state[segment.doc_id] = compressed
            if self.is_index_source:
                index_key = self.indexer.k_norm(self.indexer.wk(latent))
                apply_rotary_emb(index_key[..., -rope_dim:], self.freqs_cis[: rows * ratio : ratio])
                index_key_state[segment.doc_id] = index_key
        else:
            compressed = compress_state[segment.doc_id]

        if not self.is_index_source:
            return compressed, topk_state[segment.doc_id]

        indexer = self.indexer
        query_index = indexer.wq_b(projection.qr).unflatten(-1, (indexer.n_local_heads, indexer.index_head_dim))
        apply_rotary_emb(query_index[..., -rope_dim:], projection.freqs_q)
        weights = indexer.weights_proj(projection.xs) * (indexer.softmax_scale * indexer.n_heads**-0.5)
        if indexer.use_fused_lightning_indexer:
            # The kernel derives the causal mask from the row index; keep document rows.
            query_padded = query_index.new_zeros(1, segment.q_hi, *query_index.shape[2:])
            query_padded[:, segment.q_lo : segment.q_hi] = query_index
            weights_padded = weights.new_zeros(1, segment.q_hi, weights.shape[-1])
            weights_padded[:, segment.q_lo : segment.q_hi] = weights
            candidate_kwargs = {}
            if indexer.uses_candidates:
                # The source published padded rows in exactly this layout.
                candidate_blocks, candidate_lengths = candidate_state[segment.doc_id]
                candidate_kwargs = dict(
                    candidate_block_size=indexer.candidate_block_size,
                    candidate_block_indices=candidate_blocks,
                    candidate_block_length=candidate_lengths,
                )
            elif indexer.is_candidate_source:
                candidate_kwargs = dict(
                    candidate_topk_blocks=indexer.candidate_topk_blocks,
                    candidate_block_size=indexer.candidate_block_size,
                )
            indices, candidates, candidate_lengths = fused_lightning_indexer(
                query_padded,
                index_key_state[segment.doc_id][:, :rows],
                weights_padded,
                topk=indexer.index_topk,
                compress_ratio=ratio,
                index_offset=0,
                attention_mask=None,
                page_size=indexer.lightning_indexer_page_size,
                **candidate_kwargs,
            )
            if indexer.is_candidate_source:
                candidate_state[segment.doc_id] = (candidates, candidate_lengths)
            indices = canonical_index_order(indices[:, segment.q_lo : segment.q_hi])
            topk_state[segment.doc_id] = indices
            return compressed, indices
        score = _compute_index_score(query_index, index_key_state[segment.doc_id], weights)
        visible = ((segment.q_lo + torch.arange(1, segment.q_hi - segment.q_lo + 1, device=device)) // ratio).unsqueeze(
            -1
        )
        score = score.masked_fill(torch.arange(rows, device=device) >= visible, -torch.inf)
        if indexer.is_candidate_source:
            candidate_state[segment.doc_id] = select_candidate_blocks(
                score,
                visible,
                indexer.candidate_topk_blocks,
                indexer.candidate_block_size,
            )
        elif indexer.uses_candidates:
            score = score.masked_fill(~candidate_state[segment.doc_id], -torch.inf)
        selected = score.topk(min(indexer.index_topk, rows), dim=-1, sorted=False).indices
        selected = selected.sort(dim=-1).values
        indices = torch.where(selected < visible, selected, -1).int()
        # Short documents yield narrower rows; pad to one width before concatenation.
        width = indices.shape[-1]
        if width < indexer.index_topk:
            indices = torch.nn.functional.pad(indices, (0, indexer.index_topk - width), value=-1)
        topk_state[segment.doc_id] = indices
        return compressed, indices

    def _apply_attention(self, projections, query, raw, compressed, indices, positions, cu_seqlens) -> torch.Tensor:
        rope_dim = self.rope_head_dim
        ratio = self.compress_ratio
        model_dtype = projections[0].xs.dtype
        cu_q, cu_kv, cu_cmp = cu_seqlens

        output = indexed_sparse_attention(
            query,
            raw,
            self.attn_sink,
            positions,
            self.softmax_scale,
            use_sparse_flash_attn=True,
            compressed_key_value=compressed,
            compressed_topk_indices=indices,
            compress_ratio=ratio or 1,
            window_size=self.window_size,
            layout="TND",
            cu_seqlens=cu_q,
            cu_seqlens_ori_kv=cu_kv,
            cu_seqlens_cmp_kv=cu_cmp if ratio else None,
        )

        outputs = []
        offset = 0
        for projection in projections:
            length = projection.segment.q_hi - projection.segment.q_lo
            attention_output = output[offset : offset + length]
            offset += length
            attention_output = attention_output.clone().unsqueeze(0)
            apply_rotary_emb(attention_output[..., -rope_dim:], projection.freqs_q, True)
            attention_output = attention_output.view(1, length, self.n_local_groups, -1)
            grouped = torch.einsum(
                "bsgd,grd->bsgr",
                attention_output,
                self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1),
            )
            outputs.append(self.wo_b(grouped.flatten(2)))
        return torch.cat(outputs, dim=1).to(model_dtype)

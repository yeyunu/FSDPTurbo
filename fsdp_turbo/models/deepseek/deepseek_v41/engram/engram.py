import hashlib
from dataclasses import dataclass
from itertools import accumulate
from math import isqrt

import numpy as np
import torch
from torch import nn

from .host_offload_engram import HostOffloadEngram, TorchHostOffloadEngram
from ..model_args import ModelArgs


def _is_prime(candidate: int) -> bool:
    if candidate < 2:
        return False
    if candidate % 2 == 0:
        return candidate == 2
    return all(candidate % divisor for divisor in range(3, isqrt(candidate) + 1, 2))


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash the same way.
    Returns the lookup plus the size of the compressed vocab -- and that size matters beyond bounds
    checking, because every hash multiplier is derived from it.
    """
    # Keep this byte-for-byte compatible with the released reference.  In
    # particular special tokens participate in the map and a token containing
    # exactly one space must not collapse into the empty-string bucket.
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers hash differently.

    Kept odd, and bounded so that `token_id * multiplier` cannot overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        # NumPy PCG64 is part of the published Engram hash ABI.  Replacing it
        # with torch.Generator changes every row ID for the same checkpoint.
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.from_numpy(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max_ngram_size-gram), each split
    over `n_heads` heads. Every (n-gram size, head) pair owns its own prime-sized bucket range in the
    layer's table; the primes are drawn in order and never reused, which keeps the ranges disjoint.
    """

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]  # table rows, per engram layer
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head] bucket modulus
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        if len(set(layer_ids)) != len(layer_ids) or any(i < 0 or i >= args.n_layers for i in layer_ids):
            raise ValueError("Engram layer IDs must be unique backbone layer indices.")
        if max_ngram_size < 2 or n_heads < 1 or args.engram_head_dim < 1 or args.engram_vocab_size < 2:
            raise ValueError("Engram requires n-grams >= 2, positive heads/dim, and vocab size >= 2.")
        if len(args.engram_num_embeddings) != len(layer_ids):
            raise ValueError("engram_num_embeddings must contain one table size per Engram layer.")
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        for rows, layer_primes in zip(args.engram_num_embeddings, primes):
            if rows < sum(sum(heads) for heads in layer_primes):
                raise ValueError("Engram table rows do not cover the generated hash bucket ranges.")
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )


class NgramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there.

    Ids go through the compressed table, then each position is hashed with the `max_ngram_size - 1`
    tokens before it. Look-back stops at the start of the sequence and at any dead token (an image
    span, cached as DEAD), so an n-gram never spans one. The cache carries all of this across the
    prefill/decode split.
    """

    DEAD = -1

    def __init__(self, args, layout: EngramLayout, tokenizer):
        super().__init__()
        if tokenizer is None:
            raise ValueError("Engram requires the model tokenizer to construct its compressed token map.")
        self.layout = layout
        # every hash multiplier derives from the compressed vocab size, so a mismatch there would
        # silently rehash the whole table
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != args.engram_compressed_vocab_size:
            raise ValueError(
                "Engram compressed tokenizer vocabulary is not checkpoint-compatible: "
                f"computed {vocab_size}, configured {args.engram_compressed_vocab_size}. "
                "Use the tokenizer revision matching the Engram checkpoint."
            )
        multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)
        self.hash_fingerprint = hashlib.sha256(
            np.asarray(token_map, dtype=np.int64).tobytes()
            + np.asarray(layout.primes, dtype=np.int64).tobytes()
            + multipliers.numpy().tobytes()
        ).hexdigest()
        self.pad_id = token_map[args.engram_pad_id]
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = [[0, *accumulate(sizes[:-1])] for sizes in flat]
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer(
            "cache",
            torch.empty(args.max_batch_size, args.max_seq_len, dtype=torch.int64),
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """token_mask: [B, L], False for tokens that take no part in an n-gram (image spans).
        Returns the hash ids, shaped [B, L, n_engram_layers, n_hash_cols].
        """
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)
        self.cache[:batch, start_pos : start_pos + seqlen] = compressed

        positions = torch.arange(start_pos, start_pos + seqlen, device=input_ids.device).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = self.cache[:batch].gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        return self._hash_tokens(torch.stack(tokens, dim=-1))

    def _hash_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Hash ``[B, L, max_ngram_size]`` compressed ids into per-layer, per-head buckets."""
        # XOR the multiplied ids together one lookback at a time, so the running value after step i
        # is the hash of the (i+1)-gram; each lands in its own prime-sized bucket range
        products = tokens.unsqueeze(2) * self.multipliers  # [B, L, n_engram_layers, max_ngram_size]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class CPNgramHashState(NgramHashState):
    """N-gram hashing over a per-document halo instead of the complete stream.

    The caller passes the tokens before ``q_lo`` (truncated at the document head)
    and the document-local start, so the halo never leaks a neighbouring
    document. Without a halo the vendored whole-stream path runs.
    """

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int | None = None,
        token_mask: torch.Tensor | None = None,
        *,
        halo_ids: torch.Tensor | None = None,
        halo_mask: torch.Tensor | None = None,
        q_lo: int | None = None,
    ) -> torch.Tensor:
        """Hash the local rows from the document halo, or the whole stream without one."""
        if halo_ids is None:
            return super().forward(input_ids, 0 if start_pos is None else start_pos, token_mask)

        local = self.token_map[input_ids]
        if token_mask is not None:
            local = torch.where(token_mask, local, self.DEAD)
        halo = self.token_map[halo_ids]
        if halo_mask is not None:
            halo = torch.where(halo_mask, halo, self.DEAD)
        cache = torch.cat([halo, local], dim=1)
        halo_len = halo.shape[1]
        length = local.shape[1]
        positions = (q_lo + torch.arange(length, device=input_ids.device)).expand_as(local)
        offsets = halo_len + torch.arange(length, device=input_ids.device)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            index = (offsets - shift).clamp_min(0).expand_as(local)
            source = cache.gather(1, index)
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        return self._hash_tokens(torch.stack(tokens, dim=-1))


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches that stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per hc copy plus a shared
    value. The gate is a normalized dot product of stream against key.
    """

    def __init__(self, args: ModelArgs, layer_id: int, layout: EngramLayout):
        # Resolve the base layer after module initialization to avoid a circular import.
        from ..layers import Linear  # pylint: disable=cyclic-import

        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.clamp_value = 1e-6

        table_rows = layout.num_embeddings[self.layer_hash_index]
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        if args.engram_storage_backend in ("host_offload", "elastic_buffer"):
            embedding_type = (
                HostOffloadEngram if args.engram_storage_backend == "elastic_buffer" else TorchHostOffloadEngram
            )
            self.embed = embedding_type(
                table_rows,
                layout.head_dim,
                layer_id=layer_id,
                num_max_tokens_per_rank=(args.max_batch_size * args.max_seq_len * n_hash_cols),
                pin_memory=True,
            )
        else:
            raise ValueError(
                "engram_storage_backend must be one of "
                "('host_offload', 'elastic_buffer'), got "
                f"{args.engram_storage_backend!r}."
            )
        self.wkv = Linear(n_hash_cols * layout.head_dim, args.dim * (args.hc_mult + 1))
        self.eps = args.norm_eps
        self.q_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))
        self.k_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, L, hc_mult, dim]; hash_ids: [B, L, n_hash_cols]; token_mask: [B, L], False shuts
        the gate so those positions pass through untouched.
        """
        memory = self.embed(hash_ids).flatten(-2).to(dtype=x.dtype)
        kv = self.wkv(memory)
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product
        h, eps = x.float(), self.eps
        # normalized per (token, hc copy) over `dim`, NOT jointly over the copies
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        # signed sqrt before the sigmoid, matching the training kernel
        magnitude = dot.abs().clamp_min(self.clamp_value).sqrt()
        gate = torch.sigmoid(torch.where(dot < 0, -magnitude, magnitude))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)

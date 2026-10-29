"""DeepSeek-V4.1 model assembly and compatibility exports for its components."""

from __future__ import annotations

# Explicit aliases preserve legacy exports while keeping Ruff aware of their public use.
# pylint: disable=useless-import-alias

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: F401 - retained for legacy imports
from torch import nn

from fsdp_turbo.ops.compressor import fused_compressor as fused_compressor
from fsdp_turbo.ops.hf32_linear import hf32_linear as hf32_linear
from fsdp_turbo.ops.lightning_indexer import fused_lightning_indexer as fused_lightning_indexer
from fsdp_turbo.ops.mhc import (
    fused_mhc_post as fused_mhc_post,
    fused_mhc_pre_sinkhorn_premix as fused_mhc_pre_sinkhorn_premix,
)
from fsdp_turbo.ops.rms_norm import rms_norm as rms_norm
from fsdp_turbo.ops.rope import rope as rope
from fsdp_turbo.ops.sinkhorn import hc_split_sinkhorn as hc_split_sinkhorn
from fsdp_turbo.ops.sparse_attention import indexed_sparse_attention as indexed_sparse_attention

from .activation import limited_swiglu as limited_swiglu, silu_mul as silu_mul
from .attention import Attention as Attention, get_window_topk_idxs as get_window_topk_idxs
from .compressor import Compressor as Compressor
from .dspark import (
    DSparkAttention as DSparkAttention,
    DSparkBlock as DSparkBlock,
    DSparkConfidenceHead as DSparkConfidenceHead,
    DSparkMarkovHead as DSparkMarkovHead,
    get_dspark_topk_idxs as get_dspark_topk_idxs,
    sample as sample,
)
from .engram import Engram as Engram, EngramLayout, NgramHashState
from .experts import DeepSeekV41Experts as DeepSeekV41Experts
from .engram.host_offload_engram import (
    HostOffloadEngram as HostOffloadEngram,
    TorchHostOffloadEngram as TorchHostOffloadEngram,
)
from .image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START
from .indexer import (
    Indexer as Indexer,
    _compute_index_score as _compute_index_score,
    select_candidate_blocks as select_candidate_blocks,
)
from .layers import (
    Block as Block,
    ColumnParallelLinear as ColumnParallelLinear,
    Expert as Expert,
    Gate as Gate,
    Linear as Linear,
    MoE as MoE,
    ParallelEmbedding as ParallelEmbedding,
    ParallelHead as ParallelHead,
    RMSNorm as RMSNorm,
    RowParallelLinear as RowParallelLinear,
    apply_rotary_emb as apply_rotary_emb,
    linear as linear,
    make_identity_pre_mix as make_identity_pre_mix,
    precompute_freqs_cis as precompute_freqs_cis,
)
from .model_args import ModelArgs as ModelArgs
from .runtime import (
    ContextParallelAttentionState,
    SharedAttentionRuntime as SharedAttentionRuntime,
    shared_attn as shared_attn,
)
from .utils import (
    _packed_freqs as _packed_freqs,
    _packed_sequence_lengths as _packed_sequence_lengths,
    _sequence_slices as _sequence_slices,
    set_dtype as set_dtype,
)
from .vision import Aligner, ViT

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from .image_processor import ImageInput


class Transformer(nn.Module):
    """DeepSeek-V4.1: embed -> expand to hc_mult copies -> blocks -> collapse -> logits.

    FSDPTurbo owns distributed sharding. The tokenizer only feeds the Engram token map.
    """

    def __init__(self, args: ModelArgs, tokenizer=None):
        if args.dtype != "bf16" or args.expert_dtype is not None:
            raise ValueError("DeepSeek-V4.1 training requires dtype='bf16' and expert_dtype=None.")
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.temperature = args.temperature
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.engram_layout = EngramLayout.from_args(args)
        self.engram_hash = (
            NgramHashState(args, self.engram_layout, tokenizer) if self.engram_layout is not None else None
        )
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(layer_id, args, self.engram_layout) for layer_id in range(args.n_layers)])
        if self.engram_hash is not None:
            for layer in self.layers:
                if layer.engram is not None:
                    layer.engram.embed.hash_fingerprint = self.engram_hash.hash_fingerprint
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        self.mtp = torch.nn.ModuleList()
        self.target_layer_ids = args.dspark_target_layer_ids
        if args.dspark_block_size:
            for layer_id in range(args.n_mtp_layers):
                self.mtp.append(DSparkBlock(args.n_layers + layer_id, args))
                self.mtp[-1].embed = self.embed
                self.mtp[-1].head = self.head
        self.hc_mult = args.hc_mult
        self.vision = None
        if args.vision_enabled:
            self.vision = ViT(args)
            self.aligner = Aligner(args)
            # learned embeddings for the image span delimiters
            self.image_start = nn.Parameter(torch.empty(args.dim))
            self.image_end = nn.Parameter(torch.empty(args.dim))
            self.image_newline = nn.Parameter(torch.empty(args.dim))

    def encode_image(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images: list[list[ImageInput] | None], h: torch.Tensor) -> None:
        """Overwrite each image's token span in h with its ViT/aligner features. The IMAGE slots take
        the aligner rows in row-major order; the span delimiters take learned embeddings.
        """
        image_records = [img for sample in images for img in (sample or ())]
        if not image_records:
            # Vision modules are FSDP-wrapped. Every model-parallel rank must
            # invoke them exactly once even when its local pack has no image.
            grid = self.aligner.downsample_ratio
            patch = self.vision.patch_size
            dummy = h.new_zeros((grid * grid, 3, patch, patch))
            patches = [dummy]
            grid_sizes = [(grid, grid)]
        else:
            patches = [img.patches.to(h.device) for img in image_records]
            grid_sizes = [(img.n_vit_h, img.n_vit_w) for img in image_records]

        vision_features = self.vision(patches, grid_sizes)
        aligned_features = self.aligner(vision_features, grid_sizes)
        if not image_records:
            # FSDP must reduce the same parameters on image and text-only ranks.
            dummy_sum = sum(features.sum() for features in aligned_features)
            dummy_sum = dummy_sum + self.image_start.sum() + self.image_end.sum() + self.image_newline.sum()
            h.add_(dummy_sum.to(h.dtype) * 0)
            return

        feature_iterator = iter(aligned_features)
        for i, sample_images in enumerate(images):
            for img in sample_images or ():
                types = img.types.to(h.device)
                span = h[i, img.start : img.start + types.numel()]
                span[types == IMAGE_START] = self.image_start.to(h.dtype)
                span[types == IMAGE_END] = self.image_end.to(h.dtype)
                span[types == IMAGE_NEW_LINE] = self.image_newline.to(h.dtype)
                embeds = next(feature_iterator)
                span[types == IMAGE] = embeds.to(h.dtype)

    def forward_backbone(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        images: list[list[ImageInput] | None] | None = None,
        token_types: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        sfa_layout: str = "BSND",
        cu_seqlens: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        cp_group: ProcessGroup | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...] | None]:
        """Run the shared backbone path used by training and generation.

        CP supplies local rows of globally computed ``engram_hashes`` to retain
        n-gram history across shard boundaries; other callers leave it unset.
        ``cp_group`` is passed to every block explicitly. Image CP batches stay
        complete until fusion, then select local embedding and mask rows.
        """
        if start_pos == 0:
            shared_attn.reset()

        image_mask = None if token_types is None else token_types >= 0  # TEXT is -1
        engram_mask = None if image_mask is None else ~image_mask
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            engram_mask = attention_mask if engram_mask is None else engram_mask & attention_mask
        # CP supplies hashes from the complete sequence before selecting local rows.
        if engram_hashes is None:
            if self.engram_hash is None:
                engram_hashes = None
            elif cu_seqlens is not None and cu_seqlens.numel() > 2:
                boundaries = cu_seqlens.tolist()
                engram_hashes = torch.cat(
                    [
                        self.engram_hash(
                            input_ids[:, begin:end], 0, None if engram_mask is None else engram_mask[:, begin:end]
                        )
                        for begin, end in zip(boundaries[:-1], boundaries[1:])
                    ],
                    dim=1,
                )
            else:
                engram_hashes = self.engram_hash(input_ids, start_pos, engram_mask)

        h = self.embed(input_ids)
        if images is not None:
            assert start_pos == 0, "image spans must be prefilled in a single chunk"
            self.merge_image_embeddings(images, h)

        if cp_group is not None and images is not None:
            from fsdp_turbo.distributed.context_parallel.batch import (
                split_tensor_to_group,
            )

            h = split_tensor_to_group(h, cp_group)
            image_mask = split_tensor_to_group(image_mask, cp_group)
            engram_mask = split_tensor_to_group(engram_mask, cp_group)
            if attention_mask is not None:
                attention_mask = split_tensor_to_group(attention_mask, cp_group)

        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        main_hiddens = []
        collected_hidden_states = [h] if output_hidden_states else None
        cp_state = ContextParallelAttentionState() if cp_group is not None else None

        for layer_id, layer in enumerate(self.layers):
            engram_hash_ids = (
                engram_hashes[:, :, layer.engram.layer_hash_index, :] if layer.engram is not None else None
            )
            if layer_id in self.target_layer_ids and layer.engram is not None:
                h = layer.engram(h, engram_hash_ids, engram_mask)
                engram_hash_ids = None
            # The MTP head reads the attention input of its target layers,
            # not their output.
            if layer_id in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
            layer_output = layer(
                h,
                start_pos,
                pre_mix,
                image_mask,
                attention_mask=attention_mask,
                sfa_layout=sfa_layout,
                cu_seqlens=cu_seqlens,
                engram_hash_ids=engram_hash_ids,
                engram_mask=engram_mask,
                cp_group=cp_group,
                **({"cp_state": cp_state} if cp_group is not None else {}),
            )
            if cp_group is not None:
                h, pre_mix, cp_state = layer_output
            else:
                h, pre_mix = layer_output
            if collected_hidden_states is not None:
                collected_hidden_states.append(h)

        h = self.layers[-1].hc_pre(h, pre_mix)
        h = self.norm(h)
        hidden_states = tuple(collected_hidden_states) + (h,) if collected_hidden_states is not None else None
        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        return h, main_hidden, hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        images=None,
        token_types: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        sfa_layout: str = "BSND",
        cu_seqlens: torch.Tensor | None = None,
    ):
        """input_ids: [b, s], every entry a real token id -- generate.py only ever passes positions it
        has already filled, so the padding it uses internally never reaches here. token_types /
        images carry the VL inputs built by image_processor.prepare_vl_inputs; image spans must lie
        inside the first (start_pos 0) chunk.
        """
        h, main_hidden, _ = self.forward_backbone(
            input_ids,
            start_pos=start_pos,
            images=images,
            token_types=token_types,
            attention_mask=attention_mask,
            sfa_layout=sfa_layout,
            cu_seqlens=cu_seqlens,
        )
        logits = self.head(h)
        output_ids = sample(logits, self.temperature)
        return output_ids, logits, main_hidden

    def forward_spec(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        start_pos: int = 0,
        *,
        attention_mask: torch.Tensor | None = None,
        sfa_layout: str = "BSND",
        cu_seqlens: torch.Tensor | None = None,
    ):
        h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        for layer in self.mtp:
            h, pre_mix = layer(
                h,
                start_pos,
                pre_mix,
                main_x,
                attention_mask=attention_mask,
                sfa_layout=sfa_layout,
                cu_seqlens=cu_seqlens,
            )
        if start_pos == 0:
            return None
        return self.mtp[-1].forward_head(h, pre_mix, input_ids)

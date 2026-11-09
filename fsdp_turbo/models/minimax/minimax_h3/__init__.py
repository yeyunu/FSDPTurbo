"""Portable MiniMax-H3 DiT structure and latent packing helpers."""

from .modeling_minimax_h3 import (
    MiniMaxH3DiT,
    MiniMaxH3ForTraining,
    MiniMaxH3TrainingOutput,
    MiniMaxH3DiTBlock,
    pack_audio,
    unpack_audio,
    patchify_video,
    unpatchify_video,
)

__all__ = [
    "MiniMaxH3ForTraining",
    "MiniMaxH3TrainingOutput",
    "MiniMaxH3DiT",
    "MiniMaxH3DiTBlock",
    "pack_audio",
    "unpack_audio",
    "patchify_video",
    "unpatchify_video",
]

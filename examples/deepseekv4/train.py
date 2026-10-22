# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# This file is adapted from the DeepSeek-V4 reference model implementation
# (DeepSeek-AI official release) for FSDPTurbo training: dense-attention and
# FSDPTurbo adaptations (expert-parallel dispatch, reduced config, hash-MoE
# buffers).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at LICENSE (Apache-2.0) in the repo root.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeek-V4-Flash training on the stock transformers (>=5.8.1) deepseek_v4
model: random init by default, or load real released weights when
``--ckpt``/``--weights`` is provided.

A single entry point (no separate train/initialize scripts): the model is
randomly initialized unless a checkpoint (or raw safetensors directory) is
given on the command line. Designed for GPU/NPU precision comparison with the
same source tree on both platforms.

Usage:
  python train.py --config config_train.yaml --max-steps 5                 # random init
  python train.py --config config_train.yaml --max-steps 5 --ckpt ckpt.pt  # real weights
  python train.py --config config_train.yaml --max-steps 5 --weights /path/DeepSeek-V4-Flash-BF16
"""

import argparse
import logging
import os
from dataclasses import replace

import torch

from fsdp_turbo.fsdp_turbo_config import load_config_from_yaml
from fsdp_turbo.models.deepseek.deepseek_v4 import (
    build_deepseek_v4_model,
    build_deepseek_v4_model_args,
    initialize_deepseek_v4_model,
    prepare_deepseek_v4_model_for_fsdp,
)
from fsdp_turbo.training.trainer import BaseTrainer
from fsdp_turbo.utils.log import print_rank

from load_weights import load_weights_into

logger = logging.getLogger(__name__)


def _expand_config_env(config) -> "object":
    """Expand ``${VAR}`` placeholders in the path fields of a parsed
    ``FSDPTurboConfig`` (the yaml loader performs no interpolation).
    """

    def expand(value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value

    model = replace(
        config.model,
        model_name_or_path=expand(config.model.model_name_or_path),
        tokenizer_name_or_path=expand(config.model.tokenizer_name_or_path),
    )
    checkpoint = replace(config.checkpoint, output_dir=expand(config.checkpoint.output_dir))
    data = replace(config.data, dataset_path=expand(config.data.dataset_path))
    return replace(config, model=model, checkpoint=checkpoint, data=data)


class DeepseekV4Trainer(BaseTrainer):
    """BaseTrainer variant that loads real weights if a checkpoint is given."""

    def __init__(self, *args, weights_dir: str = "", ckpt_path: str = "", **kwargs):
        self._weights_dir = weights_dir
        self._ckpt_path = ckpt_path
        super().__init__(*args, **kwargs)

    def _model_args(self):
        if not hasattr(self, "_cached_model_args"):
            self._cached_model_args = build_deepseek_v4_model_args()
        return self._cached_model_args

    def build_processor(self):
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast.from_pretrained(
            self.config.model.tokenizer_name_or_path,
        )

    def build_model(self):
        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        print_rank(logger.info, "> Building DeepSeek-V4-Flash reference model (transformers >= 5.8.1)...")
        training_model = build_deepseek_v4_model(
            model_args=self._model_args(),
            tokenizer=self.tokenizer,
        )

        if self._ckpt_path and os.path.exists(self._ckpt_path):
            # Load a prepared .pt checkpoint (heavyweight layout mirroring the
            # transformers state dict). ``training_model.model`` is the HF
            # backbone whose state-dict keys are the plain "model.*" paths used
            # by loaders and the parallelism plans.
            print_rank(logger.info, f"> Loading checkpoint {self._ckpt_path} ...")
            ckpt = torch.load(self._ckpt_path, map_location="cpu")
            missing, unexpected = training_model.model.load_state_dict(ckpt, strict=False)
            print_rank(
                logger.info,
                f"> Checkpoint loaded: {len(ckpt)} keys, {len(missing)} missing, {len(unexpected)} unexpected",
            )
            self._clip_tid2eid(training_model.model)
        elif self._weights_dir and os.path.isdir(self._weights_dir):
            # Load raw released safetensors (subset mapped into the model).
            stats = load_weights_into(training_model.model, self._weights_dir, device="cpu")
            print_rank(
                logger.info,
                f"> Loaded {stats['loaded']} keys from {self._weights_dir}; "
                f"{len(stats['skipped_shape_mismatch'])} shape mismatches",
            )
            self._clip_tid2eid(training_model.model)
        else:
            # No weights given -> random initialization (official init rules via
            # the stock DeepseekV4PreTrainedModel._init_weights).
            print_rank(logger.info, "> No checkpoint/weights provided, using random initialization...")
            initialize_deepseek_v4_model(training_model)

        training_model = prepare_deepseek_v4_model_for_fsdp(
            training_model,
            device=torch.accelerator.current_accelerator(),
            parameter_dtype=self.config.model.torch_dtype,
        )

        # Parallelism plans are written against the transformers layout
        # ("model.layers.{*}.mlp.experts", ...), so FSDPTurbo wraps the HF
        # backbone (``training_model.model``) rather than the adapter facade:
        # the adapter's extra nesting level would shift every module path.
        # Note: SwapActivation is applied inside FSDPTurbo.__init__
        # (fsdp_turbo/fsdp_turbo.py:apply_swap_activation_modules) when
        # memory.swap_activation is enabled in the config, so no extra calls
        # are needed here.
        print_rank(logger.info, "> Applying FSDPTurbo parallelism...")
        model = FSDPTurbo(self.config, training_model.model)
        print_rank(logger.info, "> SwapActivation handled by FSDPTurbo (memory.swap_activation).")
        return model

    def _clip_tid2eid(self, training_model) -> None:
        """Render the hash-table routing safe for the reduced expert count: the
        released tid2eid maps tokens to experts 0..255, but the reduced model
        only has <n_routed_experts> experts. Upcast to int64 (required by
        gather) and clamp indices into [0, n_routed_experts) so the forward
        pass cannot index out of bounds. Identical logic in eval_weights.py.
        """
        n_experts = self._model_args().n_routed_experts
        clipped_total = 0
        tid_total = 0
        for buf_name, buf in training_model.named_buffers():
            if "tid2eid" not in buf_name:
                continue
            as_int64 = buf.data.to(torch.int64)
            clipped = int((as_int64 >= n_experts).sum())
            clipped_total += clipped
            tid_total += int(as_int64.numel())
            buf.data = as_int64.clamp(0, n_experts - 1)
        print_rank(
            logger.info,
            f"> tid2eid clip: {clipped_total}/{tid_total} indices clamped to {n_experts} experts",
        )

    def build_optimizer(self):
        """Standard AdamW via ``BaseTrainer.build_optimizer``.

        The base implementation (fsdp_turbo/training/trainer.py) already
        carries the full optimizer construction: ``OptimizerFactory``
        (optimizer_type-driven), AdamWSwap-specific swap arguments when
        ``optimizer_type == "AdamWSwap"``, the quantization
        ``hook_optimizer_step`` wiring and the "model is None" guard — so
        delegating here preserves those abilities instead of reimplementing
        them.
        """
        print_rank(logger.info, "> Building DeepSeek-V4 optimizer (via BaseTrainer)...")
        return super().build_optimizer()


def parse_arguments():
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash FSDP Training")
    parser.add_argument("--config", type=str, default="config_train.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--weights", type=str, default="", help="Path to raw safetensors dir")
    parser.add_argument("--ckpt", type=str, default="", help="Path to prepared .pt checkpoint")
    return parser.parse_args()


def main():
    parsed = parse_arguments()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parsed.config)
    fsdp_config = _expand_config_env(load_config_from_yaml(config_path))

    # Command-line overrides construct a new validated config instead of
    # mutating an already-created FSDPTurboConfig at runtime.
    if parsed.output_dir:
        checkpoint_config = replace(fsdp_config.checkpoint, output_dir=parsed.output_dir)
        fsdp_config = replace(fsdp_config, checkpoint=checkpoint_config)
    if parsed.max_steps is not None:
        run_config = replace(fsdp_config.run, max_steps=parsed.max_steps)
        fsdp_config = replace(fsdp_config, run=run_config)

    trainer = DeepseekV4Trainer(
        config=fsdp_config,
        weights_dir=parsed.weights,
        ckpt_path=parsed.ckpt,
    )
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()

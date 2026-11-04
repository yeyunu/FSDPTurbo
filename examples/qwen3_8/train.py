# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Qwen3.8-Flash-Next (qwen4_exp) SFT training example.

Qwen3.8-Flash-Next is a multimodal MoE model whose language backbone combines:
  - Full attention (QSA-indexed) + GatedDeltaNet (linear attention)
  - PLE n-gram embedding (huge vocab-parallel embedding table)
  - MoE experts with SiTU activation

This example supports FSDP2 + EP + optional Ulysses CP + optional VP/VFSDP.
"""

import argparse
import logging
import os

import torch
from transformers import AutoConfig

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import load_config_from_yaml
from fsdp_turbo.training.trainer import BaseTrainer
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.model import convert_model_dtype, set_num_hidden_layers

logger = logging.getLogger(__name__)


def _set_attn_implementation(model_config, impl: str):
    for cfg in (
        model_config,
        getattr(model_config, "text_config", None),
        getattr(model_config, "vision_config", None),
    ):
        if cfg is not None:
            cfg._attn_implementation = impl


def _overwrite_qwen4_exp_config(model_config, config):
    """Apply qwen4_exp-specific config overrides on the HF model_config.

    Model-specific knobs (GDN/QSA backends, expert matmul, PLE offload, etc.)
    are not part of the generic ``ModelConfig`` dataclass, so they are set
    directly on the HuggingFace config here — matching the convention used by
    the qwen3_5 / qwen3_5_moe examples for ``_attn_implementation``.
    """
    text_cfg = getattr(model_config, "text_config", model_config)
    vision_cfg = getattr(model_config, "vision_config", None)

    text_cfg.gdn_implementation = "triton"
    text_cfg.causal_conv1d_implementation = "triton"
    text_cfg.qsa_implementation = "triton"
    # Qwen-specific CP settings belong to the HF text config.
    # Preserve values loaded from the model config; adjust them here to tune CP.
    cp_plan = config.distributed.cp_plan
    qsa_chunk_size = getattr(cp_plan, "qsa_cp_query_chunk_size", None)
    text_cfg.qsa_cp_query_chunk_size = (
        qsa_chunk_size if qsa_chunk_size is not None else getattr(text_cfg, "qsa_cp_query_chunk_size", 128)
    )
    qsa_sparse_gather = getattr(cp_plan, "qsa_sparse_gather", None)
    text_cfg.qsa_sparse_gather = (
        qsa_sparse_gather if qsa_sparse_gather is not None else getattr(text_cfg, "qsa_sparse_gather", True)
    )
    text_cfg.skip_gdn_recompute = True
    text_cfg.skip_flash_attn_recompute = True
    if vision_cfg is not None:
        vision_cfg.skip_flash_attn_recompute = True

    text_cfg.use_grouped_expert_matmul = True
    text_cfg._experts_implementation = "eager"
    model_config._experts_implementation = "eager"

    # PLE n-gram embedding: the CPU-offload variant
    # __init__ always builds a zero-size table so from_pretrained stays cheap;
    # the full table is materialized on host afterwards.
    #   - VP off: materialized AFTER FSDPTurbo, frozen, stays on CPU.
    #   - VP on:  materialized BEFORE FSDPTurbo (trainable) so
    #     vocab_parallelize_modules can shard it onto the accelerator.
    text_cfg.ple_embedding_on_cpu = True
    text_cfg._ngram_hf_dir = config.model.model_name_or_path

    # Shrink the PLE n-gram table for functional tests. The checkpoint shards
    # are then truncated deterministically.
    ngram_vocab_size_base = os.environ.get("NGRAM_VOCAB_SIZE_BASE", "").strip()
    if ngram_vocab_size_base:
        text_cfg.ngram_vocab_size_base = int(ngram_vocab_size_base)
        if text_cfg.ngram_vocab_size_base <= 0:
            raise ValueError(f"Invalid NGRAM_VOCAB_SIZE_BASE='{ngram_vocab_size_base}'.")

    text_cfg.enable_chunk_loss = config.memory.chunk_loss
    text_cfg.chunk_loss_size = config.memory.chunk_loss_size
    text_cfg.enable_dynamic_chunk_loss = False

    text_cfg.enable_ep_balance = False
    text_cfg.max_dup_experts_num = 2

    # MTP (Multi-Token Prediction) — number of MTP layers to train.
    # 0 disables MTP.
    text_cfg.mtp_num_layers = getattr(config.model, "mtp_num_layers", 0)
    if text_cfg.mtp_num_layers and text_cfg.mtp_num_layers < 0:
        raise ValueError(f"Invalid mtp_num_layers='{text_cfg.mtp_num_layers}'. Must be a non-negative integer.")


class Qwen38FlashNextTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.processor = None

    def build_tokenizer(self):
        if self.config.data.is_multimodal:
            from transformers import AutoProcessor

            print_rank(logger.info, "> Building Processor (multimodal)...")
            processor = AutoProcessor.from_pretrained(self.config.model.tokenizer_name_or_path, trust_remote_code=True)
            processor.tokenizer.model_max_length = self.config.data.max_seq_length
            self.processor = processor
            return processor.tokenizer
        return super().build_tokenizer()

    def build_model(self):
        print_rank(logger.info, "> Building Qwen3.8-Flash-Next Model...")

        model_config = AutoConfig.from_pretrained(self.config.model.model_name_or_path, trust_remote_code=True)
        _set_attn_implementation(model_config, "sdpa")
        from transformers.integrations.sdpa_attention import (
            sdpa_attention_forward as _orig_sdpa_attention,
        )
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _ATTN_REGISTRY

        _vision_called = []

        def _vision_attention_local(*args, **kwargs):
            if not _vision_called:
                _vision_called.append(1)
            return _orig_sdpa_attention(*args, **kwargs)

        _ATTN_REGISTRY.register("sdpa_vision", _vision_attention_local)
        model_config.vision_config._attn_implementation = "sdpa_vision"

        set_num_hidden_layers(model_config, self.config.model.num_hidden_layers)

        _overwrite_qwen4_exp_config(model_config, self.config)

        # Build the model from the FSDPTurbo qwen4_exp modeling (subclass-based,
        # no monkey-patching): fused NPU ops, QSA indexer operator dispatch, GDN
        # dispatch, PLE CPU offload, post-fusion Ulysses forward and MTP all
        # live in fsdp_turbo/models/qwen/modeling_qwen4_exp.py and are steered
        # by the config knobs set in _overwrite_qwen4_exp_config above.
        from fsdp_turbo.models.qwen.modeling_qwen4_exp import (
            Qwen4ExpForConditionalGeneration,
        )

        model = Qwen4ExpForConditionalGeneration.from_pretrained(
            self.config.model.model_name_or_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
        )

        convert_model_dtype(model, self.config.model.torch_dtype)

        from fsdp_turbo.models.qwen.modeling_qwen4_exp import restore_ngram_integer_buffers

        restore_ngram_integer_buffers(model)

        # from_pretrained() runs Qwen4ExpTextNGramEmbeddingCpu.__init__ which
        # calls get_parallel_state(), creating a Singleton with default values
        # (FSDP=1, EP=1). Clear it so FSDPTurbo can re-init with the real config.
        from fsdp_turbo.distributed.parallel_state import ParallelState, Singleton

        Singleton._instances.pop(ParallelState, None)

        print_rank(logger.info, "> Applying FSDPTurbo parallelism...")

        # qwen3_8's fsdp_plan.apply_modules has parent-child nesting
        # (model.visual -> model.visual.blocks.{*}, model.language_model ->
        # layers.{*} -> linear_attn/mlp.experts). FSDP2 requires children
        # to be sharded before parents, so patch get_fsdp_modules to use
        # post-order traversal during FSDPTurbo init.
        from fsdp_turbo.distributed.fully_shard_parallel import (
            fully_shard_parallel as _fsp,
        )

        _orig_get_fsdp_modules = _fsp.get_fsdp_modules

        def _get_fsdp_modules_post_order(model, fsdp_plan, ignored_modules):
            fsdp_modules = {}
            for name, module in _fsp._post_order_traverse(model):
                for pattern, plan in fsdp_plan.apply_modules.items():
                    if _fsp.module_name_match(pattern, name) and name not in ignored_modules:
                        if module not in fsdp_modules:
                            fsdp_modules[module] = {}
                        fsdp_modules.get(module).update(plan)
            if len(fsdp_modules) == 0:
                raise RuntimeError(f"[FSDP2] No module named {fsdp_plan.apply_modules.keys()}.")
            return fsdp_modules

        _fsp.get_fsdp_modules = _get_fsdp_modules_post_order
        try:
            vp_enabled = self.config.distributed.vp_plan is not None
            if vp_enabled:
                # VP mode: materialize the trainable table on host BEFORE
                # FSDPTurbo — apply_vp_modules shards it onto the accelerator
                # (only this rank's slice is ever moved to the device).
                from fsdp_turbo.models.qwen.modeling_qwen4_exp import (
                    materialize_ngram_embedding_on_cpu,
                )

                materialize_ngram_embedding_on_cpu(
                    model,
                    dtype=torch.bfloat16,
                    hf_dir=self.config.model.model_name_or_path,
                    freeze=False,
                )
            model = FSDPTurbo(self.config, model)
        finally:
            _fsp.get_fsdp_modules = _orig_get_fsdp_modules

        # VP off (CPU offload): materialize the frozen PLE tables on CPU
        # *after* FSDPTurbo so model weights are already on device and CPU
        # RAM is freed for the huge n-gram table.
        if getattr(model_config.text_config, "ple_embedding_on_cpu", False) and not vp_enabled:
            from fsdp_turbo.models.qwen.modeling_qwen4_exp import (
                materialize_ngram_embedding_on_cpu,
            )

            materialize_ngram_embedding_on_cpu(
                model,
                dtype=torch.bfloat16,
                hf_dir=self.config.model.model_name_or_path,
                freeze=True,
            )

        print_rank(logger.info, "> Model built successfully")
        return model

    def build_dataloader(self):
        from fsdp_turbo.training.dataloader import build_dataloader

        return build_dataloader(self.config.data, tokenizer=self.tokenizer, processor=self.processor)

    def train_step(self, batch):
        """Execute a single training step with MTP loss accumulation.

        Overrides BaseTrainer.train_step to add mtp_loss (returned by the
        MTP-patched forward) to the main loss before backward. When MTP is
        disabled, this degrades to the base behavior exactly.
        """
        batch["use_cache"] = False
        batch["output_hidden_states"] = True
        outputs = self.model(**batch)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Model did not return a loss. Make sure the batch contains labels.")
        # MTP loss is stashed on the module instance (_last_mtp_loss) because
        # FSDP2's _post_forward hook rebuilds the output dataclass via
        # dataclasses.replace(), dropping the dynamically-set mtp_loss
        # attribute. The module survives the rebuild; the output does not.
        # FSDPTurbo wraps the model, so unwrap to reach the patched
        # Qwen4ExpForConditionalGeneration instance.
        inner_model = getattr(self.model, "model", self.model)
        mtp_loss = getattr(inner_model, "_last_mtp_loss", None)
        if mtp_loss is not None:
            inner_model._last_mtp_loss = None  # consume for next step
            if isinstance(mtp_loss, list):
                mtp_loss = [item for item in mtp_loss if item is not None]
                if mtp_loss:
                    mtp_loss = torch.mean(torch.stack(mtp_loss))
                else:
                    mtp_loss = None
            if mtp_loss is not None:
                mtp_scaling = self.config.model.mtp_loss_scaling_factor
                print_rank(logger.info, f"[MTP] mtp_loss={mtp_loss.item():.6f}, scaling_factor={mtp_scaling}")
                loss = loss + mtp_scaling * mtp_loss.to(loss.device)
            else:
                print_rank(logger.warning, "[MTP] mtp_loss list contained only None values")
        else:
            hs_status = (
                "None (capture_outputs failed)"
                if outputs.hidden_states is None
                else f"count={len(outputs.hidden_states)}"
            )
            print_rank(logger.warning, f"[MTP] mtp_loss is None — MTP did not run. outputs.hidden_states: {hs_status}")
        loss = loss / self.config.run.gradient_accumulation_steps
        loss.backward()
        aux_loss = outputs.aux_loss if hasattr(outputs, "aux_loss") else None
        return loss, aux_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.8-Flash-Next FSDP2 Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--model-name", type=str, default=None, help="Override model name or path")
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
        help="Override language model layer count",
    )
    parser.add_argument("--ulysses-size", type=int, default=None, help="Override Ulysses parallel size")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    config = load_config_from_yaml(config_path)

    if args.model_name:
        config.model.model_name_or_path = args.model_name
    if args.num_hidden_layers is not None:
        config.model.num_hidden_layers = args.num_hidden_layers
    if args.ulysses_size:
        config.distributed.ulysses_parallel_size = args.ulysses_size

    trainer = Qwen38FlashNextTrainer(config=config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()

import argparse
import logging
import os
from dataclasses import replace

import torch

from fsdp_turbo.fsdp_turbo_config import load_config_from_yaml
from fsdp_turbo.models.deepseek_v41 import (
    DeepSeekV41Processor,
    build_deepseek_v41_model,
    build_deepseek_v41_model_args,
    initialize_deepseek_v41_model,
    prepare_deepseek_v41_tokenizer,
    prepare_deepseek_v41_model_for_fsdp,
)
from fsdp_turbo.models.deepseek_v41.engram_optimizer import EngramSchedulerContainer
from fsdp_turbo.models.deepseek_v41.host_offload_embedding import host_engram_tables
from fsdp_turbo.quantization.cache import hook_optimizer_step
from fsdp_turbo.training.factories import OptimizerFactory, SchedulerFactory
from fsdp_turbo.training.trainer import BaseTrainer
from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


class DeepseekV41Trainer(BaseTrainer):
    def _model_args(self):
        if not hasattr(self, "_cached_model_args"):
            self._cached_model_args = build_deepseek_v41_model_args()
        return self._cached_model_args

    def build_processor(self):
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.config.model.tokenizer_name_or_path,
        )
        model_args = self._model_args()
        if prepare_deepseek_v41_tokenizer(tokenizer, model_args):
            self._using_validation_tokenizer = True
        return DeepSeekV41Processor(tokenizer, model_args)

    def build_model(self):
        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        print_rank(logger.info, "> Building DeepSeek-V4.1-Flash reference model...")
        model_args = self._model_args()
        if getattr(self, "_using_validation_tokenizer", False):
            model_args = replace(
                model_args,
                engram_layer_ids=(),
                engram_num_embeddings=(),
            )
        training_model = build_deepseek_v41_model(
            model_args=model_args,
            tokenizer=self.tokenizer,
            engram_storage_backend=self.config.model.engram_storage_backend,
            use_sparse_flash_attn=self.config.model.use_sparse_flash_attn,
            use_fused_lightning_indexer=(self.config.model.use_fused_lightning_indexer),
            use_fused_compressor=self.config.model.use_fused_compressor,
            use_fused_mhc=self.config.model.use_fused_mhc,
            use_fused_vision_flash_attn=(self.config.model.use_fused_vision_flash_attn),
        )
        training_model = prepare_deepseek_v41_model_for_fsdp(
            training_model,
            device=torch.accelerator.current_accelerator(),
            parameter_dtype=self.config.model.torch_dtype,
        )

        print_rank(logger.info, "> Applying FSDPTurbo parallelism...")
        parallel_model = FSDPTurbo(self.config, training_model)
        print_rank(logger.info, "> Initializing local DeepSeek-V4.1 parameter shards...")
        initialize_deepseek_v41_model(parallel_model.model)
        return parallel_model

    def build_optimizer(self):
        """Build the model-specific dense + Engram optimizer container."""
        print_rank(logger.info, "> Building DeepSeek-V4.1 optimizer...")
        kwargs = {
            "lr": self.config.optimizer.lr,
            "weight_decay": self.config.optimizer.weight_decay,
            "betas": (
                self.config.optimizer.adam_beta1,
                self.config.optimizer.adam_beta2,
            ),
            "eps": self.config.optimizer.adam_epsilon,
        }
        if self.config.optimizer.mem_fraction_static is not None:
            kwargs["mem_fraction_static"] = self.config.optimizer.mem_fraction_static
        if self.config.optimizer.state_keys is not None:
            kwargs["state_keys"] = self.config.optimizer.state_keys

        engram_tables = host_engram_tables(self.model)
        if not engram_tables:
            return super().build_optimizer()
        optimizer = OptimizerFactory.create(
            "EngramOptimizer",
            self.model,
            engram_tables,
            dense_optimizer_type=self.config.optimizer.optimizer_type,
            **kwargs,
        )
        hook_optimizer_step(self.model, optimizer.dense_optimizer)
        return optimizer

    def build_scheduler(self):
        """Build one scheduler for each optimizer managed by EngramOptimizer."""
        if self.optimizer is None:
            return None
        if not hasattr(self.optimizer, "checkpoint_optimizers"):
            return super().build_scheduler()
        max_steps = self._resolve_max_steps()
        schedulers = [
            SchedulerFactory.create(
                self.config.optimizer.lr_scheduler_type,
                optimizer,
                T_max=max_steps,
                eta_min=self.config.optimizer.min_lr,
            )
            for optimizer in self.optimizer.checkpoint_optimizers
        ]
        return EngramSchedulerContainer(schedulers)


def parse_arguments():
    parser = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash FSDP Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override checkpoint output directory",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override run.max_steps")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override data.dataset_path",
    )
    return parser.parse_args()


def main():
    parsed_arguments = parse_arguments()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), parsed_arguments.config)
    fsdp_config = load_config_from_yaml(config_path)

    # Command-line overrides construct a new validated config instead of
    # mutating an already-created FSDPTurboConfig at runtime.
    if parsed_arguments.output_dir:
        checkpoint_config = replace(fsdp_config.checkpoint, output_dir=parsed_arguments.output_dir)
        fsdp_config = replace(fsdp_config, checkpoint=checkpoint_config)
    if parsed_arguments.max_steps is not None:
        run_config = replace(fsdp_config.run, max_steps=parsed_arguments.max_steps)
        fsdp_config = replace(fsdp_config, run=run_config)
    if parsed_arguments.dataset_path:
        data_config = replace(fsdp_config.data, dataset_path=parsed_arguments.dataset_path)
        fsdp_config = replace(fsdp_config, data=data_config)

    trainer = DeepseekV41Trainer(config=fsdp_config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()

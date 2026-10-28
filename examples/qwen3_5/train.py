import argparse
import logging
import os
import torch

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import load_config_from_yaml
from fsdp_turbo.training.trainer import BaseTrainer
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.model import (
    convert_model_dtype,
    set_attn_implementation,
    set_num_hidden_layers,
)

logger = logging.getLogger(__name__)


class Qwen35Trainer(BaseTrainer):
    def build_model(self):
        from transformers import AutoConfig, AutoModelForImageTextToText

        print_rank(logger.info, "> Building Qwen3.5 Model...")

        model_config = AutoConfig.from_pretrained(self.config.model.model_name_or_path)
        set_attn_implementation(model_config, self.config.model.attn_implementation)
        set_num_hidden_layers(model_config, self.config.model.num_hidden_layers)

        model = AutoModelForImageTextToText.from_pretrained(
            self.config.model.model_name_or_path,
            config=model_config,
            torch_dtype=torch.bfloat16,
        )
        convert_model_dtype(model, self.config.model.torch_dtype)

        print_rank(logger.info, "> Applying FSDPTurbo parallelism...")
        model = FSDPTurbo(self.config, model)

        print_rank(logger.info, "> Model built successfully")
        return model

    def train_step(self, batch):
        batch['use_cache'] = False
        return super().train_step(batch)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5 Training with Ulysses Context Parallel")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--model-name", type=str, default=None, help="Override model name or path")
    parser.add_argument("--num-hidden-layers", type=int, default=None, help="Override language model layer count")
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

    trainer = Qwen35Trainer(config=config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()

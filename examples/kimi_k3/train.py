import argparse
import logging
import os

import torch
from transformers import AutoConfig

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import load_config_from_yaml
from fsdp_turbo.training.trainer import BaseTrainer
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.model import (
    convert_model_dtype,
    set_attn_implementation,
)

logger = logging.getLogger(__name__)


class KimiK3Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.processor = None

    def build_tokenizer(self):
        if self.config.data.is_multimodal:
            from fsdp_turbo.models.kimi.kimi_k3_processor import KimiK3Processor

            print_rank(logger.info, "> Building Processor (multimodal)...")
            processor = KimiK3Processor.from_pretrained(
                self.config.model.tokenizer_name_or_path, trust_remote_code=True
            )
            self.processor = processor
            return processor.tokenizer
        else:
            return super().build_tokenizer()

    def build_model(self):
        from fsdp_turbo.modules.kcp import _ensure_triton_driver

        _ensure_triton_driver()

        from transformers import AutoModel
        from fsdp_turbo.models.kimi import register_kimi_k3

        register_kimi_k3()

        print_rank(logger.info, "> Building KimiK3 Model...")
        model_config = AutoConfig.from_pretrained(self.config.model.model_name_or_path, trust_remote_code=False)
        set_attn_implementation(model_config, self.config.model.attn_implementation)
        model = AutoModel.from_pretrained(
            self.config.model.model_name_or_path,
            config=model_config,
            attn_implementation=self.config.model.attn_implementation,
            trust_remote_code=False,
        )
        convert_model_dtype(model, self.config.model.torch_dtype)

        print_rank(logger.info, "> Applying FSDPTurbo parallelism...")
        model = FSDPTurbo(self.config, model)

        print_rank(logger.info, "> Model built successfully")
        return model

    def build_dataloader(self):
        from fsdp_turbo.training.dataloader import build_dataloader

        return build_dataloader(self.config.data, tokenizer=self.tokenizer, processor=self.processor)

    def train_step(self, batch):
        batch['use_cache'] = False

        outputs = self.model(**batch)
        loss = outputs.loss
        aux_loss = outputs.aux_loss if hasattr(outputs, 'aux_loss') else None
        if loss is None:
            raise RuntimeError("Model did not return a loss. Make sure the batch contains labels.")

        loss = loss / self.config.run.gradient_accumulation_steps
        del outputs
        with torch.utils.checkpoint.set_checkpoint_debug_enabled(False):
            loss.backward()

        return loss, aux_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5 MoE Training with Ulysses Context Parallel")
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
    if args.ulysses_size:
        config.distributed.ulysses_parallel_size = args.ulysses_size

    trainer = KimiK3Trainer(config=config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()

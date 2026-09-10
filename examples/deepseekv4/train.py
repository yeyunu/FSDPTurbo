import argparse
import logging
import os

import torch
import yaml

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import load_config_from_yaml
from fsdp_turbo.training.trainer import BaseTrainer
from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


def convert_model_dtype(model, dtype):
    for param in model.parameters():
        param.data = param.data.to(dtype)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(dtype)


class DeepseekV4Trainer(BaseTrainer):
    def __init__(self, config, features=None):
        super().__init__(config)
        self.config = config
        self.features = features or {}

    def build_model(self):
        from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Config
        import modeling_deepseek_v4  # noqa: F401

        print_rank(logger.info, "> Building DeepseekV4 Model...")
        model_config = DeepseekV4Config()

        # reduce layer and expert for demo
        model_config.num_hidden_layers = 4
        model_config.n_routed_experts = 16
        model_config.index_topk = 16

        with torch.device('meta'):
            model = DeepseekV4ForCausalLM(model_config)
        convert_model_dtype(model, self.config.model.torch_dtype)

        init_weights = model._init_weights

        print_rank(logger.info, "> Applying FSDPTurbo parallelism...")
        model = FSDPTurbo(self.config, model)
        model.to_empty(device='npu')

        with torch.no_grad():
            model.apply(init_weights)

        # SwapActivation: must be applied after model has actual memory allocated
        if self.features.get("swap_activation"):
            from fsdp_turbo.memory.swap_activation.swap_activation import (
                swap_activation_modules,
            )

            plan = self.features.get("swap_activation_plan") or ["model.layers.{*}"]
            print_rank(logger.info, "> Applying SwapActivation...")
            model = swap_activation_modules(model, plan)

        # indexer grad bypass
        for name, module in model.named_modules():
            if "indexer" in name:
                for n, p in module.named_parameters():
                    p.grad = torch.zeros_like(p.data)
        return model

    def build_optimizer(self):
        if self.model is None:
            print_rank(logger.warning, "Model is None, skipping optimizer build.")
            return None

        from fsdp_turbo.quantization.cache import hook_optimizer_step

        print_rank(logger.info, "> Building Optimizer...")
        optim_cfg = self.config.optimizer

        if self.features.get("swap_optimizer"):
            from fsdp_turbo.memory.swap_optimizer.swap_optimizer import AdamWSwap

            optimizer = AdamWSwap(
                self.model.parameters(),
                lr=optim_cfg.lr,
                weight_decay=optim_cfg.weight_decay,
                betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
                eps=optim_cfg.adam_epsilon,
            )
        else:
            from fsdp_turbo.training.factories import OptimizerFactory

            optimizer = OptimizerFactory.create(
                optim_cfg.optimizer_type,
                self.model.parameters(),
                lr=optim_cfg.lr,
                weight_decay=optim_cfg.weight_decay,
                betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
                eps=optim_cfg.adam_epsilon,
            )

        hook_optimizer_step(self.model, optimizer)
        return optimizer

    def train_step(self, batch):
        batch['use_cache'] = False
        if self.features.get("chunk_loss"):
            chunk_size = self.features.get("chunk_loss_size", 1024)
            loss = self._compute_chunk_loss(batch, chunk_size)
        else:
            return super().train_step(batch)

        loss = loss / self.config.run.gradient_accumulation_steps
        loss.backward()
        return loss, None

    def _compute_chunk_loss(self, batch, chunk_size):
        from fsdp_turbo.memory.chunk_loss.chunk_loss import calculate_lm_loss, chunk_loss

        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask', None),
            use_cache=False,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        hidden = hidden_states[:, :-1, :].contiguous()

        head = self.model.model.lm_head
        shift_labels = batch['labels'][:, 1:].contiguous()

        num_items = (shift_labels != -100).sum().clamp(min=1)

        loss_kwargs_chunks = [
            {"shift_labels": chunk, "alpha": num_items, "ignore_index": -100, "reduction": "sum"}
            for chunk in torch.split(shift_labels, chunk_size, dim=1)
        ]

        return chunk_loss(
            hidden,
            head.weight,
            getattr(head, 'bias', None),
            calculate_lm_loss,
            loss_kwargs_chunks,
            chunk_size,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeekV4 Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--max-steps", type=int, default=None, help="Override run.max_steps")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = load_config_from_yaml(config_path)
    if args.output_dir:
        config.checkpoint.output_dir = args.output_dir
    if args.max_steps is not None:
        config.run.max_steps = args.max_steps

    features = raw.get("features") or {}
    trainer = DeepseekV4Trainer(config=config, features=features)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()

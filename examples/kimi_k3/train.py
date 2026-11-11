import argparse
import logging
import os
from dataclasses import replace

import torch
from transformers import AutoConfig

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

    def build_model(self) -> torch.nn.Module:
        """Load one repository-local K3 reference family, then attach parallelism."""
        from fsdp_turbo.models.kimi import register_kimi_k3
        from fsdp_turbo.models.kimi.modeling_kimi_k3 import (
            KimiK3ForConditionalGeneration,
            build_kimi_k3_model_class,
        )
        from fsdp_turbo.models.kimi.modeling_kimi_linear import KimiDeltaAttention, KimiMLAAttention

        register_kimi_k3()
        print_rank(logger.info, "> Building KimiK3 Model...")
        model_config = AutoConfig.from_pretrained(self.config.model.model_name_or_path, trust_remote_code=False)
        set_num_hidden_layers(model_config, self.config.model.num_hidden_layers)
        set_attn_implementation(model_config, self.config.model.attn_implementation)
        if self.config.memory.mm_merge_chunk_tokens > 0:
            model_config.mm_merge_chunk_tokens = self.config.memory.mm_merge_chunk_tokens
        cp_plan = self.config.distributed.cp_plan
        model_class = build_kimi_k3_model_class(
            KimiK3ForConditionalGeneration,
            KimiDeltaAttention,
            KimiMLAAttention,
            kda_backend=cp_plan.kda_backend,
            mla_backend=cp_plan.mla_backend,
        )
        # Perf-test switch: skip checkpoint reading, build the network on the
        # meta device, then materialize per-rank shards after FSDP sharding.
        skip_weight_load = self.config.model.skip_weight_load
        if skip_weight_load:
            with torch.device("meta"):
                model = model_class.from_config(
                    model_config,
                    attn_implementation=self.config.model.attn_implementation,
                    trust_remote_code=False,
                )
        else:
            model = model_class.from_pretrained(
                self.config.model.model_name_or_path,
                config=model_config,
                attn_implementation=self.config.model.attn_implementation,
                trust_remote_code=False,
            )
        convert_model_dtype(model, self.config.model.torch_dtype)
        if self.config.memory.chunk_loss:
            # The CP-aware forward reads these attributes off the top-level model
            # (see fsdp_turbo/models/kimi/kimi_k3_cp.py); the switches live in
            # config.yaml under ``memory``.
            # The chunked loss runs inside the head's module call, so lm_head's
            # gradient goes through FSDP's reduce-scatter like any other sharded
            # unit. Install the forward before FSDPTurbo wraps the model.
            from fsdp_turbo.memory.chunk_loss.chunkloss_lm_head import apply_chunkloss_module

            apply_chunkloss_module(model.language_model.lm_head)
            model.enable_chunk_loss = True
            model.chunk_size = self.config.memory.chunk_loss_size

        print_rank(logger.info, "> Applying FSDPTurbo parallelism...")
        model = FSDPTurbo(self.config, model)

        if skip_weight_load:
            # Materialize only this rank's shard (host-side under cpu_offload;
            # buffers always follow the compute device). Zero-fill keeps
            # startup collective-free; pair with ep_plan.fixed_router so the
            # zeroed router does not skew MoE a2a load balance.
            device = (
                torch.device("cpu")
                if self.config.distributed.fsdp_plan.cpu_offload
                else torch.accelerator.current_accelerator()
            )
            model.to_empty(device=device)
            if device.type == "cpu":
                compute_device = torch.accelerator.current_accelerator()
                for buffer in model.buffers():
                    buffer.data = buffer.data.to(compute_device)
            with torch.no_grad():
                for param in model.parameters():
                    param.zero_()
                for buffer in model.buffers():
                    buffer.zero_()

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


def parse_args() -> argparse.Namespace:
    """Parse the YAML path and optional language-model layer count."""
    parser = argparse.ArgumentParser(description="Kimi K3 MoE Training with Ulysses Context Parallel")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--num-hidden-layers", type=int, default=None, help="Override language model layer count")
    return parser.parse_args()


def main() -> None:
    """Load the YAML configuration and launch Kimi K3 training."""
    args = parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    config = load_config_from_yaml(config_path)

    if args.num_hidden_layers is not None:
        config = replace(
            config,
            model=replace(config.model, num_hidden_layers=args.num_hidden_layers),
        )

    trainer = KimiK3Trainer(config=config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()

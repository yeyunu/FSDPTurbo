import argparse
import logging
import os


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


class Qwen3MoeTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def build_model(self):
        from transformers.models.qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeConfig
        import modeling_qwen3_moe  # noqa: F401  patches RMSNorm

        print_rank(logger.info, "> Building Qwen3-30B-A3B Model...")
        # Qwen3-30B-A3B architecture: 30.5B total / 3.3B activated params,
        # 128 experts (8 activated per token), MoE.
        # num_hidden_layers reduced to 1 for demo; full model uses 48.
        model_config = Qwen3MoeConfig(
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=1,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            hidden_act="silu",
            max_position_embeddings=40960,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=False,
            tie_word_embeddings=False,
            rope_theta=1000000.0,
            attention_dropout=0.0,
            attention_bias=False,
            decoder_sparse_step=1,
            moe_intermediate_size=768,
            num_experts_per_tok=8,
            num_experts=128,
            norm_topk_prob=True,
            output_router_logits=False,
            router_aux_loss_coef=0.001,
            mlp_only_layers=[],
            max_window_layers=48,
            use_sliding_window=False,
            sliding_window=None,
            rope_scaling=None,
        )

        model = Qwen3MoeForCausalLM(model_config).npu()
        convert_model_dtype(model, self.config.model.torch_dtype)

        print_rank(logger.info, "> Applying FSDPTurbo parallelism...")
        model = FSDPTurbo(self.config, model)
        return model

    def train_step(self, batch):
        batch['use_cache'] = False
        return super().train_step(batch)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-30B-A3B Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    config = load_config_from_yaml(config_path)

    trainer = Qwen3MoeTrainer(config=config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()

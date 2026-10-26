import argparse
import logging
import os
import torch

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import load_config_from_yaml
from fsdp_turbo.training.trainer import BaseTrainer
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.model import convert_model_dtype, set_num_hidden_layers

logger = logging.getLogger(__name__)


class Glm53FlashTrainer(BaseTrainer):
    def __init__(self, config, features=None):
        super().__init__(config)
        # Runtime feature toggles read from the raw yaml 'features:' section
        # (mirrors examples/deepseekv4); only chunk_loss lives here now — the
        # DSA/indexer/attention switches moved to the 'model:' section of the
        # config and are consumed via self.config.model.
        self.features = features or {}

    def build_model(self):
        from transformers import AutoConfig

        print_rank(logger.info, "> Building GLM5.3-flash Model...")

        model_config = AutoConfig.from_pretrained(self.config.model.model_name_or_path)
        model_config._attn_implementation = "eager"
        # Propagate to the text sub-config as well: the local modeling file builds
        # the language model directly from `config.text_config`, and an unset value
        # there may be auto-resolved to "sdpa" and bypass the local attention
        # function (where the NPU fused-attention dispatch lives).
        if getattr(model_config, "text_config", None) is not None:
            model_config.text_config._attn_implementation = "eager"
        # DSA attention implementation, read from the model config (config.yaml
        # 'model:' section) so the checkpoint's config.json stays untouched.
        # Must be set before from_pretrained: Glm5NextTextAttention (and its
        # indexer's pack_indices_for_sparse_kernel) read it at __init__ time.
        dsa_implementation = self.config.model.dsa_implementation
        if dsa_implementation not in ("dense", "sfa"):
            raise ValueError(f"model.dsa_implementation must be 'dense' or 'sfa', got {dsa_implementation!r}.")
        if getattr(model_config, "text_config", None) is not None:
            model_config.text_config.dsa_implementation = dsa_implementation
        print_rank(logger.info, f"> DSA implementation: {dsa_implementation}")
        # Indexer scoring implementation, same config->text_config route. 'triton'
        # keeps the per-head scores on-chip and applies the validity gate in the
        # kernel, which is what dsa_implementation='sfa' requires (the cann
        # use_fused_lightning_indexer cannot guarantee an all-valid selection).
        indexer_implementation = self.config.model.indexer_implementation
        if getattr(model_config, "text_config", None) is not None:
            model_config.text_config.indexer_implementation = indexer_implementation
        print_rank(logger.info, f"> Indexer implementation: {indexer_implementation}")
        # NPU fused-attention dispatch switch (MLA/DSA text attention and the
        # vision tower), read from the model config (config.yaml 'model:'
        # section) so the checkpoint's config.json stays untouched. Accepts
        # yaml 1/0/true/false; disabled keeps the bit-exact eager path
        # (loss / grad_norm alignment).
        npu_fused_attention = self.config.model.npu_fused_attention
        if npu_fused_attention not in (0, 1, True, False):
            raise ValueError(f"model.npu_fused_attention must be 1/0/true/false, got {npu_fused_attention!r}.")
        use_npu_fused_attention = bool(npu_fused_attention)
        for sub_config_name in ("text_config", "vision_config"):
            sub_config = getattr(model_config, sub_config_name, None)
            if sub_config is not None:
                sub_config.use_npu_fused_attention = use_npu_fused_attention
        print_rank(logger.info, f"> NPU fused attention: {use_npu_fused_attention}")
        set_num_hidden_layers(model_config, self.config.model.num_hidden_layers)

        # The checkpoint may carry a fine-grained fp8 quantization_config. Its
        # weights are stored as fp8 + per-block scales, so the HF quantizer must
        # run to dequantize them to bf16 at load time. The quantizer's
        # validate_environment checks a CUDA compute capability and crashes on
        # NPU; bypass only that check and keep the dequantization path intact.
        if getattr(model_config, "quantization_config", None):
            from transformers.quantizers import quantizer_finegrained_fp8 as _fp8_mod

            _orig_validate = _fp8_mod.FineGrainedFP8HfQuantizer.validate_environment

            def _npa_validate_environment(self, *args, **kwargs):
                try:
                    return _orig_validate(self, *args, **kwargs)
                except TypeError:
                    # compute_capability is None on NPU; the fp8 dequant path
                    # itself only needs plain torch ops.
                    print_rank(logger.info, "> Skipping CUDA-only fp8 environment check (NPU).")
                    return None

            _fp8_mod.FineGrainedFP8HfQuantizer.validate_environment = _npa_validate_environment
            print_rank(
                logger.info,
                f"> Checkpoint carries quantization_config "
                f"({model_config.quantization_config.get('quant_method')}); "
                f"weights will be dequantized to {self.config.model.torch_dtype} on load.",
            )

        # Use the local (NPU-customized) modeling file instead of the copy shipped
        # inside transformers: AutoModelForImageTextToText resolves "glm5_next" to
        # transformers' own modeling_glm5_next.py, which silently bypasses all
        # fsdp_turbo customizations (NPU fused attention switch, NPU op dispatches).
        from fsdp_turbo.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration

        model, loading_info = Glm5NextForConditionalGeneration.from_pretrained(
            self.config.model.model_name_or_path,
            config=model_config,
            torch_dtype=torch.bfloat16,
            output_loading_info=True,
        )

        # Dequantize fp8 expert weights (if any) to bf16 in place BEFORE the
        # blind convert_model_dtype (.to(bf16) on a fp8 param drops its scale
        # and silently corrupts the weights). Must also run before saving and
        # FSDPTurbo wrapping; the EP forward methods live on
        # Glm5NextTextExperts itself (no patching involved).
        from fsdp_turbo.models.glm5_next.glm5_fp8_dequant import dequantize_glm5_experts_fp8

        dequantized = dequantize_glm5_experts_fp8(model)
        print_rank(logger.info, f"> Dequantized fp8 on {dequantized} modules")

        convert_model_dtype(model, self.config.model.torch_dtype)

        if self.features.get("chunk_loss"):
            # Chunked lm_head loss (docs/zh/features/chunk_loss.md): enabled via
            # model attributes. Works standalone and with Ulysses CP (the model
            # forward has a combined path). Orthogonal to memory.recompute.
            model.enable_chunk_loss = True
            model.chunk_size = int(self.features.get("chunk_loss_size", 1024))
            print_rank(logger.info, f"> Chunk loss enabled (chunk_size={model.chunk_size})")

        if self.config.model.use_fused_lightning_indexer:
            if dsa_implementation == "sfa":
                # Same incompatibility the indexer guards on at forward time;
                # failing here keeps a 16-rank job from dying after DDP setup.
                raise ValueError(
                    "model.use_fused_lightning_indexer cannot be combined with "
                    "model.dsa_implementation: 'sfa': the cann op cannot guarantee "
                    "an all-valid pool selection, which SFA's packed indices require. "
                    "Set use_fused_lightning_indexer: false and "
                    "indexer_implementation: 'triton' instead."
                )
            # Fused lightning indexer (cann_ops_transformer via
            # fsdp_turbo.ops.npu): on-chip DSA scoring + pool top-k, avoiding
            # the eager [B, S, NH, P] fp32 score tensor (the 32k OOM source).
            # The indexer modules are already built at load time, so set the
            # runtime attributes on the instances (same effect as
            # text_config.use_fused_lightning_indexer). NPU-only; a missing
            # cann_ops_transformer fails fast here.
            from fsdp_turbo.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer
            from fsdp_turbo.ops.npu.lightning_indexer_bsnd import lightning_indexer_bsnd

            n_indexers = 0
            for module in model.modules():
                if isinstance(module, Glm5NextTextIndexer):
                    module.use_fused_lightning_indexer = True
                    module._lightning_indexer_bsnd = lightning_indexer_bsnd
                    n_indexers += 1
            print_rank(logger.info, f"> Fused lightning indexer enabled on {n_indexers} indexer modules")

        print_rank(logger.info, "> Applying FSDPTurbo parallelism...")
        model = FSDPTurbo(self.config, model)
        print_rank(logger.info, "> Model built successfully")
        return model

    def train_step(self, batch):
        batch['use_cache'] = False
        return super().train_step(batch)


def parse_args():
    parser = argparse.ArgumentParser(description="GLM5.3-flash")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--model-name", type=str, default=None, help="Override model name or path")
    parser.add_argument("--ulysses-size", type=int, default=None, help="Override Ulysses parallel size")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    config = load_config_from_yaml(config_path)

    # Runtime feature toggles live in the raw yaml 'features:' section.
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        features = (yaml.safe_load(f) or {}).get("features") or {}

    if args.model_name:
        config.model.model_name_or_path = args.model_name
    if args.ulysses_size:
        config.distributed.ulysses_parallel_size = args.ulysses_size

    trainer = Glm53FlashTrainer(config=config, features=features)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()

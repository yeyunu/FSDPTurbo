# Model Adaptation Guide

This document describes how to integrate a new Hugging Face Transformers model into FSDPTurbo for distributed training. Refer to the existing examples under `examples/` (e.g. `qwen3_5`).

## Adaptation Workflow Overview

1. Prepare the example directory: `config.yaml` + `train.py` + `run.sh`
2. Write the training entry point `train.py`: build the model → apply FSDPTurbo → start the Trainer
3. Write the training configuration `config.yaml`: configure the parallelism plans as needed (fsdp_plan/tp_plan/cp_plan/ep_plan)
4. Write the launch script `run.sh`: start with torchrun

## Writing the Training Entry Point

Load the model using the native Transformers approach, then apply parallelization with `FSDPTurbo(config, model)`:

```python
import torch
from transformers import AutoConfig, AutoModel

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import load_config_from_yaml
from fsdp_turbo.training.trainer import BaseTrainer


class MyTrainer(BaseTrainer):
    def build_model(self):
        model_config = AutoConfig.from_pretrained(self.config.model.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name_or_path,
            config=model_config,
            torch_dtype=torch.bfloat16,
        )
        # Apply parallelization with one line of code
        model = FSDPTurbo(self.config, model)
        return model


def main():
    config = load_config_from_yaml("config.yaml")
    trainer = MyTrainer(config=config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
```

## Configuring Parallelism Plans

### Determining Module Names

Use `model.named_modules()` to inspect the model's module paths. All FSDPTurbo plans are based on module name matching and support wildcards:

- `{*}`: matches a single-layer index, e.g. `model.layers.{*}`
- `*`: matches any prefix, e.g. `*.q_proj`

### FSDP2 Sharding Plan (Required)

Configure the FSDP2 sharding plan in the `distributed` section, specifying the modules to shard and the precision:

```yaml
distributed:
  fully_shard_parallel_size: 8   # Recommended: equal to the number of cards per node
  fsdp_plan:
    apply_modules:               # Modules to shard with FSDP2
      model.layers.{*}: {}
      model.embed_tokens: {}
      lm_head: {}
    param_dtype: "bf16"
    reduce_dtype: "fp32"
    num_to_forward_prefetch: 1
    num_to_backward_prefetch: 1
```

### Tensor Parallelism/Context Parallelism (Optional)

Configure tensor parallelism and context parallelism in the `distributed` section as needed:

```yaml
distributed:
  tensor_parallel_size: 2
  tp_plan:
    colwise_parallel: ['*.q_proj', '*.k_proj', '*.v_proj']
    rowwise_parallel: ['*.o_proj']

  ulysses_parallel_size: 2
  cp_plan:
    ulysses_function_patches:
      - target_functions:
          - transformers.models.xxx.modeling_xxx.eager_attention_forward
        type: full_attention
    loss_function_patches:
      - target_functions:
          - transformers.loss.loss_utils.ForCausalLMLoss
        type: causal_lm_loss
```

> [!NOTE]
>
> Ulysses CP may require replacing the model forward via `module_patches` (refer to `examples/qwen3_5/config.yaml`):
>
>```yaml
>module_patches:
>  - target: transformers.models.xxx.modeling_xxx.XxxModel.forward
>    replacement: fsdp_turbo.models.qwen.qwen3_5.qwen3_5_model_forward
>```

### Expert Parallelism (MoE Models, Optional)

MoE models can enable expert parallelism as needed:

```yaml
distributed:
  expert_parallel_size: 8
  ep_plan:
    apply_modules:
      - model.layers.{*}.mlp.experts
    dispatcher: "fused"        # eager / fused / mc2 / domino
    apply_efsdp_modules:
      - model.layers.{*}.mlp.experts
```

> [!NOTE]
>
> The EP path requires the experts module to hold 3D tensor parameters (`gate_up_proj`, `down_proj`) rather than independent expert submodules in ModuleList form. If the model's expert computation semantics are non-standard, use `dispatcher='custom_ep_forward_eager'` / `'custom_ep_forward_fused'` and implement the following methods on the expert module yourself:
>
> ```python
> ep_forward_eager(hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router)
> ep_forward_fused(hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router)
> ```

## Writing the Launch Script

Start distributed training with torchrun:

```bash
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config config.yaml
```

## Verification Recommendations

- First trim the number of layers with `--num-hidden-layers` (if supported by the model) and verify that forward and backward pass through on a small-scale parallelism configuration.
- Compare the loss curves of single-card/single-node and multi-node configurations to confirm that parallelism introduces no precision anomalies.

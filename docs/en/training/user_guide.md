# Model Usage Guide

FSDPTurbo provides an end-to-end large language model training solution covering distributed pretraining and distributed fine-tuning.

In [Quick Start](./quick_start_fsdpturbo.md), you can quickly master the pretraining and fine-tuning tasks of large language models based on the Qwen3.5-27B model. The following table describes in detail how to use models with FSDPTurbo.

**Table 1** Model training solutions and usage instructions

| Category | Content | Application Scenarios | Description |
|---|---|---|---|
| Pretraining | [Distributed Pretraining of Large Models](./pretrain/pretrain.md) | Offline pretraining of models based on large-scale unlabeled corpora | FSDPTurbo directly loads Hugging Face-format weights with no offline weight conversion; the dtype conversion is done by the training script according to the `torch_dtype` configuration. Data is automatically loaded and tokenized at startup, with no preprocessing required. |
| Fine-tuning | [Model Fine-Tuning](./finetune/finetune.md) | Full-parameter fine-tuning based on a pretrained model using instruction/conversation data | The model size, sequence length, batch size, and parallelism scale need to be adjusted according to the number of devices and the memory capacity. |

## Training Workflow Overview

The FSDPTurbo training workflow is as follows:

1. **Write the training entry point** (`train.py`): Load the model in the native Transformers way and apply parallelization with `FSDPTurbo(config, model)`.
2. **Write the training configuration** (`config.yaml`): Configure the model path, optimizer, dataset, parallel strategy, and memory optimization.
3. **Write the launch script** (`run.sh`): Launch distributed training via `torchrun`.

Core code:

```python
import torch
from transformers import AutoModelForCausalLM

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import load_config_from_yaml
from fsdp_turbo.training.trainer import BaseTrainer


class MyTrainer(BaseTrainer):
    def build_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        return FSDPTurbo(self.config, model)   # Apply parallelization with one line of code


config = load_config_from_yaml("config.yaml")
trainer = MyTrainer(config=config)
trainer.setup()
trainer.train()
```

## Differences from Native Transformers Training

The main differences between FSDPTurbo and native Hugging Face Transformers training:

- **Parallelization**: FSDP2/TP/EP/CP multi-dimensional parallelism is applied automatically through `FSDPTurbo(config, model)`, with no need to write distributed code manually.
- **Configuration**: Training parameters and parallel strategies are managed centrally through YAML configuration files.
- **Data loading**: A built-in data loader based on Hugging Face `datasets` supports plain text and multimodal conversation data.
- **Training workflow**: `BaseTrainer` encapsulates the training loop, providing capabilities such as checkpoint-based resumable training, gradient clipping, and log monitoring.

## Typical Training Scenarios

- **Dense model training**: Dense models such as Qwen3.5-27B, configured with FSDP2 + TP + Ulysses CP + recomputation.
- **MoE model training**: MoE models such as Qwen3.5-35B-A3B, adding EP + EFSDP on top of the dense configuration.
- **Multimodal model training**: Multimodal models such as Qwen3.5, loading image-text data through `is_multimodal` and the conversation column configuration.

For details, see the [Supported Models list](../models/supported_models.md) and the [Model Adaptation Guide](../develop/model_adaptation.md).

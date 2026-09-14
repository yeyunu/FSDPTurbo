# 模型使用导读

FSDPTurbo提供端到端的大语言模型训练方案，涵盖分布式预训练与分布式微调。

在 [快速入门](./quick_start_fsdpturbo.md) 中，用户可以基于Qwen3.5-27B模型快速掌握大语言模型的预训练和微调任务，下表详细介绍了FSDPTurbo的模型使用方法。

**表1**模型训练方案及使用说明

| 分类 | 内容 | 使用场景 | 说明 |
|---|---|---|---|
| 预训练 | [大模型分布式预训练](./pretrain/pretrain.md) | 基于大规模无标签语料进行模型离线预训练 | FSDPTurbo直接加载HuggingFace格式权重，无需离线权重转换，dtype转换由训练脚本按`torch_dtype`配置完成；数据启动时自动加载与Tokenize，无需预处理。 |
| 微调 | [模型微调](./finetune/finetune.md) | 基于预训练模型，使用指令/对话数据进行全参数微调 | 模型规模、序列长度、batch size和并行规模需要根据设备数量及显存容量进行调整。 |

## 训练流程总览

FSDPTurbo的训练流程如下：

1. **编写训练入口**（`train.py`）：使用Transformers原生方式加载模型，一行代码`FSDPTurbo(config, model)`应用并行化。
2. **编写训练配置**（`config.yaml`）：配置模型路径、优化器、数据集、并行策略与内存优化。
3. **编写启动脚本**（`run.sh`）：通过`torchrun`拉起分布式训练。

核心代码：

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
        return FSDPTurbo(self.config, model)   # 一行代码应用并行化


config = load_config_from_yaml("config.yaml")
trainer = MyTrainer(config=config)
trainer.setup()
trainer.train()
```

## 与Transformers原生训练的差异

FSDPTurbo与HuggingFace Transformers原生训练的主要差异：

- **并行化**：通过`FSDPTurbo(config, model)`自动应用FSDP2/TP/EP/CP多维并行，无需手动编写分布式代码。
- **配置方式**：通过YAML配置文件统一管理训练参数与并行策略。
- **数据加载**：内置基于HuggingFace `datasets`的数据加载器，支持纯文本与多模态对话数据。
- **训练流程**：提供`BaseTrainer`封装训练循环，包含断点续训、梯度裁剪、日志监控等能力。

## 典型训练场景

- **Dense模型训练**：Qwen3.5-27B等Dense模型，配置FSDP2 + TP + Ulysses CP + 重计算。
- **MoE模型训练**：Qwen3.5-35B-A3B等MoE模型，在Dense基础上增加EP + EFSDP。
- **多模态模型训练**：Qwen3.5等多模态模型，通过`is_multimodal`与对话列配置加载图文数据。

详见 [模型支持列表](../models/supported_models.md) 与 [模型适配指南](../develop/model_adaptation.md)。

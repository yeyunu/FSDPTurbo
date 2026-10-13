# 大模型分布式预训练

## 使用场景

大模型预训练（Pretraining）是语言模型发展的核心步骤，目标是让模型通过大规模无标签语料学习语言规律与世界知识。预训练过程更关注语言建模本身，而非具体任务执行。以GPT类模型为例，它是一种典型的自回归语言模型，其核心思想是基于历史上下文预测下一个标记。预训练的过程就是通过反复优化这种预测能力，使模型逐渐学会如何理解语境、保持句子连贯性，并掌握更高层次的语言结构，为多种下游任务提供通用的语言表示能力。

预训练数据通常为纯文本格式，无任务导向，例如：

```json
{"text": "今天是个好天气，我们一起去爬山。"}
{"text": "深度学习正在改变世界。"}
{"text": "AI 的出现推动了人类社会的发展。"}
```

> [!NOTE]
>
> - FSDPTurbo可直接加载HuggingFace格式权重，无需转换为Megatron等格式（加载后的dtype转换由训练脚本按`torch_dtype`配置完成）。若不加载初始权重，则模型权重采用随机初始化，从头开始训练。
> - 预训练与微调使用相同的训练入口（`train.py`），通过YAML配置中的数据格式、优化器参数等区分。

## 使用说明

本文以Qwen3.5-27B模型为例，介绍预训练启动方法。大模型分布式预训练主要包含以下流程：

1. 环境搭建
2. 模型权重准备
3. 预训练数据准备
4. 配置预训练参数
5. 启动预训练

### 环境搭建

启动预训练前请参考[软件安装](../install_guide.md)完成环境安装，并确保已完成相关环境变量配置：

```shell
source /usr/local/Ascend/cann/set_env.sh
```

以上命令以root用户安装后的默认路径为例，请根据set_env.sh的实际路径进行替换。

### 模型权重准备

FSDPTurbo直接加载HuggingFace格式权重，无需转换为Megatron等格式；权重加载后，训练脚本会按`torch_dtype`配置对模型参数与buffer做dtype转换。获取Qwen3.5-27B权重：

```shell
mkdir -p ./model_from_hf/qwen3_5_hf
huggingface-cli download Qwen/Qwen3.5-27B --local-dir ./model_from_hf/qwen3_5_hf
```

### 预训练数据准备

FSDPTurbo数据加载基于HuggingFace `datasets`实现，**无需提前进行数据预处理**，训练启动时自动完成加载与Tokenize。支持本地数据文件（`.json`、`.jsonl`、`.csv`、`.parquet`、`.txt`等）或HuggingFace Hub数据集。

常见的预训练数据集有：

- [Alpaca数据集](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [EnWiki数据集](https://huggingface.co/datasets/lsb/enwiki20230101)
- [C4数据集](https://huggingface.co/datasets/allenai/c4)
- [ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText)

预训练数据需包含`text`列，格式如下：

```json
[
    {"text": "document 1"},
    {"text": "document 2"}
]
```

在`config.yaml`的`data`段配置数据集：

```yaml
data:
  dataset_path: "./dataset/train.json"    # 本地文件路径或 HuggingFace 数据集名
  text_column: "text"                      # 文本列名
  split: "train"
  batch_size: 1
  max_seq_length: 4096
  num_workers: 8
```

### 配置预训练参数

模型权重路径、数据集路径、并行规模和输出目录等参数配置在`config.yaml`中。详细配置请参考示例`examples/qwen3_5/config.yaml`。预训练典型配置要点：

- 优化器学习率较大（如`3.0e-4`）
- 数据使用纯文本格式（`text_column`）
- 开启激活重计算以降低显存占用

```yaml
model:
  model_name_or_path: "./model_from_hf/qwen3_5_hf/"
  tokenizer_name_or_path: "./model_from_hf/qwen3_5_hf/"
  torch_dtype: "bf16"

optimizer:
  optimizer_type: "AdamW"
  lr: 3.0e-4
  weight_decay: 0.1
  warmup_ratio: 0.03
  lr_scheduler_type: "CosineAnnealingLR"
  min_lr: 3.0e-5
  clip_grad: 1.0

run:
  seed: 42
  max_steps: 2000
  gradient_accumulation_steps: 1
  logging_steps: 1

data:
  dataset_path: "./dataset/train.json"
  text_column: "text"
  split: "train"
  batch_size: 1
  max_seq_length: 4096
  num_workers: 8

distributed:
  fully_shard_parallel_size: 8
  tensor_parallel_size: 1
  ulysses_parallel_size: 2
  fsdp_plan:
    param_dtype: "bf16"
    reduce_dtype: "fp32"
    num_to_forward_prefetch: 1
    num_to_backward_prefetch: 1
    hook_modules:
      - model.language_model.layers.{*}

memory:
  recompute: true
  recompute_plan:
    - model.language_model.layers.{*}
```

### 启动预训练

参数配置完成后，在示例目录执行：

```shell
cd examples/qwen3_5
bash run.sh
```

`run.sh`内部通过`torchrun`拉起训练，集群参数可通过环境变量覆盖：

```bash
NNODES=1                 # 节点数，单机为 1
GPUS_PER_NODE=16         # 单节点卡数
MASTER_ADDR=127.0.0.1    # 主节点 IP
MASTER_PORT=29500        # 主节点端口
```

执行一段时间后，可在终端看到如下日志：

```shell
Step 1 | Loss: 1.304194 | Grad Norm: 8.5150 | Step Time: 3.2145s | Memory: 28.3124 GB | Peak Memory: 46.8732 GB
Step 2 | Loss: 1.232217 | Grad Norm: 4.3460 | Step Time: 2.8761s | Memory: 28.3056 GB | Peak Memory: 46.8732 GB
Step 3 | Loss: 1.134654 | Grad Norm: 1.5500 | Step Time: 2.7934s | Memory: 28.2988 GB | Peak Memory: 46.8732 GB
```

终端开始持续输出步数、损失值、梯度范数、单步耗时与显存占用等信息，表示训练任务已经正常运行。

> [!NOTE]
>
> - 多机训练需在多个终端同时启动预训练脚本（每个终端的`NODE_RANK`不同，`MASTER_ADDR`/`MASTER_PORT`相同），由torchrun组网。
> - 多机训练时，请确保各节点均能正确访问模型和数据集路径。
> - 训练权重保存在`checkpoint.output_dir`指定的目录。

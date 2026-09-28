# 模型微调

## 使用场景

监督微调（Supervised Fine-Tuning，SFT）通过高质量的指令和回复数据继续训练预训练模型，使模型学习特定任务、领域知识或对话方式。该方法适用于问答、文本生成、摘要、翻译、代码生成和领域适配等场景。

> [!NOTE]
>
> 若您是首次使用FSDPTurbo，建议先按 [快速入门](../quick_start_fsdpturbo.md)（Qwen3.5-27B预训练与微调）完成一次端到端跟练。本文档侧重全参数微调的模型、数据集、YAML配置与参数说明，适用于更换模型或自定义数据集场景。

本文介绍如何基于HuggingFace格式的预训练模型，使用FSDPTurbo完成全参数微调。示例采用Qwen3.5-27B模型，主要流程如下：

1. 环境搭建
2. 模型权重准备
3. 数据集准备
4. 配置微调参数
5. 配置启动脚本
6. 启动微调

## 环境搭建

启动微调前，请参考 [软件安装](../install_guide.md) 完成环境安装。

NPU环境下训练前需加载CANN环境：

```shell
source /usr/local/Ascend/cann/set_env.sh
```

## 模型权重准备

模型权重下载地址请参考 [模型支持列表](../../models/supported_models.md)。本示例使用Qwen3.5-27B的HuggingFace格式权重。

FSDPTurbo可直接加载HuggingFace格式权重，**开始训练前不需要执行模型权重转换**；权重加载后，训练脚本会按`torch_dtype`配置对模型参数与buffer做dtype转换。在`config.yaml`的`model.model_name_or_path`中配置模型权重目录即可：

```yaml
model:
  model_name_or_path: "./model_from_hf/qwen3_5_hf/"
  tokenizer_name_or_path: "./model_from_hf/qwen3_5_hf/"
  torch_dtype: "bf16"
```

## 数据集准备

FSDPTurbo数据加载基于HuggingFace `datasets`实现，支持本地数据文件或HuggingFace Hub数据集，训练任务启动时完成加载、格式对齐和Tokenize，无需提前转换为Megatron Indexed Dataset。

### 支持的数据格式

- **纯文本格式**：每条样本包含`text`字段，用于预训练或已拼接好的指令微调数据。

    ```json
    [
      {"text": "用户：请介绍 FSDPTurbo\n助手：FSDPTurbo 是一个分布式训练加速库..."}
    ]
    ```

- **对话格式**：通过`conversation_column`指定对话列名，每条消息包含`role`和`content`字段，适用于指令微调与多模态微调。

    ```json
    [
      {
        "messages": [
          {"role": "user", "content": "请介绍一下 FSDPTurbo"},
          {"role": "assistant", "content": "FSDPTurbo 是一个分布式训练加速库..."}
        ]
      }
    ]
    ```

    多模态场景下，在样本中添加`images`字段并在`content`中使用`<image>`占位符：

    ```json
    [
      {
        "messages": [
          {"role": "user", "content": "<image>\n请描述这张图片"},
          {"role": "assistant", "content": "这张图片展示了..."}
        ],
        "images": ["/path/to/image1.jpg"]
      }
    ]
    ```

> [!NOTE]
>
> 本地文件支持`.json`、`.jsonl`、`.csv`、`.parquet`、`.txt`等格式，由HuggingFace `datasets`自动识别。多模态数据目前仅支持JSON格式文件。

## 配置微调参数

模型权重路径、数据集路径、并行规模和输出目录等参数配置在`config.yaml`中。详细配置请参考示例`examples/qwen3_5/config.yaml`。YAML示例内容如下：

```yaml
model:
  model_name_or_path: "./model_from_hf/qwen3_5_hf/"
  tokenizer_name_or_path: "./model_from_hf/qwen3_5_hf/"
  torch_dtype: "bf16"

optimizer:
  optimizer_type: "AdamW"
  lr: 1.0e-5
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1e-8
  clip_grad: 1.0
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.0
  min_lr: 1.0e-6

run:
  seed: 42
  max_steps: 2000
  gradient_accumulation_steps: 1
  logging_steps: 1

checkpoint:
  output_dir: "./output/qwen3_5_27b"
  save_steps: 500

data:
  dataset_path: "./dataset/finetune.json"
  conversation_column: "messages"
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

YAML中的主要参数说明如下：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `model_name_or_path` | HuggingFace模型权重目录路径。 | — |
| `dataset_path` | 训练数据集路径，支持本地文件或HuggingFace数据集名。 | — |
| `conversation_column` | 对话格式数据的列名。 | `messages` |
| `max_seq_length` | 分词后训练序列的最大长度，超过该长度的内容将被截断。 | `4096` |
| `fully_shard_parallel_size` | FSDP2全分片数据并行大小，应与单节点卡数匹配。 | `1` |
| `recompute` | 是否启用激活重计算，以计算开销换取显存空间。 | `false` |
| `lr` | 初始学习率。 | `1e-4` |
| `max_steps` | 最大训练步数；大于0时覆盖`num_train_epochs`。 | `-1` |
| `save_steps` | 保存训练检查点的步数间隔。 | `500` |

## 配置启动脚本

打开对应的`run.sh`文件，配置分布式训练参数。详细配置请参考示例`examples/qwen3_5/run.sh`。

单机16卡配置如下：

```bash
NNODES=1                              # 节点总数，单机配置为 1
GPUS_PER_NODE=16                      # 单节点卡数
MASTER_ADDR=127.0.0.1                 # 主节点 IP，单机可配置为 127.0.0.1
MASTER_PORT=29500                     # 主节点通信端口
```

多机配置示例如下：

```bash
NNODES=2                              # 参与训练的节点总数
GPUS_PER_NODE=16                      # 每个节点的卡数
MASTER_ADDR="主节点 IP"               # 所有节点均配置为主节点 IP，不能使用 localhost
MASTER_PORT=29500                     # 所有节点保持相同的主节点通信端口
```

> [!NOTE]
>
> 不同节点的`MASTER_ADDR`、`MASTER_PORT`和`NNODES`应保持一致，`NODE_RANK`（由torchrun `--nnodes`与`--node_rank`控制）从0开始且不能重复。

`run.sh`内部通过`torchrun`拉起训练：

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

## 启动微调

参数配置完成后，在示例目录执行：

```shell
cd examples/qwen3_5
bash run.sh
```

多机训练时，需要在所有节点执行启动脚本，并分别设置对应的`NODE_RANK`。训练检查点保存在`checkpoint.output_dir`指定的目录。

执行一段时间后，可在终端看到如下日志：

```shell
Step 1 | Loss: 3.316887 | Grad Norm: 70.9590 | Step Time: 3.0872s | Memory: 28.3124 GB | Peak Memory: 46.8732 GB
Step 2 | Loss: 2.443476 | Grad Norm: 41.9860 | Step Time: 2.8145s | Memory: 28.3056 GB | Peak Memory: 46.8732 GB
Step 3 | Loss: 2.669216 | Grad Norm: 45.3350 | Step Time: 2.7691s | Memory: 28.2988 GB | Peak Memory: 46.8732 GB
```

终端开始持续输出步数、损失值、梯度范数、单步耗时与显存占用等信息，表示训练任务已经正常运行。

## 使用约束

- 本指南适用于FSDPTurbo全参数SFT，不涵盖LoRA、DPO等其他后训练方法。
- `model_name_or_path`、数据集和输出目录应使用训练环境中可访问的有效路径。
- 模型规模、序列长度、batch size和并行规模需要根据设备数量及显存容量进行调整。
- 多机训练时，请确保各节点均能正确访问模型和数据集路径。

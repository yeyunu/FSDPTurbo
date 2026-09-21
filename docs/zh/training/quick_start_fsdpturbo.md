# 快速入门：Qwen3.5模型预训练及微调

## 概述

本文档提供一个简易示例，帮助初次接触FSDPTurbo的开发者快速启动模型训练任务，并基于预训练模型完成微调任务。以下将以Qwen3.5-27B模型为例，指导开发者完成大语言模型的预训练和微调，主要步骤包括：

- **环境准备**：安装FSDPTurbo及其依赖
- **模型和数据集准备**：获取Qwen3.5-27B模型权重与训练数据
- **启动训练任务**：在NPU上进行模型预训练和微调

> [!NOTE]
>
> - FSDPTurbo支持 <term>Ascend 950PR&950DT系列产品</term>、<term>Atlas A3训练系列产品</term> 和 <term>Atlas A2训练系列产品</term>，要求单卡显存满足模型规模需求，详见[模型支持列表](../models/supported_models.md)。
>
> - 本文档为 **快速入门跟练教程**。若需更换模型或注册自定义数据集，请参考 [模型适配指南](../develop/model_adaptation.md)。
>
> - 当前Qwen3.5-27B的示例脚本中`GPUS_PER_NODE=16`表示需要16张卡，如果实际情况低于此配置，需相应调整并行度参数，否则可能遇到OOM（Out Of Memory，内存不足）问题。

开发者入门基础：

- 具备基础的PyTorch使用经验
- 具备初级的Python开发经验
- 对HuggingFace Transformers有基本的了解

## 环境准备

### 环境搭建

环境搭建请参考[软件安装](./install_guide.md)。

### 模型权重准备

FSDPTurbo直接加载HuggingFace格式权重，**无需进行任何权重格式转换**；权重加载后，训练脚本会按`torch_dtype`配置对模型参数与buffer做dtype转换。获取Qwen3.5-27B权重：

```shell
# 创建一个目录存储权重文件
mkdir -p ./model_from_hf/qwen3_5_hf

# 方式一：huggingface-cli 下载
huggingface-cli download Qwen/Qwen3.5-27B --local-dir ./model_from_hf/qwen3_5_hf

# 方式二：训练配置中直接填写 "Qwen/Qwen3.5-27B"，框架自动下载
```

完整模型目录应包含模型配置、Tokenizer配置和全部权重文件。

### 数据集准备

FSDPTurbo数据加载基于HuggingFace `datasets`实现，支持本地数据文件或HuggingFace Hub数据集，训练任务启动时完成加载与Tokenize，无需提前转换为Megatron Indexed Dataset。

**预训练数据**：纯文本格式，每条样本包含`text`字段。

```json
[
  {"text": "训练样本文本内容..."}
]
```

**微调数据**：支持对话格式，通过`conversation_column`指定对话列名（多模态场景），或通过`text_column`指定已拼接好的文本列（纯文本场景）。

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

> [!NOTE]
>
> 本地文件支持`.json`、`.jsonl`、`.csv`、`.parquet`、`.txt`等格式，由HuggingFace `datasets`自动识别。

## 启动预训练

在这一阶段，我们将修改预训练配置文件，启动模型预训练，具体步骤如下：

1. 编辑训练配置文件。

    ```shell
    vi examples/qwen3_5/config.yaml
    ```

2. 修改并保存训练参数配置，配置示例如下：

    ```yaml
    # --- 模型配置 ---
    model:
      model_name_or_path: "./model_from_hf/qwen3_5_hf/"   # 模型权重路径
      tokenizer_name_or_path: "./model_from_hf/qwen3_5_hf/"  # 词表路径
      torch_dtype: "bf16"

    # --- 优化器配置 ---
    optimizer:
      optimizer_type: "AdamW"
      lr: 3.0e-4
      weight_decay: 0.1
      adam_beta1: 0.9
      adam_beta2: 0.95
      warmup_ratio: 0.03
      lr_scheduler_type: "CosineAnnealingLR"
      min_lr: 3.0e-5
      clip_grad: 1.0

    # --- 运行配置 ---
    run:
      seed: 42
      max_steps: 2000
      gradient_accumulation_steps: 1
      logging_steps: 1

    # --- 数据配置 ---
    data:
      dataset_path: "./dataset/train.json"
      text_column: "text"
      split: "train"
      batch_size: 1
      max_seq_length: 4096
      num_workers: 8

    # --- 分布式配置 ---
    distributed:
      fully_shard_parallel_size: 8       # FSDP2 并行度（建议 = 单节点卡数）
      tensor_parallel_size: 1            # 张量并行度
      ulysses_parallel_size: 2           # Ulysses 上下文并行度
      fsdp_plan:
        param_dtype: "bf16"              # 计算精度
        reduce_dtype: "fp32"             # 梯度归约精度
        num_to_forward_prefetch: 1       # 前向预取层数
        num_to_backward_prefetch: 1      # 反向预取层数
        hook_modules:
          - model.language_model.layers.{*}

    # --- 内存配置 ---
    memory:
      recompute: true
      recompute_plan:
        - model.language_model.layers.{*}
    ```

3. 执行预训练脚本。

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

    执行后一段时间可见终端显示：

    ```shell
    Step 1 | Loss: 1.304194 | Grad Norm: 8.5150 | Step Time: 3.2145s | Memory: 28.3124 GB | Peak Memory: 46.8732 GB
    Step 2 | Loss: 1.232217 | Grad Norm: 4.3460 | Step Time: 2.8761s | Memory: 28.3056 GB | Peak Memory: 46.8732 GB
    Step 3 | Loss: 1.134654 | Grad Norm: 1.5500 | Step Time: 2.7934s | Memory: 28.2988 GB | Peak Memory: 46.8732 GB
    ```

    进入迭代，说明训练正常进行。

> [!NOTE]
>
> - 多机训练需在多个终端同时启动预训练脚本（每个终端的`NODE_RANK`不同，`MASTER_ADDR`/`MASTER_PORT`相同），由torchrun组网。
> - FSDP2后端会自动将模型参数分片到各设备，确保每个设备只存储部分参数，从而支持超大规模模型训练。
> - 训练权重保存在`checkpoint.output_dir`指定的目录。

## 启动微调

在这一阶段，我们将基于预训练权重，使用指令数据完成模型微调，具体步骤如下：

1. 编辑训练配置文件。

    ```shell
    vi examples/qwen3_5/config.yaml
    ```

2. 修改并保存微调参数配置。与预训练相比，主要调整优化器与数据配置：

    ```yaml
    # --- 优化器配置（微调学习率更小）---
    optimizer:
      optimizer_type: "AdamW"
      lr: 1.0e-5
      weight_decay: 0.01
      warmup_ratio: 0.0
      lr_scheduler_type: "cosine"
      min_lr: 1.0e-6
      clip_grad: 1.0

    # --- 数据配置（使用对话格式数据）---
    data:
      dataset_path: "./dataset/finetune.json"
      conversation_column: "messages"
      split: "train"
      batch_size: 1
      max_seq_length: 4096
      num_workers: 8
    ```

3. 执行微调脚本。

    ```shell
    cd examples/qwen3_5
    bash run.sh
    ```

    执行后一段时间可见终端显示：

    ```shell
    Step 1 | Loss: 3.316887 | Grad Norm: 70.9590 | Step Time: 3.0872s | Memory: 28.3124 GB | Peak Memory: 46.8732 GB
    Step 2 | Loss: 2.443476 | Grad Norm: 41.9860 | Step Time: 2.8145s | Memory: 28.3056 GB | Peak Memory: 46.8732 GB
    Step 3 | Loss: 2.669216 | Grad Norm: 45.3350 | Step Time: 2.7691s | Memory: 28.2988 GB | Peak Memory: 46.8732 GB
    ```

    进入迭代，说明训练正常进行。

> [!NOTE]
>
> - 多机微调需在多个终端同时启动微调脚本（每个终端的`NODE_RANK`不同，其余参数相同）。

## 训练脚本参数说明

下表为`config.yaml`中部分关键参数解释：

**表1**训练配置参数说明

| 参数名 | 说明 | 配置示例 |
|---|---|---|
| `model_name_or_path` | HuggingFace权重目录路径，替换为实际模型路径 | `./model_from_hf/qwen3_5_hf/` |
| `dataset_path` | 训练数据集路径，支持本地文件或HuggingFace数据集名 | `./dataset/train.json` |
| `text_column` / `conversation_column` | 文本列名 / 对话列名 | `text` / `messages` |
| `max_seq_length` | 训练序列最大长度 | `4096` |
| `fully_shard_parallel_size` | FSDP2全分片数据并行大小 | `8` |
| `tensor_parallel_size` | 张量并行大小 | `1` |
| `ulysses_parallel_size` | Ulysses上下文并行大小 | `2` |
| `expert_parallel_size` | 专家并行大小（Dense模型配置为`1`） | `1` |
| `recompute` | 是否启用激活重计算，以计算开销换取显存空间 | `true` |
| `batch_size` | 单卡训练batch size | `1` |
| `gradient_accumulation_steps` | 梯度累积步数 | `1` |
| `max_steps` | 最大训练步数 | `2000` |

## MoE模型训练

以Qwen3.5-35B-A3B（`examples/qwen3_5_moe`）为例，在Dense配置基础上增加**专家并行（EP）**配置：

```yaml
distributed:
  fully_shard_parallel_size: 8
  expert_parallel_size: 8             # 专家并行度
  expert_fully_shard_parallel_size: 1 # 专家 FSDP2 并行度（EFSDP）
  ep_plan:
    apply_modules:                    # 应用 EP 的模块（通配符）
      - model.language_model.layers.{*}.mlp.experts
    dispatcher: "fused"               # 调度器：eager / fused / mc2 / domino
    apply_efsdp_modules:              # 应用 EFSDP 的模块
      - model.language_model.layers.{*}.mlp.experts
```

专家并行详细说明（dispatcher类型、自定义EP forward）见[专家并行](../features/expert_parallel.md)。

MoE示例启动方式与Dense一致：

```shell
cd examples/qwen3_5_moe
bash run.sh
```

## 后续步骤

- 查看更多支持模型：[模型支持列表](../models/supported_models.md)
- 接入自定义模型：[模型适配指南](../develop/model_adaptation.md)
- 训练性能分析：[性能数据采集](../tools/profiling.md)

# 模型支持列表

FSDPTurbo与模型结构解耦，理论上支持各类基于Transformer架构的HuggingFace Transformers模型，并通过通配符plan适配不同模型结构。根据模型架构特点分为稠密模型和MoE模型两大类别，详情请查看以下支持列表。

**表格字段说明**

| 字段 | 说明 |
| :--- | :--- |
| **模型** | 模型名称。 |
| **下载链接** | 模型权重下载地址，点击可直接访问Hugging Face模型仓库。 |
| **示例目录** | 模型在本项目中的训练示例路径，可用于快速定位和使用模型。 |
| **序列长度** | 训练示例配置的最大文本序列长度。 |
| **集群规模** | 训练示例使用的集群规模配置，格式为"节点数 × 卡数"。 |
| **关键特性** | 训练示例中启用的关键特性组合。 |

> [!NOTE]
>
> - 仓库支持**Ascend 950PR&950DT系列产品**、**Atlas A3训练系列产品**和**Atlas A2训练系列产品**，可通过[硬件信息查询](https://www.hiascend.com/cann/download)获取NPU产品信息。并且要求单NPU的片上内存为64GB及以上，[安装驱动固件](../training/install_guide.md)后可通过`npu-smi info`查看单NPU的片上内存容量。
> - 各训练示例的默认集群规模见上表"集群规模"列，在`examples/*/run.sh`中通过`NNODES`与`GPUS_PER_NODE`设置，其中`NNODES`指代机器数量，`GPUS_PER_NODE`指代每台机器上的卡数。若实际卡数不同，需相应调整`distributed`下的并行度参数。
> - 如无法顺利访问Hugging Face社区下载权重资源，推荐前往[ModelScope](https://www.modelscope.cn/models)下载权重，需关注待下载文件的正确性和安全性。

## 稠密模型

稠密模型（Dense Model）是传统的深度学习模型结构，其神经元之间的连接是密集的，每一层的大多数或所有神经元都与下一层的所有神经元相连。

| 模型 | 下载链接 | 示例目录 | 序列长度 | 集群规模 | 关键特性 |
|---|---|---|---|---|---|
| Qwen3.5-27B（多模态） | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | `examples/qwen3_5` | 2048 | 1x16 | FSDP2 + TP + Ulysses CP + 重计算 |

## MoE模型

MoE模型（Mixture of Experts Model）通过门控网络将输入路由到部分专家网络，在扩大参数规模的同时控制计算开销。

| 模型 | 下载链接 | 示例目录 | 序列长度 | 集群规模 | 关键特性 |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | `examples/qwen3_5_moe` | 2048 | 1x16 | Dense特性 + EP（Expert Parallelism）+ EFSDP |
| Qwen3 MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | `examples/qwen3_moe` | 4096 | 1x8 | EP + 自定义modeling |
| DeepSeek V4 | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | `examples/deepseekv4` | 1024 | 1x8 | EP + 自定义modeling |
| Kimi K3 | [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | `examples/kimi_k3` | 2048 | 1x16 | EP + 自定义modeling |

## 新模型适配

新模型的接入方式参见[模型适配指南](../develop/model_adaptation.md)。

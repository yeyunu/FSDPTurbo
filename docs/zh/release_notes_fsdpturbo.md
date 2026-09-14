# 版本说明

## 版本配套说明

### 产品版本信息

| 产品名称 | FSDPTurbo |
| --- | --- |
| 产品版本 | 0.1.0 |
| 版本类型 | 正式版本（首次商发） |
| 组件名称 | FSDPTurbo |
| 发布时间 | （待补充） |
| 维护周期 | （待补充） |

> [!NOTE]
>
> FSDPTurbo基于PyTorch FSDP2构建，**不依赖MindSpeed Core与Megatron**，仅需配套PyTorch生态软件（PyTorch、TorchNPU、Triton-Ascend、transformers）即可运行。

### 相关产品版本配套说明

**表1** FSDPTurbo软件版本配套表

| FSDPTurbo版本 | Python版本 | PyTorch版本 | torchvision版本 | TorchNPU版本 | Triton-Ascend版本 | transformers版本 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1.0 | 3.11 | 2.10.0 | 0.25.0 | 2.10.0 | 3.2.2 | 5.15.0 |

> [!NOTE]
>
> - 示例验证版本组合：`torch==2.10.0`、`torchvision==0.25.0`、`transformers==5.15.0`；NPU环境另含`torch_npu==2.10.0`。
> - torchvision为随FSDPTurbo自动安装的基础依赖，仅多模态（图文）训练需要，纯文本训练不使用；若环境中已安装PyTorch，安装torchvision时需选择与该PyTorch配套的版本（如`torch==2.10.0`对应`torchvision==0.25.0`），否则pip可能因依赖解析变更已安装的PyTorch版本。
> - TorchNPU与Triton-Ascend仅NPU环境需要；GPU环境无需上述两者，需安装与PyTorch配套的CUDA驱动。
> - NPU环境支持的产品：Ascend 950系列、Atlas A3训练系列、Atlas A2训练系列；GPU环境支持Ampere架构及以上且支持BF16的NVIDIA GPU。硬件与软件配套详情参见[软件安装](./training/install_guide.md)。

## 更新说明

### 新增特性

| 特性 | 描述 | 目的 |
| --- | --- | --- |
| 多维并行策略 | FSDP2全分片数据并行、张量并行、专家并行（eager/fused/mc2/domino dispatcher与EFSDP）、上下文并行（Ulysses与KV AllGather），支持自由组合 | 兼顾训练吞吐与集群扩展性 |
| 内存优化 | 激活重计算、微批次切分、分块损失计算、激活值交换、优化器状态交换（AdamWSwap）、CPU Offload | 降低显存占用，支持低显存大模型训练 |
| 量化训练 | MXFP8（E4M3/E5M2/HIF8）与分块量化、W8A16量化感知训练、低精度AllGather | 降低显存占用并提升训练速度 |
| 断点续训 | 基于PyTorch分布式Checkpoint（DCP）的权重与优化器状态保存/恢复 | 支持故障恢复与长周期训练 |
| HuggingFace原生兼容 | `FSDPTurbo(config, model)`一行完成并行化改造，模型经Transformers原生接口加载 | 降低大模型分布式训练接入成本 |
| 双后端支持 | 昇腾NPU与GPU，算子统一注册、按设备自动分发 | 一套代码适配多种硬件 |

### 新增模型支持

| 特性 | 描述 | 目的 |
| --- | --- | --- |
| 新增模型支持 | 稠密模型：Qwen3.5-27B（多模态）；MoE模型：Qwen3.5-35B-A3B、Qwen3-30B-A3B、DeepSeek V4、Kimi K3 | 覆盖主流稠密与MoE模型，开箱即用 |

### 删除特性

无。首次发布。

### 接口变更说明

无。

## 已解决问题

无。

## 遗留问题

无。

## 升级影响

首次商发版本，不涉及从历史版本升级的场景。

## 配套文档

| 文档名称 | 内容简介 |
| --- | --- |
| [FSDPTurbo软件安装](./training/install_guide.md) | 硬件与操作系统配套说明、软件依赖版本要求与源码安装流程 |
| [FSDPTurbo快速入门](./training/quick_start_fsdpturbo.md) | 从环境准备到启动第一个训练任务的完整上手流程 |
| [特性列表](./features/supported_features.md) | 支持的训练特性汇总与各特性文档入口 |
| [模型支持列表](./models/supported_models.md) | 支持的模型清单与训练示例说明 |
| [常见问题FAQ](./FAQ.md) | 安装、训练、并行策略与断点续训等常见问题及解决方法 |

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

（待补充）

### 漏洞修补列表

无。

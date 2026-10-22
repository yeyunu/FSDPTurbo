# 版本说明

## 关键特性

FSDPTurbo支持FSDP、TP、EP、EFSDP、CP多维并行自由组合，兼顾训练吞吐与集群扩展性，轻松支撑长序列训练；同时提供激活重计算、微批次切分、激活值与优化器状态交换等内存优化手段，以及MXFP8、W8A16低比特量化加速，用更少显存、更低成本训练更大模型。

## 版本配套说明

### 产品版本信息

<table>
  <tbody>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>产品名称</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>MindSpeed</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>产品版本</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>26.2.0</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>版本类型</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>Release版本</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>组件名称</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>FSDPTurbo</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>发布时间</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>2026年10月</p></td>
    </tr>
    <tr>
      <th class="firstcol" valign="top" width="26.25%"><p>维护周期</p></th>
      <td class="cellrowborder" valign="top" width="73.75%"><p>6个月</p></td>
    </tr>
  </tbody>
</table>

### 相关产品版本配套说明

**表1** FSDPTurbo软件版本配套表

| FSDPTurbo版本 | Python版本 | PyTorch版本 | torchvision版本 | TorchNPU版本 |  CANN版本 |Triton-Ascend版本 | transformers版本 |
| --- | --- | --- | --- | --- | --- |--- | --- |
| main（在研版本） | 3.12 | 2.10.0 | 0.25.0 | 在研版本 | 在研版本 | 在研版本 | 5.15.0 |
| 26.2.0 | 3.12 | 2.10.0 | 0.25.0 | 26.2.0 | 9.2.0 | 3.2.2 | 5.15.0 |

> [!NOTE]
>
> - FSDPTurbo基于PyTorch FSDP2构建，**不依赖MindSpeed Core与Megatron**，仅需配套PyTorch生态软件（CANN、PyTorch、TorchNPU、Triton-Ascend、transformers）即可运行。
> - torchvision为随FSDPTurbo自动安装的基础依赖，仅多模态（图文）训练需要，纯文本训练不使用；若环境中已安装PyTorch，安装torchvision时需选择与该PyTorch配套的版本（如`torch==2.10.0`对应`torchvision==0.25.0`），否则pip可能因依赖解析变更已安装的PyTorch版本。
> - 支持的产品：<term>Ascend 950PR&950DT系列产品</term>、<term>Atlas A3训练系列产品</term>、<term>Atlas A2训练系列产品</term>。硬件与软件配套详情参见[软件安装](./training/install_guide.md)。
> - Triton-Ascend版本与CANN版本强绑定，Triton-Ascend的使用应该与CANN版本一一对应，详见[Triton-Ascend兼容性](https://triton-ascend.readthedocs.io/zh-cn/latest/release_note.html#id13)。

## 版本兼容性说明

> [!NOTE]
>
> 本节表格中“/”表示不可配套（未验证或不支持），“Y”表示可配套。

**表2** FSDPTurbo与TorchNPU版本兼容

<table style="table-layout: fixed; width: 750px; text-align:center">
  <colgroup>
    <col style="width: 250px">
    <col style="width: 250px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">FSDPTurbo</th>
      <th colspan="1">TorchNPU版本</th>
    </tr>
    <tr>
      <th>26.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.2.0</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

**表3** FSDPTurbo与CANN版本兼容

<table style="table-layout: fixed; width: 750px; text-align:center">
  <colgroup>
    <col style="width: 250px">
    <col style="width: 250px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">FSDPTurbo</th>
      <th colspan="1">CANN版本</th>
    </tr>
    <tr>
      <th>9.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.2.0</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

## 更新说明

### 新增特性

| 特性 | 描述 | 目的 |
| --- | --- | --- |
| 多维并行策略 | FSDP2全分片数据并行、张量并行、专家并行（eager/fused/mc2/domino dispatcher与EFSDP）、上下文并行（Ulysses与KV AllGather），支持自由组合 | 兼顾训练吞吐与集群扩展性 |
| 内存优化 | 激活重计算、微批次切分、分块损失计算、激活值交换、优化器状态交换（AdamWSwap）、CPU Offload | 降低显存占用，支持低显存大模型训练 |
| 量化训练 | MXFP8（E4M3/E5M2/HIF8）与分块量化、W8A16量化感知训练、低精度AllGather | 降低显存占用并提升训练速度 |
| 断点续训 | 基于PyTorch分布式Checkpoint（DCP）的权重与优化器状态保存/恢复 | 支持故障恢复与长周期训练 |
| HuggingFace原生兼容 | `FSDPTurbo(config, model)`一行完成并行化改造，模型经Transformers原生接口加载 | 降低大模型分布式训练接入成本 |
| 模型支持 | 新增支持稠密模型Qwen3.5-27B（多模态），MoE模型包括Qwen3.5-35B-A3B、Qwen3-30B-A3B、DeepSeek V4、Kimi K3 | 覆盖主流稠密与MoE模型，开箱即用本 |

### 删除特性

无

### 接口变更说明

无

## 已解决问题

无

## 遗留问题

无

## 升级影响

首次商发版本，不涉及从历史版本升级的场景。

## 配套文档

| 文档名称 | 内容简介 |
| --- | --- |
| [FSDPTurbo软件安装](./training/install_guide.md) | 硬件与操作系统配套说明、软件依赖版本要求与源码安装流程 |
| [FSDPTurbo快速入门](./training/quick_start_fsdpturbo.md) | 从环境准备到启动第一个训练任务的完整上手流程 |
| [特性列表](./features/supported_features.md) | 支持的训练特性汇总与各特性文档入口 |

## 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-10-07 08:00:00.0|2026-10-08|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-10-08 10:11:00|2026-10-08|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.101645|2026-10-08|无病毒，无恶意|

## 漏洞修补列表

无

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| --- | --- | --- |
| 01 | 2026-10-15 | 第一次正式发布。 |

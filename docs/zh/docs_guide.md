# FSDPTurbo 文档导航

## 文档介绍

FSDPTurbo 文档按内容类型组织，位于`docs/zh/`，主要包含以下核心目录：

- `training/`：训练指南，包含安装、快速入门、预训练、微调与性能调优
- `features/`：特性说明，包含并行策略、内存优化、量化与断点续训
- `develop/`：开发指南，包含模型适配与特性开发
- `models/`：模型支持列表
- `tools/`：开发工具，包含性能数据采集

## 文档目录结构

FSDPTurbo 文档目录层级介绍如下：

```text
docs/zh/
├── introduction.md           # 产品简介
├── project_guide.md          # 项目导读
├── docs_guide.md             # 文档导航
├── appendixes.md             # 附录（术语表等）
├── release_notes_fsdpturbo.md # 版本发布说明
├── SECURITYNOTE.md           # 安全声明
├── FAQ.md                    # 常见问题
├── develop/                  # 开发指南
│   ├── model_adaptation.md   # 模型适配指南
│   └── feature_development.md # 特性开发指南
├── features/                 # 特性说明
│   ├── supported_features.md # 特性列表
│   ├── fully_shard_parallel.md # FSDP2全分片数据并行
│   ├── tensor_parallel.md    # 张量并行
│   ├── expert_parallel.md    # 专家并行
│   ├── context_parallel.md   # 上下文并行
│   ├── quantization.md       # 量化训练
│   ├── recompute.md          # 激活重计算
│   ├── chunk_batch.md        # 微批次切分
│   ├── chunk_loss.md         # 分块损失计算
│   ├── swap_activation.md    # 激活值交换
│   ├── swap_optimizer.md     # 优化器状态交换
│   ├── checkpoint_resume.md  # 断点续训
│   └── environment_variable.md # 环境变量说明
├── models/                   # 模型支持列表
│   └── supported_models.md
├── tools/                    # 开发工具
│   └── profiling.md          # 性能数据采集
└── training/                 # 训练指南
    ├── install_guide.md      # 软件安装
    ├── quick_start_fsdpturbo.md # 快速入门
    ├── user_guide.md         # 模型使用导读
    ├── pretrain/             # 预训练
    │   └── pretrain.md       # 大模型分布式训练
    ├── finetune/             # 微调
    │   └── finetune.md       # 模型微调
    └── tuning/               # 性能调优
        └── performance_tuning.md # 模型性能调优
```

## 核心文档导航

快速跳转：[入门指南](#入门指南) | [训练方案](#训练方案) | [特性说明](#特性说明) | [开发指南](#开发指南) | [工具与调优](#工具与调优) | [其他](#其他)

### 入门指南

| 内容 | 说明 |
| --- | --- |
| [FSDPTurbo简介](./introduction.md) | 产品定位与核心能力介绍 |
| [软件安装](./training/install_guide.md) | 环境依赖说明与安装指导 |
| [快速入门](./training/quick_start_fsdpturbo.md) | 从环境安装到启动训练的上手指导 |
| [模型支持列表](./models/supported_models.md) | 仓库支持的模型清单 |

### 训练方案

| 分类 | 内容 | 说明 |
| --- | --- | --- |
| 导读 | [模型使用导读](./training/user_guide.md) | 按训练需求选择方案与配置的导读 |
| 预训练 | [大模型分布式训练](./training/pretrain/pretrain.md) | 分布式预训练方案 |
| 微调 | [模型微调](./training/finetune/finetune.md) | 模型微调方案 |

### 特性说明

| 分类 | 内容 | 说明 |
| --- | --- | --- |
| 总览 | [特性列表](./features/supported_features.md) | 仓库支持的特性一览 |
| 并行策略 | [FSDP2全分片数据并行](./features/fully_shard_parallel.md) | FSDP2参数与梯度分片 |
| 并行策略 | [张量并行](./features/tensor_parallel.md) | 层内张量并行 |
| 并行策略 | [专家并行](./features/expert_parallel.md) | MoE专家并行 |
| 并行策略 | [上下文并行](./features/context_parallel.md) | 长序列上下文并行 |
| 内存与计算优化 | [激活重计算](./features/recompute.md) | 丢弃并重算激活以节省显存 |
| 内存与计算优化 | [微批次切分](./features/chunk_batch.md) | 微批次进一步切分 |
| 内存与计算优化 | [分块损失计算](./features/chunk_loss.md) | 损失分块计算降低显存峰值 |
| 内存与计算优化 | [激活值交换](./features/swap_activation.md) | 激活值与NPU内存交换 |
| 内存与计算优化 | [优化器状态交换](./features/swap_optimizer.md) | 优化器状态与Host内存交换 |
| 其他 | [量化训练](./features/quantization.md) | 量化训练特性 |
| 其他 | [断点续训](./features/checkpoint_resume.md) | checkpoint保存与恢复训练 |
| 其他 | [环境变量说明](./features/environment_variable.md) | 集群与性能调优环境变量 |

### 开发指南

| 内容 | 说明 |
| --- | --- |
| [模型适配指南](./develop/model_adaptation.md) | 新模型接入FSDPTurbo的适配方法 |
| [特性开发指南](./develop/feature_development.md) | 新特性开发方法 |

### 工具与调优

| 内容 | 说明 |
| --- | --- |
| [性能数据采集](./tools/profiling.md) | 基于昇腾芯片采集profiling数据 |
| [模型性能调优](./training/tuning/performance_tuning.md) | 性能与显存调优指南 |

### 其他

| 内容 | 说明 |
| --- | --- |
| [FAQ](./FAQ.md) | 常见问题与解决方法 |
| [附录](./appendixes.md) | 术语表等 |
| [项目导读](./project_guide.md) | 项目导读 |
| [版本发布说明](./release_notes_fsdpturbo.md) | 版本发布说明 |
| [安全声明](./SECURITYNOTE.md) | 安全使用声明 |

## 文档撰写约定

面向文档贡献者的核心约定：

- 参数名、默认值、取值范围与生效条件以代码为准（配置定义见`fsdp_turbo/fsdp_turbo_config.py`），代码未实现的能力不写入文档。
- 文档间使用相对路径链接；图片存放于`docs/zh/figures/<主题>/`下并以相对路径引用。
- 代码块按内容标注语言：命令`bash`、配置`yaml`、示例`python`；配置示例与`examples/*/config.yaml`保持一致。
- 术语统一参照[附录](./appendixes.md)术语表，引入新术语时同步更新。
- 补充说明使用`> [!NOTE]`，安全与风险警示使用`> [!NOTICE]`。
- 新增文档后需同步评估`_menu.md`导航与本页目录结构。

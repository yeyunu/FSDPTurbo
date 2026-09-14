# FSDPTurbo 项目导读

欢迎向FSDPTurbo贡献代码与文档！

## 项目介绍

FSDPTurbo项目代码按照模块化设计原则进行组织，主要包含以下核心模块：

- `fsdp_turbo/`：核心代码目录，包含分布式并行策略、显存优化、量化训练、训练框架等核心功能实现
- `examples/`：模型示例脚本，涵盖Qwen、Kimi、DeepSeek等模型的训练示例
- `tests/`：测试用例目录，包含单元测试与系统测试
- `docs/`：项目文档目录，提供安装指南、快速入门、特性说明与训练方案等
- `ci/`：CI构建与测试脚本
- `pre-commit/`：代码质量钩子配置，包含拼写检查与密钥扫描

## 代码目录结构

上方列出的各模块在项目仓库中以目录和文件的形式组织。以下为项目目录的概览结构：

```text
FSDPTurbo/
 ├── ci/                        # CI构建与测试脚本
 ├── docs/                      # 项目文档目录
 │   └── zh/                    # 中文文档
 ├── examples/                  # 模型示例脚本
 ├── fsdp_turbo/                # 核心代码目录
 ├── pre-commit/                # pre-commit钩子配置
 ├── tests/                     # 测试用例目录
 ├── setup.py                   # 安装配置文件
 ├── pyproject.toml             # 项目配置文件
 ├── README.md                  # 项目说明文档
 └── LICENSE                    # 许可证文件
```

其中`fsdp_turbo/`是最核心的代码目录，其内部结构如下：

```text
fsdp_turbo/
 ├── distributed/               # 分布式并行策略
 │   ├── context_parallel/      # 上下文并行
 │   │   ├── kv_allgather/      # KV AllGather上下文并行
 │   │   └── ulysses/           # Ulysses上下文并行
 │   ├── expert_parallel/       # 专家并行（dispatcher与EFSDP）
 │   ├── fine_grained_fully_shard/ # 细粒度全分片
 │   ├── fully_shard_parallel/  # FSDP2全分片数据并行
 │   ├── tensor_parallel/       # 张量并行
 │   ├── parallel_state.py      # DeviceMesh与并行度管理
 │   └── dist_ops.py            # 分布式通信操作
 ├── memory/                    # 显存优化
 │   ├── chunk_batch/           # 微批次切分
 │   ├── chunk_loss/            # 分块损失计算
 │   ├── recompute/             # 激活重计算
 │   ├── swap_activation/       # 激活值交换
 │   └── swap_optimizer/        # 优化器状态交换
 ├── models/                    # 内置模型实现
 │   ├── kimi/                  # Kimi K3模型
 │   └── qwen/                  # Qwen3.5系列模型
 ├── modules/                   # 通用模型组件
 ├── ops/                       # 多后端算子
 │   ├── npu/                   # NPU后端实现
 │   ├── cuda/                  # CUDA后端实现
 │   ├── cpu/                   # CPU实现
 │   └── triton/                # Triton kernel实现
 ├── quantization/              # 量化训练
 │   ├── converter/             # 模块量化转换器
 │   ├── core/                  # 量化核心流程
 │   ├── mx_formats/            # MX格式算子
 │   └── qat/                   # 量化感知训练
 ├── training/                  # 训练框架
 │   ├── factories/             # 模型/优化器/调度器/分词器工厂
 │   ├── checkpoint.py          # checkpoint保存与恢复
 │   ├── dataloader.py          # 数据加载
 │   ├── trainer.py             # 训练主循环
 │   ├── monitor.py             # 训练监控
 │   └── profiler.py            # 性能数据采集
 ├── utils/                     # 通用工具
 ├── fsdp_turbo.py              # 训练入口
 └── fsdp_turbo_config.py       # 全量训练配置定义
```

## 核心子模块说明

上方的目录树展示了项目的整体结构，其中`fsdp_turbo/`下各子模块各司其职：

- `distributed/`：并行策略实现。`parallel_state.py`管理三维DeviceMesh与并行度自动推导；`fully_shard_parallel/`提供FSDP2全分片数据并行，`tensor_parallel/`提供张量并行，`expert_parallel/`提供专家并行（含dispatcher通信与EFSDP专家分片），`context_parallel/`提供KV AllGather与Ulysses两种上下文并行，`fine_grained_fully_shard/`支持参数细粒度分片。
- `memory/`：显存优化手段，包含激活重计算、激活值交换、优化器状态交换、微批次切分与分块损失计算。
- `models/`：内置模型实现（Qwen3.5、Qwen3.5-MoE、Kimi K3等）。
- `modules/`：通用模型组件（RMSNorm、KDA、GatedDeltaNet、HyperConnection、短卷积等）。
- `ops/`：按设备后端组织的算子实现（`npu/`、`cuda/`、`cpu/`、`triton/`），经`registry.py`统一注册，按运行设备自动分发。
- `quantization/`：量化训练，包含MXFP8格式算子、W8A16量化感知训练与模块量化转换器。
- `training/`：训练生命周期管理，包含训练主循环、checkpoint保存恢复、数据加载、训练监控、性能数据采集，以及模型/优化器/调度器/分词器工厂。
- `utils/`：设备、精度、日志、patch等通用工具。

`fsdp_turbo.py`为训练入口，`fsdp_turbo_config.py`定义全量训练配置。

## 模型运行样例

`examples/`目录提供了主流模型的完整训练示例，每个示例包含`config.yaml`（训练配置）、`run.sh`（torchrun启动脚本）与`train.py`（训练入口脚本）：

```text
examples/
├── deepseekv4/     # DeepSeek V4示例
├── kimi_k3/        # Kimi K3示例
├── qwen3_5/        # Qwen3.5示例
├── qwen3_5_moe/    # Qwen3.5-MoE示例
└── qwen3_moe/      # Qwen3-MoE示例
```

进入对应模型目录启动训练：

```bash
# 单机16卡
NNODES=1 GPUS_PER_NODE=16 bash run.sh
# 多机需额外指定主节点IP与端口
NNODES=2 GPUS_PER_NODE=16 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 bash run.sh
```

完整使用流程见[快速入门](./training/quick_start_fsdpturbo.md)，支持的模型清单见[模型支持列表](./models/supported_models.md)。

## 总结

FSDPTurbo提供了一套兼具高性能与易用性的大模型分布式训练解决方案，具有以下核心特点：

- **多维并行**：支持FSDP/TP/EP/EFSDP/CP多维并行自由组合，基于通配符模式选择模块，与具体模型结构解耦
- **易接入**：基于PyTorch FSDP2构建，直接对接HuggingFace Transformers生态，`FSDPTurbo(config, model)`一行完成并行化改造
- **显存优化**：提供激活重计算、分块损失、激活值交换、优化器状态交换等手段，支持低显存大模型训练
- **量化加速**：支持MXFP8量化格式、分块量化与量化感知训练（QAT），兼容专家并行路径
- **多后端支持**：面向昇腾NPU与GPU，算子按设备统一注册、自动分发
- **开箱即用**：提供Qwen、Kimi、DeepSeek等主流模型的训练示例脚本与配置

项目通过清晰的目录结构、完善的文档体系与丰富的示例脚本，帮助开发者高效地进行大模型分布式训练工作。

## 贡献流程

1. Fork本仓库
2. 创建特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交更改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 提交Pull Request

## 问题反馈

- 通过[Issues](https://gitcode.com/Ascend/FSDPTurbo/issues)提交功能问题或合规问题。
- 若文档中引用了您的模型或数据集且不希望被提及，请提交issue说明，我们将按您的要求删除或更新相关描述。

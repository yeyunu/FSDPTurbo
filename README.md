# FSDPTurbo

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**兼具高性能与易用性的分布式训练加速库**

</div>

FSDPTurbo 是一个兼具高性能与易用性的分布式训练加速库，基于高内聚低耦合的架构设计理念，打造一套与模型结构解耦的通用加速方案，用于训练大规模深度学习模型。

FSDPTurbo 基于 PyTorch FSDP2（DTensor 全分片数据并行）构建，直接对接 HuggingFace Transformers 生态：用户使用原生 `AutoModel` 加载模型后，通过 `FSDPTurbo(config, model)` 即可完成并行化改造，显著降低大模型分布式训练的接入成本。

FSDPTurbo 面向昇腾 NPU，额外提供：

- **多维并行策略**：FSDP2 全分片数据并行、张量并行（TP）、专家并行（EP）、上下文并行（CP）自由组合
- **内存优化**：激活重计算、微批次切分、分块损失计算、激活值/优化器状态交换与 CPU Offload
- **量化加速**：MXFP8（E4M3/E5M2/HIF8）分块量化、W8A16 量化感知训练、低精度 AllGather
- **开箱即用的模型支持**：预置 Qwen3.5、Qwen3 MoE、DeepSeek V4、Kimi K3 等系列模型的训练适配与训练脚本

详细的框架介绍参见[项目简介](docs/zh/introduction.md)。

# 最新消息

---

- 🔥 **2026**：FSDPTurbo 仓库正式发布首个版本 0.1.0，提供多维并行、内存优化与量化训练能力，预置 Qwen3.5、Kimi-K3、DeepSeek-V4 等模型训练示例

# 目录结构

---

FSDPTurbo 项目代码按照模块化设计原则进行组织。

```shell
FSDPTurbo/
 ├── .gitcode                  # Gitcode 平台配置（PR 模板等）
 ├── ci                        # CI 门禁与 Docker 镜像构建
 ├── docs                      # 项目文档目录
 │   └── zh                    # 中文文档
 ├── examples                  # 训练示例（按模型组织）
 ├── fsdp_turbo                # 核心代码目录
 │   ├── distributed           # 分布式并行（FSDP/EFSDP/TP/EP/CP/parallel_state）
 │   ├── memory                # 内存优化（重计算/微批次切分/分块损失/交换）
 │   ├── quantization          # 量化（MXFP8/W8A16/模型转换框架）
 │   ├── modules               # 自定义模块（GDN/KDA/短卷积等）
 │   ├── ops                   # 后端算子（npu/cpu/triton/cuda 统一注册分发）
 │   ├── training              # 训练辅助（trainer/checkpoint/factories）
 │   └── utils                 # 工具函数
 ├── pre-commit                # pre-commit 钩子配置
 ├── tests                     # 测试用例目录（ST/UT）
 ├── pyproject.toml            # 构建配置
 ├── setup.py                  # 安装配置文件
 └── README.md                 # 项目说明文档
```

# 文档导航

---

FSDPTurbo 文档目录包含以下核心内容：

- **项目简介**：[简介](docs/zh/introduction.md) —— 框架定位、架构与功能特性
- **环境安装指导**：[软件安装](docs/zh/training/install_guide.md) —— 硬件配套、软件版本配套与安装配置说明
- **快速入门**：[快速入门](docs/zh/training/quick_start_fsdpturbo.md) —— 从环境安装到训练拉起的入门指导
- **用户指南**：[模型使用导读](docs/zh/training/user_guide.md) —— 预训练与微调训练方案
- **特性说明**：[特性列表](docs/zh/features/supported_features.md) —— 全量训练特性及文档链接
- **性能调优**：[性能调优](docs/zh/training/tuning/performance_tuning.md)
- **开发指南**：[特性开发指南](docs/zh/develop/feature_development.md)、[模型适配指南](docs/zh/develop/model_adaptation.md)
- **测试说明**：[测试用例贡献说明](tests/README.md)
- **Docker 镜像**：[FSDPTurbo Runtime Docker 镜像概述](ci/OVERVIEW.zh.md)

# 版本说明

---

查看[版本说明](./docs/zh/release_notes_fsdpturbo.md)，了解 FSDPTurbo 的最新发布版本以及历史版本，包含对应版本的软件配套表、版本兼容性说明，以及当前版本的更新说明。

# 安装

---

- 详细的安装步骤和环境配置请参考 [FSDPTurbo 软件安装](docs/zh/training/install_guide.md)。

从源码安装：

```bash
git clone https://gitcode.com/Ascend/FSDPTurbo.git
cd FSDPTurbo
pip install -e .
```

# 快速上手

---

指导开发者快速启动大语言模型/多模态模型的预训练及微调任务，具体的操作请参考：

- [快速入门（以 Qwen3.5-27B 为例）](./docs/zh/training/quick_start_fsdpturbo.md)

FSDPTurbo 已预置以下训练示例脚本（位于 `examples/`）：

| 模型 | 示例目录 | 关键特性 |
| --- | --- | --- |
| Qwen3.5-27B（多模态） | [examples/qwen3_5](examples/qwen3_5) | FSDP2 + TP + Ulysses CP + 重计算 |
| Qwen3.5-35B-A3B | [examples/qwen3_5_moe](examples/qwen3_5_moe) | Dense 特性 + EP + EFSDP |
| Qwen3 MoE | [examples/qwen3_moe](examples/qwen3_moe) | EP + 自定义 modeling |
| DeepSeek V4 | [examples/deepseekv4](examples/deepseekv4) | EP + 自定义 modeling |
| Kimi K3 | [examples/kimi_k3](examples/kimi_k3) | EP + 自定义 modeling |

```python
import torch
from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, FSDPPlanConfig, TPPlanConfig
from fsdp_turbo.fsdp_turbo import FSDPTurbo

# 使用原生方式创建/加载模型
model = create_your_model()

# 配置并行策略与内存优化
config = FSDPTurboConfig(
    fully_shard_parallel_size=8,
    fsdp_plan=FSDPPlanConfig(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    ),
    tensor_parallel_size=4,
    tp_plan=TPPlanConfig(
        colwise_parallel=['*.q_proj', '*.k_proj', '*.v_proj'],
        rowwise_parallel=['*.o_proj'],
    ),
    recompute=True,
    recompute_plan=['model.layers.*'],
)

# 一行完成并行化改造
model = FSDPTurbo(config, model)

# 训练循环
for batch in dataloader:
    output = model(batch)
    loss = compute_loss(output, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

专家并行、量化等更多配置参见[模型使用导读](./docs/zh/training/user_guide.md)。

实际训练任务按「训练入口 `train.py` + 训练配置 `config.yaml` + 启动脚本 `run.sh`」三件套组织，数据在启动时自动加载与 Tokenize，无需离线预处理（更换数据集需参考[模型适配指南](./docs/zh/develop/model_adaptation.md)注册自定义数据集）；部分模型（如 Kimi K3、DeepSeek V4）需配套自定义 modeling，完整示例参见 `examples/` 下各模型目录。

# 支持模型

---

FSDPTurbo 与模型结构解耦，理论上支持各类基于 Transformer 架构的 HuggingFace Transformers 模型，并通过通配符 plan 适配不同模型结构，涵盖稠密、MoE 与多模态架构。支持的模型清单可查看[模型支持列表](./docs/zh/models/supported_models.md)。

# 训练方案与特性

---

FSDPTurbo 提供大语言模型/多模态模型的预训练与微调训练方案，支持多维并行策略自由组合，并预置面向昇腾硬件的内存优化与量化加速特性：

- **全分片数据并行（FSDP2）**：实现 PyTorch 的 FSDP2 API，支持混合精度训练、前向/反向预取和选择性模块分片
- **张量并行（TP）**：支持列并行、行并行和序列并行，集成 DTensor
- **专家并行（EP）**：专为混合专家（MoE）模型设计，支持多种调度器类型（`eager`、`fused`、`mc2`、`domino`）
- **上下文并行（CP）**：Ulysses 与 KV AllGather 风格，支持长序列训练
- **内存优化**：梯度检查点、微批次切分、分块损失计算、激活值交换、优化器状态交换
- **量化训练**：MXFP8 量化（E4M3、E5M2、HIF8）、分块量化、W8A16 量化感知训练、低精度 AllGather
- **断点续训**：基于 PyTorch 分布式 Checkpoint（DCP）的权重与优化器状态保存/恢复

完整的特性列表及详细说明参见[特性列表](./docs/zh/features/supported_features.md)。

# 测试

---

FSDPTurbo 提供单测（UT）与端到端系统测试（ST，含基线比对）双测试体系，用于看护多维并行（FSDP/TP/EP/CP）、显存优化与量化加速等核心能力，具体参见[测试用例贡献说明](tests/README.md)。

# 版本维护策略

---

FSDPTurbo版本有以下五个维护阶段：

| **状态**            | **时间**  | **说明**                                                     |
| ------------------- | --------- | ------------------------------------------------------------ |
| 计划                | 1—3 个月  | 计划特性                                                     |
| 开发                | 3 个月    | 开发特性                                                     |
| 维护                | 6-12 个月 | 合入所有已解决的问题并发布版本，针对不同的FSDPTurbo版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月  | 合入所有已解决的问题，无专职维护人员，无版本发布             |
| 生命周期终止（EOL） | N/A       | 分支不再接受任何修改                                         |

# FAQ

---

仓库基本使用过程中的常见问题可以参考 [FSDPTurbo FAQ](./docs/zh/FAQ.md)。FAQ 中未能涵盖的问题，可以在仓库的 [issues 列表](https://gitcode.com/Ascend/FSDPTurbo/issues)中尝试寻找类似问题，或者提交新的 issue。

# 安全声明

---

FSDPTurbo 仅用于研究与开发目的。使用过程中请遵守相关法律法规，并遵守所使用模型与数据集的 License 要求。系统加固、数据安全、运行安全、通信安全及漏洞反馈等说明请参见 [FSDPTurbo 安全声明](./docs/zh/SECURITYNOTE.md)。

# 免责声明

---

## 致 FSDPTurbo 使用者

1. FSDPTurbo 提供的所有内容仅供您用于非商业目的。
2. 对于 FSDPTurbo 测试用例以及示例文件中所涉及的各模型和数据集，平台仅用于功能测试，华为不提供任何模型权重和数据集，如您使用这些数据进行训练，请您特别注意应遵守对应模型和数据集的 License，如您因使用这些模型和数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用 FSDPTurbo 过程中，发现任何问题（包括但不限于功能问题、合规问题），请在 Gitcode 提交 issue，我们将及时审视并解决。
4. FSDPTurbo 功能依赖的第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，FSDPTurbo 仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。

## 致数据所有者

如果您不希望您的模型或数据集在 FSDPTurbo 中被提及，或希望更新 FSDPTurbo 中有关的描述，请在 Gitcode 提交 issue，我们将根据您的 issue 要求删除或更新您相关描述。衷心感谢您对 FSDPTurbo 的理解和贡献。

# License 声明

---

- FSDPTurbo 产品的使用许可证，具体请参见 [LICENSE](LICENSE)。
- FSDPTurbo 中涉及的文件，如文件目录下存在 License 的，以该 License 为准；如文件目录下不存在 License 的，以 Apache 2.0 许可证许可。

# 贡献声明

---

如果您希望向 FSDPTurbo 报告问题和贡献代码，具体请参见[贡献指南](./CONTRIBUTING.md)。

# 致谢

---

感谢来自社区的每一个 PR，欢迎贡献 FSDPTurbo。

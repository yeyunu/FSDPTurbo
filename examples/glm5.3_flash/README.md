# GLM-5.3-Flash 使用指南

## 目录

- [GLM-5.3-Flash 使用指南](#glm-53-flash-使用指南)
  - [版本说明](#版本说明)
    - [参考实现](#参考实现)
    - [变更记录](#变更记录)
  - [环境安装](#环境安装)
    - [1. 环境准备](#1-环境准备)
    - [2. 环境搭建](#2-环境搭建)
    - [3. 安装配套版本的 Triton Ascend](#3-安装配套版本的-triton-ascend)
    - [4. 安装 fla-npu](#4-安装-fla-npu)
  - [模型检查点准备](#模型检查点准备)
  - [数据集准备及处理](#数据集准备及处理)
  - [训练](#训练)
    - [1. 配置参数](#1-配置参数)
    - [2. 启动训练](#2-启动训练)
  - [长序列与并行配置建议](#长序列与并行配置建议)
  - [环境变量声明](#环境变量声明)
  - [注意事项](#注意事项)

## 版本说明

> 当前版本仅验证减层、减专家的受限场景训练，使用前请先阅读[模型检查点准备](#模型检查点准备)章节。更多能力正在支持，敬请期待！

当前目录提供 GLM-5.3-Flash 在 FSDPTurbo 训练流程中的示例配置。模型实现位于 `fsdp_turbo/models/glm5_next`，训练配置和启动脚本位于 `examples/glm5.3_flash`。

### 参考实现

```shell
url=https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/tree/main
```

### 变更记录

2026.10.25: 首次支持 GLM-5.3-Flash 模型

---

## 环境安装

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](../../docs/zh/training/install_guide.md)，完成昇腾软件安装。

推荐环境版本如下：

| 组件 | 推荐版本 |
|------|----------|
| Python | 3.12 |
| torch | 2.10.0 |
| TorchNPU | 2.10.0.post4 及以上 |
| CANN | 9.2.0 及以上 |

### 2. 环境搭建

拉取 FSDPTurbo 代码仓，并进入代码仓根目录：

```bash
git clone https://gitcode.com/Ascend/FSDPTurbo.git
cd FSDPTurbo
pip3 install -e .
```

执行如下指令安装基础依赖：

```bash
# 安装transformer库
pip install transformers==5.17.0
pip install tiktoken==0.13.0
```

### 3. 安装配套版本的 Triton Ascend

GLM-5.3-Flash 的 KDA（Kimi Delta Attention）等线性注意力融合算子基于 Triton 实现，在昇腾环境下需要安装配套版本的 Triton-Ascend。
请参考《Triton-Ascend》中的"[通过pip安装Triton-Ascend](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#piptriton-ascend)"章节，获取配套版本的 Triton-Ascend 安装指令。

```shell
# 供参考, 命令以文档为准
pip install triton-ascend==3.2.2 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

KDA 算子实现依赖 `triton-ascend-kernels` 算子库（`modeling_glm5_next.py` 中的 `chunk_kda` 来自该包），安装步骤如下：

Atlas A3 训练产品安装步骤：

```shell
# 拉取 triton-ascend-kernels 代码仓
git clone https://gitcode.com/fengrui886/triton-ascend-kernels
cd triton-ascend-kernels
git checkout kda_a3

# 安装
pip install -e . --no-build-isolation --no-deps
cd ..
```

# 安装
pip install -e . --no-build-isolation --no-deps
cd ..
```

### 4. 安装 fla-npu

GLM-5.3-Flash 适配包含 AscendC KDA wrapper，需安装 fla-npu 以保证相关模块可正常导入。

拉取 flash-linear-attention-npu 代码仓，并进入代码仓根目录，切到对应 commit：

```bash
git clone https://github.com/flashserve/flash-linear-attention-npu
cd flash-linear-attention-npu
git checkout 2062460754c8
```

安装步骤可参考 fla-npu 仓 README：[flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu/blob/main/README.md)，参考"源码一键编译并生成 wheel"完整 fla 编译。

```bash
# fla需安装gawk
python -m pip install -r requirements.txt
# 检查编译环境满足要求, 如果不满足需解决
python scripts/check_npu_env.py
# FLA_NPU_SOC对应A2/A3/A5分别填ascend910b/ascend910_93/ascend950
FLA_NPU_BUILD_ARGS="-j256" FLA_NPU_SOC=ascend910_93 python scripts/build_wheel.py
# 编译成功后安装
pip install --force-reinstall --no-cache-dir --no-deps ./dist/*.whl
cd ..
```

安装后检验 fla-npu 是否安装成功：

```bash
pip list | grep fla
```


### 4. 安装 cann-ops-transformer
git clone https://atomgit.com/cann/ops-transformer.git && cd ops-transformer

bash build.sh --pkg --soc=ascend910_93 -j16  # 全部算子编译
./build_out/cann-ops-transformer-*linux*.run --install-path=/指定路径
source指定路径下的set_env.sh

---

## 模型检查点准备

FSDPTurbo 无法直接加载官方 GLM-5.3-Flash 原始权重（fp8 全量检查点约 155 GiB/rank，加载即 OOM），必须先使用随仓库提供的检查点转换工具完成**一次性的离线减层、减专家与 fp8 反量化**：

```bash
python examples/glm5.3_flash/convert_checkpoint.py \
    --src /path/to/GLM-5.3-Flash \
    --dst /path/to/glm53-flash-4l-64e \
    --num-hidden-layers 4 \
    --n-routed-experts 64
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--src` | - | 官方 GLM-5.3-Flash fp8 检查点目录 |
| `--dst` | - | 输出目录（自包含 bf16 HF 检查点） |
| `--num-hidden-layers` | **无** | 保留的语言模型层数，建议至少 4 层以上，需包含 KDA+DSA 结构 |
| `--n-routed-experts` | **无** | 保留的专家个数，需能被 `expert_parallel_size` 整除 |

`--num-hidden-layers` 与 `--n-routed-experts` **没有默认值**，必须通过命令行显式传入。

**说明**：

- 转换在单进程 CPU 上完成（全量模型载入主机内存并反量化），请确保主机空余内存充足（建议 800 GiB 以上），耗时较长属正常现象；
- 转换会自动完成 fp8→bf16 反量化（按 128x128 block scale）、层截断（同步裁剪 `layer_types` / `mlp_layer_types` / `indexer_types`）、专家裁剪，并清理 config.json 中的 `quantization_config`，tokenizer/processor 文件随同拷贝；
- Vision tower 与多模态融合结构保持不变；
- 训练侧无需再手工修改 `config.json` 的层数/专家数字段，也不要手工清除 `quantization_config`（会导致 fp8 张量被当作 bf16 垃圾数据加载）。

---

## 数据集准备及处理

- 使用**geo3k数据集**训练：参考[针对VL模型的数据构造 · 使用真实数据集](../../docs/zh/training/geo3k_finetune_guide.md)

## 训练

### 1. 配置参数

训练配置文件为 [config.yaml](./config.yaml)。

【模型路径配置】

将 `model_name_or_path`、`tokenizer_name_or_path` 指向[模型检查点准备](#模型检查点准备)生成的转换后检查点目录：

```yaml
model:
  model_name_or_path: "/path/to/glm53-flash-4l-64e"
  tokenizer_name_or_path: "/path/to/glm53-flash-4l-64e"
  torch_dtype: "bf16"
```

**说明**：代码仓中已包含适配 FSDPTurbo 的模型实现文件 `fsdp_turbo/models/glm5_next`，无需额外下载或替换。

【数据配置】

根据实际情况修改 `config.yaml` 中的数据路径：

```yaml
data:
  dataset_path: "/path/to/dataset.json"
  max_seq_length: 2048
  is_multimodal: true
```

【检查点保存配置】

```yaml
checkpoint:
  output_dir: "./output/glm-5.3-Flash"
  save_steps: 10000
  save_optim: true
  load_optim: true
  resume_from_checkpoint: null   # 断点续训时指向待恢复的检查点
```

【主要配置项说明】

| 配置项 | 配置路径 | 参数说明 | 取值 |
|--------|----------|----------|------|
| `expert_parallel_size` | `distributed` | EP专家并行度，值为1时不开启，仅对MoE模型生效 | 默认 `16` |
| `ulysses_parallel_size` | `distributed` | Ulysses CP 序列并行度 | 默认 `1` |
| `kda_implementation` | `model`（config.json） | KDA 实现选择 | `triton` / `eager` |
| `causal_conv1d_implementation` | `model`（config.json） | KDA 短卷积实现选择 | `triton` / `eager` |
| `dsa_implementation` | `model` | DSA 注意力实现选择 | `sfa` / `dense` |
| `indexer_implementation` | `model` | DSA indexer 实现选择 | `triton` / `eager` |
| `use_fused_lightning_indexer` | `model` | CANN 融合 lightning indexer 开关 | 默认 `false` |
| `npu_fused_attention` | `model` | NPU 融合注意力开关（MLA/DSA 文本注意力与视觉塔） | 默认 `1` |
| `recompute` | `memory` | 重计算开关，开启后节省显存占用 | 默认 `true` |
| `chunk_loss` | `features` | chunk loss 开关，需与 `chunk_loss_size` 配套 | 默认 `true` |

【单机运行配置】

启动脚本 [run.sh](./run.sh) 默认按单机 16 卡配置：

```shell
NNODES=1
GPUS_PER_NODE=16
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
NODE_RANK=0
```

如需改为其他卡数，需同步调整 `GPUS_PER_NODE` 与 `distributed.expert_parallel_size`（专家个数需能被 EP 并行度整除）。

【多机运行配置】

使用 [run_multinode.sh](./run_multinode.sh)，通过 `HOSTS` 列表（master 优先）静态分配 rank，所有节点执行相同命令即可：

```shell
MASTER_ADDR=10.0.0.1 HOSTS=10.0.0.1,10.0.0.2 bash run_multinode.sh
```

### 2. 启动训练

完成环境安装、检查点转换和数据路径配置后，执行：

```shell
cd FSDPTurbo
bash examples/glm5.3_flash/run.sh
```

训练日志默认写入 `examples/glm5.3_flash/log/train_glm53flash.log`。

---

## 长序列与并行配置建议

GLM-5.3-Flash 的 Ulysses CP 前向已内置于模型实现（`Glm5NextModel` / `Glm5NextTextAttention` / `Glm5NextTextLinearAttention`），开启 CP 只需设置 `distributed.ulysses_parallel_size > 1`，无需额外配置 patch。

不同序列长度下的推荐配置（`config.yaml` 的 `model` 段，chunk loss 在 `features` 段）：

**序列长度 <= 64K**：

```yaml
model:
  use_fused_lightning_indexer: true

features:
  chunk_loss: true
  chunk_loss_size: 512
```

**序列长度 >= 128K**（eager indexer 的 `[B, S, NH, P]` fp32 score 张量与 dense mask 均会 OOM，需切换 SFA 路径）：

```yaml
model:
  use_fused_lightning_indexer: false
  indexer_implementation: "triton"
  dsa_implementation: "sfa"

features:
  chunk_loss: true
  chunk_loss_size: 512
```

> 注意：`use_fused_lightning_indexer: true` 与 `dsa_implementation: "sfa"` **互斥**（CANN indexer 无法保证所选 pool 全部有效，会导致 SFA backward AICore 访存越界），二者不可同时开启。

**性能优化配置**：

- [重计算](../../docs/zh/features/recompute.md)：

```yaml
memory:
  recompute: true   # true 表示开启，false 表示关闭
  recompute_plan:
    # Vision tower (runs on the full sequence under Ulysses CP).
    - model.visual.blocks.{*}
    # LM decoder layers (run on the local CP shard; one unit per layer).
    - model.language_model.layers.{*}
```

- [chunk loss](../../docs/zh/features/chunk_loss.md)：

```yaml
features:
  chunk_loss: true          # true 表示开启，false 表示关闭
  chunk_loss_size: 512      # 计算loss时在seq维度切分成大小为chunk_loss_size的小块进行计算
```

---

## 环境变量声明

| 环境变量 | 描述 | 取值说明 |
|----------|------|----------|
| `ASCEND_LAUNCH_BLOCKING` | NPU 算子同步执行调试开关 | 默认 `0` |
| `TASK_QUEUE_ENABLE`      | 用于控制开启task_queue算子下发队列优化的等级 | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化 |
| `CPU_AFFINITY_CONF`      | 控制CPU端算子任务的处理器亲和性，即设定任务绑核 | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核 |
| `HCCL_CONNECT_TIMEOUT`   | 用于限制不同设备之间socket建链过程的超时等待时间 | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s` |
| `PYTORCH_NPU_ALLOC_CONF` | 控制缓存分配器行为 | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征 |
| `MULTI_STREAM_MEMORY_REUSE` | 配置多流内存复用是否开启 | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用 |
| `TRITON_ALWAYS_COMPILE`  | 控制Triton算子是否总是重新编译 | `0`: 命中编译缓存时不重复编译<br>`1`: 每次运行强制重新编译（一般用于算子调试） |

---

## 注意事项

1. 启动脚本中的 CANN 路径为示例环境路径，使用前需改为当前机器实际路径。
2. 默认配置使用 16 卡单机训练，运行前请检查 `npu-smi info`，避免占用他人任务。
3. 官方 fp8 全量检查点**不能直接训练**，必须先经 [模型检查点准备](#模型检查点准备) 章节的转换工具完成减层、减专家与反量化。
4. `dsa_implementation: "sfa"` 依赖 `torch_npu.npu_sparse_flash_attention`；如果当前 TorchNPU 不支持该接口，可先切换为 `dense` 做功能验证。
5. 多模态样本的图片 token 展开后实际序列长度可能超过 `max_seq_length`，配置长序列训练时请为多模态 token 预留足够长度预算。

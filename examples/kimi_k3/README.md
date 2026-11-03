# Kimi K3 训练示例

本目录提供 Kimi K3 多模态 MoE 模型的训练示例。默认的
[`config.yaml`](./config.yaml) 已开启 Ulysses Context Parallel（CP），CP
并行度为 2。

## 模型文件准备

从下载的 Kimi-K3 模型目录中拷贝`media_utils.py` 和
`configuration_kimi_k3.py` 到 FSDPTurbo 的`fsdp_turbo/models/kimi/` 目录。
以本地仓库为例，目标目录为`D:\FSDPTurbo\fsdp_turbo\models\kimi`这两个文件需要与
`modeling_kimi_k3.py`、`kimi_k3_processor.py` 同级，以满足相对导入。

在训练服务器上执行以下命令，并将路径替换为实际位置：

```bash
HF_PATH="/download/Kimi-K3"
TURBO_PATH="/home/workspace/FSDPTurbo"

cp "${HF_PATH}/media_utils.py" "${TURBO_PATH}/fsdp_turbo/models/kimi/"
cp "${HF_PATH}/configuration_kimi_k3.py" "${TURBO_PATH}/fsdp_turbo/models/kimi/"
```

请保留仓库内已适配的 `modeling_kimi_k3.py`、`modeling_kimi_linear.py`
等实现文件。模型权重、tokenizer、processor 配置等资源仍由
`config.yaml` 中的 `model_name_or_path` 指定。

## 模型代码加载

`train.py` 和系统测试先调用 `fsdp_turbo.models.kimi.register_kimi_k3()`，
将仓库内的 `KimiK3Config`、`KimiLinearConfig` 和 `KimiK3ForConditionalGeneration`
注册到 Transformers。随后使用 `trust_remote_code=False` 加载配置和模型，
确保使用 `fsdp_turbo/models/kimi/` 内的 `modeling_kimi_k3.py` 与
`modeling_kimi_linear.py`，包括本地算子和 MoE 适配。
`model_name_or_path` 仍指定配置及权重来源；processor 显式使用仓库内的
`KimiK3Processor`，保留对列表文本输入的适配。其 tokenizer 和图像处理器
继续通过 `trust_remote_code=True` 从模型目录加载。

此方式对应“导入本地模型包、注册类、加载权重”的流程，
通过 Transformers 注册表接入，无需修改模型目录中的 modeling 文件。

## 开启 Ulysses CP

开启 CP 时，配置中需要同时包含下面四部分：

```yaml
distributed:
  # 每个 CP 组包含的 rank 数。大于 1 时开启 Ulysses CP。
  ulysses_parallel_size: 2

  cp_plan:
    ulysses_function_patches:
      # Kimi MLA 全注意力：模型按「CP 是否开启」自行选择内核（CP→flash，否则 eager），
      # CP 包装统一安装在 modeling_kimi_linear.dispatch_attention 上。
      - target_functions:
          - modeling_kimi_linear.dispatch_attention
        type: full_attention

      # Kimi Delta Attention（KDA）：使用支持 Ulysses 的 forward。
      - target_functions:
          - modeling_kimi_linear.KimiDeltaAttention.forward
        type: kda

    # 在完整 CP 组上完成 label shift 和归一化，保证各 rank 的 loss/梯度正确。
    loss_function_patches:
      - target_functions:
          - transformers.loss.loss_utils.ForCausalLMLoss
        type: causal_lm_loss

# Kimi K3 必须先完成文本与图像 token 融合，再切分序列。
module_patches:
  - target: modeling_kimi_k3.KimiK3ForConditionalGeneration.forward
    replacement: fsdp_turbo.models.kimi.kimi_k3_cp.kimi_k3_model_forward
```

各配置项的作用如下：

- `ulysses_parallel_size`：设置 Ulysses CP 并行度；值大于 1 才会应用
  `ulysses_function_patches` 和 CP loss patch。
- `full_attention`：为Kimi的MLA全注意力安装Ulysses all-to-all包装；包装目标是
  `modeling_kimi_linear.dispatch_attention`，具体内核（eager/flash）由模型按CP状态自行选择。
- `kda`：替换每个 `KimiDeltaAttention` 实例的 `forward`，使 KDA 支持序列切分。
- `causal_lm_loss`：labels 保持完整序列，由 CP-aware loss 做全局 shift、切分和归一化。
- `module_patches`：Kimi K3 是多模态模型，图像 token 会改变序列长度，因此必须先融合
  图文 embedding，再按 CP rank 切分；不能在 dataloader 输出处提前切分。

CP 路径的内核选择由 CP 状态驱动：`KimiMLAAttention.forward` 检测到 CP 开启时改走
`modeling_kimi_linear.dispatch_attention`（flash 内核），视觉塔在 forward 时用同一信号
改用融合实现。非 CP（单 rank）路径保持上游逻辑不变：内核与 mask 仍由
`model.attn_implementation` / transformers 的 `_attn_implementation` 决定
（示例 config.yaml 为 `flash_attention_2`）；需要 eager 数值参考时把它设为 `eager`。

## 使用 fla_npu AscendC 算子

从 flash-linear-attention-npu 源码构建与当前 CANN/SoC 匹配的 wheel。新版仓库推荐先做
环境预检，再生成包含 AscendC OPP 和 Python wrapper 的完整 wheel：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/check_npu_env.py

# A2/910B 使用 ascend910b；A3 使用 ascend910_93
FLA_NPU_SOC=ascend910_93 python scripts/build_wheel.py
python -m pip install --force-reinstall --no-cache-dir --no-deps dist/<本次生成的准确wheel文件名>.whl
```

不建议使用旧版裸 `bash build.sh` 流程构建 legacy PyTorch extension；旧
`torchnpugen` 无法解析算子 schema 中的可选返回值 `Tensor?`。安装后确认：

```bash
python -c "import fla_npu; from fla_npu.ops.ascendc import chunk_kda_fwd, chunk_kda_bwd, causal_conv1d, causal_conv1d_bwd"
```

KDA 和 causal conv1d 不需要配置 `module_patches`。模型统一调用 FSDPTurbo 算子注册入口，
运行时依次选择 AscendC、Triton 和 eager 实现；回退到 eager 时会输出 warning。
当前 AscendC KDA 路径用于 `use_cache=False` 的训练，不支持 generation/cache。

## 并行度约束

`torchrun` 的总进程数（world size）必须分别满足以下整除关系：

```text
world_size % (fully_shard_parallel_size * tensor_parallel_size) == 0
world_size % (expert_fully_shard_parallel_size * expert_parallel_size) == 0
world_size % (ulysses_parallel_size * tensor_parallel_size) == 0
```

FSDP、EP 和 CP 使用独立的 device mesh，因此三组并行度不需要彼此相乘。当前
`config.yaml` 的 16 rank 配置为：

```yaml
distributed:
  fully_shard_parallel_size: 16
  tensor_parallel_size: 1
  ulysses_parallel_size: 2
  expert_parallel_size: 16
  expert_fully_shard_parallel_size: 1
```

此外，`ulysses_parallel_size` 必须能够整除 KDA 的 `num_heads` 和全注意力的 query
head 数。图文融合后的序列长度不必手动对齐 CP 并行度，forward 会在切分前自动向右
padding，并通过 `attention_mask` 排除 padding token。

## 环境依赖

在 A5 服务器上执行 Kimi K3 预训练前，需要先下载并安装 `mindspeed_ops` 库（https://gitcode.com/Ascend/MindSpeed-Ops）。

### 安装 ops-transformer 以适配 AscendC 版本 attn_res

Attention Residual（attn_res）通过算子注册入口自动选择后端。在 NPU 上使用 AscendC 实现需要安装
[`ops-transformer`](https://gitcode.com/cann/ops-transformer) 的
`block_attention_residuals` 前向和反向算子。参考版本和安装方式如下：

```bash
git clone https://gitcode.com/cann/ops-transformer.git
cd ops-transformer
git checkout a041b05638
source /usr/local/Ascend/cann/set_env.sh
# 根据设备选择：A2=ascend910b，A3=ascend910_93，950 系列=ascend950
soc_version=ascend910_93
bash build.sh --pkg --soc=${soc_version} \
  --ops=block_attention_residuals,block_attention_residuals_grad -j16
bash build/cann-ops-transformer-custom_linux-aarch64.run
bash build.sh --torch_extension
pip install torch_extension/dist/*.whl --force-reinstall --no-deps
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
```

`${soc_version}` 根据设备选择，例如 Atlas A2 使用 `ascend910b`，Atlas A3 使用
`ascend910_93`，950 系列使用 `ascend950`。缺少扩展时自动回退到 eager 实现。

安装完成后，在训练使用的 Python 环境中验证扩展可以导入：

```bash
python -c "from cann_ops_transformer import block_attention_residuals"
```

上述命令正常退出表示前向 Python 接口可导入；反向算子的正确性仍需通过实际训练验证。
启动训练的每个终端都需要加载对应 CANN 环境，并设置上述 `LD_LIBRARY_PATH`。

## 启动训练

先按实际环境修改 `config.yaml` 中的模型、数据集和并行度，然后运行：

```bash
cd examples/kimi_k3
GPUS_PER_NODE=16 LOG_PATH=./kimi_k3.log bash run.sh
```

也可以直接用torchrun启动（并行度、模型路径等全部由`config.yaml`决定）：

```bash
cd examples/kimi_k3
torchrun --nproc_per_node=16 train.py --config config.yaml
```

## 当前限制

Kimi K3 的 Ulysses CP 当前面向完整序列训练，不支持以下用法：

- generation/cache 输入（`past_key_values`、`cache_position` 或 `use_cache=True`）；
- KDA 的预打包 `cu_seqlens` 输入。

Chunk loss 已支持与 CP 同时开启（见下文"长序列（100k+/1M）训练"），且是超长序列的
必选项。

示例的 `train_step` 已设置 `use_cache=False`。如果使用自定义训练入口，也需要保持该
设置。若模型 remote-code 中的模块路径发生变化，还需要同步修改 `target_functions`
和 `module_patches.target`，否则配置校验或 patch 匹配会直接报错。

## 长序列（100k+/1M）训练

超长序列有三个硬性显存瓶颈，各有对应开关（第三个仅多模态）：

1. **注意力中间量**：eager 的 `[batch, 1, S, S]` mask 与 scores/probs 在 1M 下直接 OOM；
   开启 CP 后模型自动改走 flash 内核（因果由 kernel 原生表达，显存随 S 线性）。
2. **lm_head/logits**：`[local_seq, vocab]` 全量物化约 200 GiB/rank（1M、CP=8、vocab≈160k）；
   开启 chunk loss 后分块计算，峰值降至约 2 GiB（chunk 1024）。
3. **图文合并临时拷贝**：单次赋值物化全量文本 embedding（1M 约 14 GiB，CP 切分前每 rank
   全量执行）；`memory.mm_merge_chunk_tokens` 设为正整数后按块搬运，结果与单次赋值
   逐位一致（1M 多模态建议 `65536`，`0` 关闭）。

推荐组合（以 Ulysses 为例，KCP 同理）：

```bash
NUM_HIDDEN_LAYERS=4 GPUS_PER_NODE=16 LOG_PATH=./kimi_k3_1m.log bash run.sh
```

```yaml
memory:
  chunk_loss: true
  chunk_loss_size: 1024
  mm_merge_chunk_tokens: 65536  # 多模态 1M 建议开启；纯文本路径无影响
  recompute: true
  recompute_plan:
    - language_model.model.layers.{*}
```

`run.sh` 支持的环境变量：`NUM_HIDDEN_LAYERS`（减层调试）。chunk loss 与图文合并分块的
开关都在 `config.yaml` 的 `memory` 段（`chunk_loss` / `chunk_loss_size` /
`mm_merge_chunk_tokens`）；用 torchrun 直启时对应 `train.py --config config.yaml`。

CP 路径不需要手工选择内核：任何 CP 配置（ulysses/kcp/kv_allgather 任一 >1）都会让
模型走 flash 内核。非 CP（单 rank）运行沿用上游 `model.attn_implementation` 的选择
（示例 config.yaml 默认 `flash_attention_2`），需要 eager 数值参考时把它设为 `eager`。

其它注意事项：

- `data.batch_size` 保持 1（KCP 模式强制）；序列长度会被自动右 padding 到 CP 度的
  整数倍，无需手工对齐。
- KDA 的 `skip_kda_recompute` 保持默认 False（kernel 内部反向重算换显存；1M 下开启
  需要多存每层 GB 级的 h 张量）。
- flash 与 eager 存在 kernel 级数值差异，切换后需重新生成精度基线。
- 若 CP 开启但 flash patch 未生效（例如自行裁剪了 patch 列表），flash 前向会显式
  报错而不是静默算错。

## 开启 KCP

默认的 [`config.yaml`](./config.yaml) 已包含 KCP patch 配置，但未开启 KCP。从默认的
Ulysses 模式切换到 KCP 时，只需修改 Context Parallel 并行度：

```yaml
distributed:
  tensor_parallel_size: 1
  kcp_parallel_size: 2
  ulysses_parallel_size: 1

  cp_plan:
    kcp_function_patches:
      # MLA 走 Ulysses all-to-all（1M 推荐）。
      - target_functions:
          - modeling_kimi_linear.dispatch_attention
        type: full_attention

      # KDA 使用状态 Context Parallel。
      - target_functions:
          - modeling_kimi_linear.KimiDeltaAttention.forward
        type: kda

    loss_function_patches:
      - target_functions:
          - transformers.loss.loss_utils.ForCausalLMLoss
        type: causal_lm_loss
```

KCP 和 Ulysses 是互斥模式，`kcp_parallel_size` 与 `ulysses_parallel_size` 不能同时
大于 1。KCP 在同一个 CP 通信组上为两类层分别选择策略：

- **MLA（全注意力）**：`full_attention`（Ulysses all-to-all，与 Ulysses 模式同一
  实现，仅通信组不同）。Kimi K3 有 96 个 head，a2a 每 rank 的 KV 随 cp 缩小，是
  1M 的唯一可行解。
- **KDA（线性注意力）**：按 rank 顺序传递递推状态，不为 KDA 在每个 rank 重建完整序列。

KCP 的 world size 约束为：

```text
world_size % (kcp_parallel_size * tensor_parallel_size) == 0
```

### KCP 当前限制

- `tensor_parallel_size` 必须为 1。
- `kv_allgather_parallel_size` 必须为 1（未配置时默认值为 1）。
- `data.batch_size` 和实际 forward batch size 必须为 1。
- 仅支持三维 BSH 输入，不支持 TND/packed 输入或预打包 `cu_seqlens`。
- 不支持 generation/cache 输入（chunk loss 已支持，见上文）。
- KDA key head dimension 不能超过 256；启用 safe gate 时，`lower_bound` 必须位于
  `[-5, 0)`。

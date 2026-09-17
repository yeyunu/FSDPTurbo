# Kimi K3 训练示例

本目录提供 Kimi K3 多模态 MoE 模型的训练示例。默认的
[`config.yaml`](./config.yaml) 已开启 Ulysses Context Parallel（CP），CP
并行度为 2。

## 开启 Ulysses CP

开启 CP 时，配置中需要同时包含下面四部分：

```yaml
distributed:
  # 每个 CP 组包含的 rank 数。大于 1 时开启 Ulysses CP。
  ulysses_parallel_size: 2

  cp_plan:
    ulysses_function_patches:
      # Kimi MLA 全注意力：在 sequence/head 维度执行 all-to-all。
      - target_functions:
          - modeling_kimi_linear.eager_attention_forward
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
- `full_attention`：为 Kimi 的 MLA 全注意力安装 Ulysses all-to-all 包装。
- `kda`：替换每个 `KimiDeltaAttention` 实例的 `forward`，使 KDA 支持序列切分。
- `causal_lm_loss`：labels 保持完整序列，由 CP-aware loss 做全局 shift、切分和归一化。
- `module_patches`：Kimi K3 是多模态模型，图像 token 会改变序列长度，因此必须先融合
  图文 embedding，再按 CP rank 切分；不能在 dataloader 输出处提前切分。

`train.py` 已将主配置、文本配置和视觉配置的 attention implementation 设为
`eager`。`full_attention` 的目标是 `eager_attention_forward`，使用其他 attention
implementation 时该 patch 不会覆盖实际执行路径。

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

## 启动训练

先按实际环境修改 `config.yaml` 中的模型、数据集和并行度，然后运行：

```bash
cd examples/kimi_k3
GPUS_PER_NODE=16 LOG_PATH=./kimi_k3.log bash run.sh
```

也可以通过命令行临时覆盖 CP 并行度：

```bash
cd examples/kimi_k3
torchrun --nproc_per_node=16 train.py --config config.yaml --ulysses-size 2
```

## 当前限制

Kimi K3 的 Ulysses CP 当前面向完整序列训练，不支持以下用法：

- generation/cache 输入（`past_key_values`、`cache_position` 或 `use_cache=True`）；
- Kimi K3 chunk loss；
- KDA 的预打包 `cu_seqlens` 输入。

示例的 `train_step` 已设置 `use_cache=False`。如果使用自定义训练入口，也需要保持该
设置。若模型 remote-code 中的模块路径发生变化，还需要同步修改 `target_functions`
和 `module_patches.target`，否则配置校验或 patch 匹配会直接报错。

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
      # MLA 在 KCP 通信组上复用 Ulysses sequence/head all-to-all。
      - target_functions:
          - modeling_kimi_linear.eager_attention_forward
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
大于 1。KCP 模式仍在同一个 CP 通信组上为 MLA 执行 Ulysses all-to-all，但 KDA
改为按 rank 顺序传递递推状态，避免为 KDA 在每个 rank 重建完整序列。

KCP 的 world size 约束为：

```text
world_size % (kcp_parallel_size * tensor_parallel_size) == 0
```

### KCP 当前限制

- `tensor_parallel_size` 必须为 1。
- `kv_allgather_parallel_size` 必须为 1（未配置时默认值为 1）。
- `data.batch_size` 和实际 forward batch size 必须为 1。
- 仅支持三维 BSH 输入，不支持 TND/packed 输入或预打包 `cu_seqlens`。
- 不支持 generation/cache 输入和 Kimi K3 chunk loss。
- KDA key head dimension 不能超过 256；启用 safe gate 时，`lower_bound` 必须位于
  `[-5, 0)`。

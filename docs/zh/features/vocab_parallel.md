# 词表并行

FSDPTurbo面向超大词表嵌入表（如Qwen4-Exp的PLE n-gram表，~51B参数）提供词表并行（Vocabulary Parallel，VP），将`nn.Embedding`的`[vocab, dim]`权重沿词表维（dim 0）切分到多个设备，token ID通过All-to-All通信分发到持有对应词表分片的设备上查表，使超大嵌入表**常驻设备且可训练**。VP是专家并行（EP）在嵌入表上的对应物：专家按专家维切分，嵌入表按词表维切分。

## 配置方法

在`distributed`段配置词表并行度与VP计划（以Qwen4-Exp PLE n-gram表为例）：

```yaml
distributed:
  vocab_parallel_size: 8
  vocab_fully_shard_parallel_size: 1
  vocab_data_parallel_size: 1
  vp_plan:
    apply_modules:
      - model.language_model.layers.{*}.ple.ple_embedding.ngram_embedding
```

完整训练示例见`examples/qwen3_8`。

## 参数说明

| 参数 | 说明 |
|---|---|
| `vocab_parallel_size` | 词表并行度，嵌入表按词表维切分到VP组 |
| `vocab_fully_shard_parallel_size` | 词表FSDP2（V-FSDP）并行度，对VP分片进一步全分片；不设置时自动按`world_size // (vocab_parallel_size × vocab_data_parallel_size)`计算 |
| `vocab_data_parallel_size` | 词表数据并行度 |
| `vp_plan.apply_modules` | 应用VP的模块列表，支持`model.language_model.layers.{*}`通配符，匹配目标必须是`nn.Embedding` |
| `vp_plan.apply_vfsdp_modules` | 应用V-FSDP的模块列表，默认与`apply_modules`相同 |
| `vp_plan.gradient_divide_factor` | 回传All-to-All的梯度除法因子，默认`WORLD_SIZE`（与FSDP对稠密参数的梯度平均语义一致） |

## 功能原理

1. **权重切分**：`vocab_parallelize_modules`将匹配到的嵌入权重沿词表维`Shard(0)`切分为DTensor，并记录各rank的词表区间`[start, end)`。对驻留主机的表（如经`/dev/shm`共享的PLE表），仅将本rank的分片搬上加速器，避免整表上卡导致OOM。
2. **All-to-All查表**：模块forward替换为all-to-all dispatcher：按token ID所属rank排序分组 → All-to-All分发ID → 各rank用本地分片`F.embedding`查表 → 可微分的All-to-All回传嵌入向量 → 逆排列恢复输入顺序。VP组同时充当数据并行组（各rank嵌入不同的token）。
3. **V-FSDP二级分片**：`vocab_fully_shard_modules`在VP分片之上叠加FSDP2全分片，单卡存储进一步降为`[vocab/(vp×vfsdp), dim]`，forward解分片只重组本rank的连续词表窗口，任何rank都不会物化整表。
4. **纯VP模式**：当`vocab_data_parallel_size × vocab_fully_shard_parallel_size == 1`时跳过fully_shard，VP参数保持在FSDP单元之外，混合精度转换与梯度除法改在回传All-to-All中完成。

VP使用独立的`vdp/vfsdp/vp` mesh，可与专家并行共存。`FSDPTurbo`在应用FSDP之前执行VP：V-FSDP模块由`vp_plan`接管（相关模式自动从`fsdp_plan.apply_modules`移除），纯VP模块自动加入`fsdp_plan.ignored_modules`，均无需手动调整FSDP计划。

## 模块要求

- 匹配目标必须是`nn.Embedding`，且词表大小`num_embeddings`需能被`vocab_parallel_size`整除。
- 不支持`nn.Embedding`的`max_norm`、`scale_grad_by_freq`与`sparse`选项。

## 注意事项

- **开启条件**：配置`vp_plan`要求`vocab_parallel_size > 1`，否则报错。不配置`vp_plan`时，超大嵌入表可回退到冻结的CPU卸载路径（如`ple_embedding_on_cpu`，表不可训练）。
- **并行度组合约束**：`vocab_data_parallel_size × vocab_fully_shard_parallel_size × vocab_parallel_size`必须等于总设备数。
- **与FSDP的关系**：VP模块不能被祖先FSDP单元接管，配置校验会自动处理模式归属，无需手动调整。
- **显存估算**：以~51B参数（bf16）的PLE表为例，`vp=8, vfsdp=1`时每卡约12.8GB参数量及对应优化器状态。

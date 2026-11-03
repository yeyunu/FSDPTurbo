# 分块损失计算

分块损失（Chunk Loss）将损失计算分块流水处理，避免大词表场景下logits全量物化带来的显存峰值。

## 功能原理

大词表模型（如词表150k+）中，`lm_head`的输出logits形状为`[batch, seq_len, vocab_size]`，全量物化会占用大量显存。Chunk Loss将hidden states沿序列维按`chunk_size`切分，逐块通过`lm_head`计算logits与损失，利用`torch.func.grad_and_value`逐块计算梯度并累加，最后聚合为总损失。

## 配置方法

Chunk Loss通过`config.yaml`的`memory`段开启。开关是否生效取决于所用训练入口的
接线（各示例接法不同，以对应示例的README为准）；完成接线的入口会按配置自动设置
模型属性`enable_chunk_loss` / `chunk_size`，无需手工改模型：

```yaml
memory:
  chunk_loss: true
  chunk_loss_size: 1024   #每块处理的token数；默认512
```

字段校验在`FSDPTurboConfig.__post_init__`的`validate_chunk_loss_config`；chunk loss
不能与head/root的recompute/swap_activation钩子叠加（启动时报错）。开启后模型前向
按块计算lm_head与交叉熵，`logits=None`。

## 与上下文并行（CP）的关系

Kimi K3上Chunk Loss与CP**兼容，且超长序列（100k+）下推荐同时开启**。CP前向（`fsdp_turbo/models/kimi/kimi_k3_cp.py`）检测到
`memory.chunk_loss`后走`cp_chunk_lm_loss`（`fsdp_turbo/distributed/context_parallel/loss.py`）：

- 标签仍按**完整序列**全局移位后切分到本rank（与CP感知损失同一套`_shift_and_shard_labels`），跨rank边界样本不丢失；
- hidden states按`chunk_size`分块，逐块通过`torch.func.grad_and_value`计算logits与梯度，任意时刻只物化一个`[chunk_size, vocab]`块；
- 跨rank归一化沿用CP损失机制：loss和在forward内all-reduce（反向乘`world_size`补偿FSDP平均），分母为全局有效token数；
- 输出`logits=None`（训练器只消费`loss`）。

## 适用场景

- 大词表模型（vocab_size > 100k）
- 超长序列训练（seq_len > 32k）
- 显存紧张、无法承受全量logits物化的场景

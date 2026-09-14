# 激活重计算

激活重计算（梯度检查点）在前向时不保存部分模块的激活值，反向时重新计算，以时间换显存。

## 配置方法

在`memory`段开启重计算并指定作用模块：

```yaml
memory:
  recompute: true
  recompute_plan:
    - model.language_model.layers.{*}
    - model.visual.blocks.{*}
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `recompute` | 激活重计算总开关 |
| `recompute_plan` | 重计算作用模块列表，支持`model.layers.{*}`层级通配符，可只对部分层开启以平衡显存与耗时 |

## 实现机制

FSDPTurbo的重计算基于`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`实现，通过`context_fn`控制重计算区域的随机数状态一致性。重计算wrapper会自动处理以下场景：

- **KV Cache处理**：若模块forward接收`past_key_values` / `past_key_value`参数，重计算前将其置为`None`并将`use_cache`设为`False`，避免KV cache在重计算路径中导致dtype不一致或CheckpointError。
- **递归防护**：通过`is_grad_enabled`与`_in_checkpoint_forward`双标志防止重计算过程中的无限递归。

## 注意事项

- **与上下文并行（CP）的组合**：重计算与Ulysses CP可同时开启，重计算作用于CP切分后的本rank序列片段，两者互不冲突。
- **与CPU Offload的组合**：重计算降低激活显存，CPU Offload降低参数/梯度显存，两者叠加可进一步压缩显存峰值，但会增加计算与数据搬运开销。
- **粒度选择**：建议以Transformer层（`model.layers.{*}`）为重计算单位，逐层开启可在显存与耗时之间取得平衡。全部开启显存最省但耗时增加最多。

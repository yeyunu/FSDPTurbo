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

YAML 的`recompute_plan`只支持模块 pattern 列表。更细的控制可由代码直接传入
`RecomputePlanConfig`：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `apply_modules` | `[]` | 模块 pattern 列表；条目也可写成`module.pattern.method_name`，此时只重计算匹配模块上的指定方法 |
| `flatten_inputs` | `False` | 是否把`(args, kwargs)`整体展平为位置参数后再交给 checkpoint |

## 实现机制

重计算基于`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`实现：

- **KV Cache处理**：若被包装的 forward 形参包含`past_key_values`，调用时通过签名 bind 将其置为
  `None`（位置传参与关键字传参都覆盖）；经`**kwargs`透传的同名实参也会置为`None`，避免 KV cache
  在重算路径中导致 dtype 不一致或 CheckpointError。
- **context_fn由调用方注入**：memory 包不内置任何低精度或随机数上下文。`recompute_modules`
  接受`context_fn_factory`，接线时对每个被包装的模块调用一次并以该模块为参数；返回的
  context_fn 由 checkpoint 每次执行时调用，返回`None`则不向 checkpoint 传 context_fn。
  多个上下文可用`compose_checkpoint_contexts`按序组合。
- **mxfp8组合**：`FSDPTurbo.apply_recompute_modules`在开启 mxfp8 时负责注入相位切换上下文，
  并在包装前完成模块打标；memory 包不依赖 quantization。
- **方法级条目**：`module.pattern.method_name`只包装匹配模块上的同名方法，向 factory 传`None`。
- **去重与嵌套**：多个 pattern 命中同一模块或方法时只包装一次；`reject_nested=True`时在接线期拒绝
  模块间的祖先/后代嵌套。方法相关的动态嵌套仅告警，需由调用方排除。

## 注意事项

- **与上下文并行（CP）的组合**：重计算与Ulysses CP可同时开启，重计算作用于CP切分后的本rank序列片段，两者互不冲突。
- **与CPU Offload的组合**：重计算降低激活显存，CPU Offload降低参数/梯度显存，两者叠加可进一步压缩显存峰值，但会增加计算与数据搬运开销。
- **粒度选择**：建议以Transformer层（`model.layers.{*}`）为重计算单位，逐层开启可在显存与耗时之间取得平衡。全部开启显存最省但耗时增加最多。

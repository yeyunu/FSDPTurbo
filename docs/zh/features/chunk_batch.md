# 微批次切分

微批次切分（Chunk Batch）将指定模块的forward沿batch维切分为更小的微批次逐段执行，输出再拼接回原形状，降低单个模块前向/反向的显存峰值。

## 功能原理

当某个模块的激活值或临时workspace峰值与batch size成正比时，将输入沿batch维切片可以显著降低显存峰值。例如输入形状为`[4, 2048, hidden]`，配置`chunk_mbs=1`时，该模块的forward会以`[1, 2048, hidden]`的输入执行4次，输出再拼接回`[4, 2048, hidden]`，对调用方透明。

## 配置方法

在`config.yaml`的`memory`段开启：

```yaml
memory:
  chunk_batch: true
  chunk_batch_plan:
    chunk_mbs: 1
    apply_modules:
      - model.layers.{*}
    batch_dim: 0
    chunk_arg_indexes: [0]
    chunk_kwarg_names: ["hidden_states", "attention_mask"]
```

| 参数 | 说明 |
|---|---|
| `chunk_batch` | 微批次切分总开关 |
| `chunk_batch_plan.chunk_mbs` | 切分后的微批次大小，默认`1` |
| `chunk_batch_plan.apply_modules` | 应用切分的模块列表，支持`model.layers.{*}`通配符 |
| `chunk_batch_plan.batch_dim` | 表示batch的张量维度，HuggingFace decoder层通常为`0` |
| `chunk_batch_plan.chunk_arg_indexes` | 需按batch切分的位置参数下标列表 |
| `chunk_batch_plan.chunk_kwarg_names` | 需按batch切分的关键字参数名列表 |

> [!NOTE]
>
> 开启`chunk_batch`时必须指定`chunk_arg_indexes`或`chunk_kwarg_names`，否则启动时会报配置校验错误。

## 适用场景

- 大batch size训练时单层前向显存峰值过高
- 模块临时workspace与batch size成正比（如大词表投影、MoE分组计算）
- 与重计算、激活值交换等手段叠加使用

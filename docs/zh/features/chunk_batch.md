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

## 与上下文并行（CP）的关系

微批次切分与CP**不兼容，不能同时开启**，当前实现不支持二者组合使用。

两者虽然都在大batch / 长序列场景下缓解显存压力，但切分路径相互冲突：

- CP开启后，模型主干forward被替换为CP感知实现，输入在进入decoder层前已沿**序列维**切分到各CP rank，注意力计算也被替换为CP感知版本，通过跨rank通信交换序列/头维信息（如Ulysses的all-to-all），并依据本rank的局部输入重建全局因果掩码。
- 微批次切分则包装decoder层的forward（见`fsdp_turbo/memory/chunk_batch/chunk_batch.py`），将输入沿**batch维**切片逐块执行，输出再拼接回原形状。
- 两种机制都以接管forward入口的方式生效，叠加后序列切分、掩码重建、跨rank通信与输出拼接所依赖的形状约定不再被保证，当前实现未提供组合适配。

> [!NOTICE]
>
> 与Chunk Loss不同（开启CP后设置`enable_chunk_loss`会直接报错），微批次切分与CP同时开启**没有启动或前向阶段的显式校验**，不会立刻报错，但行为未定义，务必避免该配置组合。

## 适用场景

- 大batch size训练时单层前向显存峰值过高
- 模块临时workspace与batch size成正比（如大词表投影、MoE分组计算）
- 与重计算、激活值交换等手段叠加使用

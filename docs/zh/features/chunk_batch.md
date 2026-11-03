# 微批次切分

微批次切分（Chunk Batch）将指定模块的forward沿batch维切分为更小的微批次逐段执行，输出再拼接回原形状，降低单个模块前向/反向的显存峰值。

## 背景与挑战

大batch size / 长序列训练场景下，部分模块的显存占用与batch size成正比：

- decoder层的激活值随batch线性增长，单层前向的激活峰值可能远超均摊到每个样本的水平；
- 大词表投影、MoE分组计算等模块在forward过程中产生与batch size成正比的临时workspace；
- 当batch维无法进一步缩小（如吞吐要求、数据并行度限制）时，即便整体batch大小合适，单个模块的计算峰值仍可能触碰显存上限，导致OOM。

重计算（以时间换激活显存）、激活值offload（以搬运换激活显存）都无法压缩模块forward过程中瞬时产生的临时workspace峰值，需要一种在batch维上做切分的手段。

## 解决方案

微批次切分将配置指定的模块forward沿batch维切分为若干微批次，逐段执行后再将输出拼接回原形状：

- 对调用方完全透明（前提见下）：模块的输入输出形状与未开启时一致，不侵入训练代码；
- 显存峰值由`O(full_batch)`降为`O(chunk_mbs)`：与batch size成正比的激活值、临时workspace只按单个微批次的规模物化；
- 执行语义与整batch一致（同一前提）：每个微批次独立走一遍模块forward，反向在微批次粒度上正常累积梯度。

上述透明性与语义一致性的共同前提是：`chunk_arg_indexes`与`chunk_kwarg_names`必须覆盖该模块forward入参中**全部携带batch维的tensor输入**。未列入切分配置的batch维tensor会以完整batch原样透传，而其余输入已被切为`chunk_mbs`大小，同一forward内部将同时收到首维为`full_batch`与`chunk_mbs`的两种tensor——典型如`hidden_states`被切分而`attention_mask`未切分时，前向阶段直接命中形状不匹配，或在广播语义下静默算出错误结果。

## 核心工作机制

微批次切分由`fsdp_turbo/memory/chunk_batch/chunk_batch.py`中的`chunk_batch_modules`实现：按`apply_modules`模式匹配模块并包装其forward，在`FSDPTurbo`初始化流程中自动应用。包装后的forward执行流程：

1. **推断完整batch size**：从第一个配置的位置参数（`chunk_arg_indexes[0]`）或关键字参数（`chunk_kwarg_names[0]`）沿`batch_dim`读取；推断不到tensor输入时报`ValueError`。
2. **按需跳过**：当完整batch size ≤ `chunk_mbs`时，切分只会增加开销，直接透传原始forward。
3. **逐微批次执行**：切分份数为`ceil(full_batch / chunk_mbs)`（注意`chunk_mbs`是每个微批次的batch size，不是切分份数）。每个微批次只对配置的位置参数下标与关键字参数名做切分；切分递归作用于tensor及tuple/list/mapping嵌套结构，非tensor值（标志位、cache对象、`None`等）原样透传。例如输入形状为`[4, 2048, hidden]`、`chunk_mbs=1`时，forward以`[1, 2048, hidden]`的输入执行4次。
4. **拼接输出**：各微批次输出沿`batch_dim`拼接回`[4, 2048, hidden]`，支持tensor及tuple/list（首元素为`hidden_states`）形式的返回值。

在`FSDPTurbo`初始化的特性包装顺序中（`recompute` → `swap_activation` → `chunk_batch`），微批次切分位于**最外层**：输入先按微批次切分，每个微批次再独立经过激活值交换、重计算包装的模块forward，最终调用链为：

```
forward() → chunk batch 切微批次（最外层）
              → 激活值交换（swap activation）包装
                    → 重计算包装
                        → 原始单块 forward（最内层）
```

因此微批次切分可与重计算、激活值交换等显存优化特性叠加使用：前者压缩与batch维成正比的峰值，后者压缩与激活规模成正比的峰值，作用维度互补。

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
> 开启`chunk_batch`时必须指定`chunk_arg_indexes`或`chunk_kwarg_names`，否则启动时会报配置校验错误；二者还需覆盖该模块全部携带batch维的tensor输入，否则前向阶段会因形状不匹配报错或在广播语义下产出错误结果（见“解决方案”一节）。

## 与上下文并行（CP）的关系

微批次切分与CP**不兼容，不能同时开启**，当前实现不支持二者组合使用。

两者虽然都在大batch / 长序列场景下缓解显存压力，但切分路径相互冲突：

- CP开启后，模型主干forward被替换为CP感知实现，输入在进入decoder层前已沿**序列维**切分到各CP rank，注意力计算也被替换为CP感知版本，通过跨rank通信交换序列/头维信息（如Ulysses的all-to-all），并依据本rank的局部输入重建全局因果掩码。
- 微批次切分则包装decoder层的forward（见`fsdp_turbo/memory/chunk_batch/chunk_batch.py`），将输入沿**batch维**切片逐块执行，输出再拼接回原形状。
- 两种机制都以接管forward入口的方式生效，叠加后序列切分、掩码重建、跨rank通信与输出拼接所依赖的形状约定不再被保证，当前实现未提供组合适配。

> [!NOTICE]
>
> 微批次切分与CP同时开启**没有启动或前向阶段的显式校验**，不会立刻报错，但行为未定义，务必避免该配置组合。

## 适用场景

- 大batch size训练时单层前向显存峰值过高
- 模块临时workspace与batch size成正比（如大词表投影、MoE分组计算）
- 与重计算、激活值交换等手段叠加使用

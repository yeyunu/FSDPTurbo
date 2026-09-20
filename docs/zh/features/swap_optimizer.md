# 优化器状态交换

优化器状态交换（Swap Optimizer）将优化器状态卸载到CPU内存，更新时按显存余量分批回装，降低NPU显存占用。

## 功能原理

AdamW等优化器的状态（一阶动量`exp_avg`、二阶动量`exp_avg_sq`）通常占用与参数相当甚至更多的显存。Swap Optimizer通过`AdamWSwap`优化器实现：

1. 初始化时，将优化器状态从device复制到pinned CPU内存。
2. `step()`执行时，按当前可用显存批量将state回装到device，计算参数更新后再交换回host。
3. 通过分批回装，将峰值显存从"全部state"降低为"单批state"。

## 配置方法

在`config.yaml`的`optimizer`段将优化器类型设为`AdamWSwap`：

```yaml
optimizer:
  optimizer_type: "AdamWSwap"
  lr: 3.0e-4
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1.0e-8
  mem_fraction_static: 0.9                    # 可选：批量 H2D 回装使用的空闲显存比例，默认 0.9
  state_keys: ["exp_avg", "exp_avg_sq"]       # 可选：参与交换的 state 键，缺省使用 AdamWSwap 内部默认
```

| 参数 | 说明 |
|---|---|
| `optimizer_type` | 设为`AdamWSwap`即启用优化器状态交换 |
| `mem_fraction_static` | 批量回装时允许使用的空闲显存比例，默认`0.9` |
| `state_keys` | 参与交换的优化器状态键列表，缺省使用AdamWSwap内部默认（AdamW为`exp_avg`、`exp_avg_sq`） |

`AdamWSwap`已注册到优化器工厂（OptimizerFactory），其余优化器参数（lr、betas、weight_decay等）与标准AdamW一致。参考示例：`examples/deepseekv4/config.yaml`。

## 与FSDP2 CPU Offload的区别

| 特性 | Swap Optimizer | FSDP2 `cpu_offload` |
|---|---|---|
| 卸载对象 | 仅优化器状态 | 参数 + 梯度 + 优化器状态 |
| 卸载粒度 | 按batch分批回装 | 整体卸载/回装 |
| 通信开销 | 较低（仅state） | 较高（参数+梯度） |
| 适用场景 | 优化器状态占显存主导 | 整体显存紧张 |

两者可组合使用，但需注意叠加后的数据搬运开销。

## 适用场景

- 优化器状态占用显存比例高（大模型 + AdamW）
- 参数已通过FSDP2分片，但优化器状态仍占大量显存

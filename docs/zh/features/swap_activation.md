# 激活值交换

激活值交换（Swap Activation）在计算过程中异步将激活值卸载到CPU内存，反向时再回传，以PCIe带宽换取设备显存。

## 功能原理

FSDPTurbo提供`TensorSwapManager`管理激活值的异步搬运。对指定模块应用交换后，其forward被包装：

1. 前向计算完成后，激活值通过`SwapTensor`异步从device卸载到host（pinned CPU内存）。
2. 反向计算需要该激活值时，异步从host回传到device。
3. 通过多流异步执行，使数据搬运与计算重叠，降低显存峰值。

## 配置方法

在`config.yaml`的`memory`段开启，模块匹配规则与重计算（recompute_plan）一致：

```yaml
memory:
  swap_activation: true
  swap_activation_plan:
    apply_modules:
      - model.layers.{*}
```

| 参数 | 说明 |
|---|---|
| `swap_activation` | 激活值交换总开关 |
| `swap_activation_plan.apply_modules` | 应用交换的模块列表，支持`model.layers.{*}`层级通配符。开启交换时该列表不能为空，否则启动报错 |

配置后由`FSDPTurbo`初始化流程自动应用，无需额外代码。

## 交换重计算边界输入

通过`memory.recompute_plan.swap_inputs`交换`recompute_plan.apply_modules`指定的checkpoint边界输入。配置示例见 [激活重计算](./recompute.md)。

## 适用场景

- 长序列训练（激活显存占比高）
- 大batch size场景
- 显存紧张、需进一步压缩激活显存的场景

## 注意事项

- 交换会引入PCIe数据搬运开销，需确保计算与搬运能有效重叠，否则可能降低吞吐。
- 建议仅对激活显存占用大的模块（如decoder层）开启。
- 可与重计算、CPU Offload等其他内存优化手段组合使用。

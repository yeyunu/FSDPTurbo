# 模型性能调优

本文档介绍FSDPTurbo训练性能调优方法，从性能分析、通信优化、内存优化、量化加速与MoE专项五个维度展开。

## 性能分析方法

性能调优的前提是准确定位瓶颈。FSDPTurbo内置PyTorch Profiler支持，详见[性能数据采集](../../tools/profiling.md)。通过Profiling结果可判断：

- **计算瓶颈**：算子耗时占比高，需优化计算效率或开启量化。
- **通信瓶颈**：AllGather / ReduceScatter耗时占比高，需调整预取策略或并行度。
- **显存瓶颈**：频繁OOM或显存碎片化，需开启重计算 / CPU offload等内存优化。

## 通信优化

| 手段 | 配置项 | 说明 |
|---|---|---|
| 前向预取 | `fsdp_plan.num_to_forward_prefetch` | 前向时提前AllGather后续N层参数，隐藏通信延迟。建议从1开始递增测试。 |
| 反向预取 | `fsdp_plan.num_to_backward_prefetch` | 反向时提前预取参数，与反向计算重叠。 |
| 细粒度通信调度 | `fsdp_plan.hook_modules` | 指定通信挂钩模块（如`model.layers.{*}`），实现逐层通信调度。 |
| 自定义FSDP2实现 | `fsdp_plan.fsdp_implementation: custom` | 使用FSDPTurbo自定义FSDP2路径，提供更细的多流通信调度能力。 |

## 内存优化与吞吐权衡

显存优化通常以计算开销换取显存空间，需根据显存余量与吞吐目标综合选择：

| 手段 | 显存收益 | 耗时影响 | 配置方式 |
|---|---|---|---|
| 激活重计算 | 高（减少激活显存） | 中（重算前向） | `memory.recompute: true` |
| 微批次切分 | 中（降低单层峰值） | 低-中 | `memory.chunk_batch: true` |
| 激活值交换 | 中（激活卸载CPU） | 低-中（PCIe搬运） | `memory.swap_activation: true` |
| 优化器状态交换 | 高（优化器状态卸载） | 低 | `optimizer.optimizer_type: "AdamWSwap"` |
| CPU Offload | 中（参数/梯度卸载） | 中（PCIe搬运） | `fsdp_plan.cpu_offload: true` |

详见各特性文档：[激活重计算](../../features/recompute.md)、[微批次切分](../../features/chunk_batch.md)、[激活值交换](../../features/swap_activation.md)、[优化器状态交换](../../features/swap_optimizer.md)、[全分片数据并行](../../features/fully_shard_parallel.md)。

推荐组合策略：

1. 先开启重计算（最常用、收益最稳定）。
2. 显存仍不足时叠加激活值交换或`AdamWSwap`（优化器状态占用大时优先）。
3. 整体显存紧张再开启CPU Offload。

## 量化加速

MXFP8量化训练可同时降低显存占用与提升计算吞吐：

- 开启`quantization.quantization_plan.quant_format: E4M3`或`HIF8`。
- 配合`enable_fsdp_low_precision_all_gather: true`降低通信量。
- MoE模型可额外开启`quant_gmm: true`加速专家计算。

详见[量化训练](../../features/quantization.md)。

## MoE模型专项

| 手段 | 说明 |
|---|---|
| Dispatcher选型 | `fused` / `mc2`性能优于`eager`，优先使用；`domino`适用于特定负载模式。 |
| EFSDP并行度 | 专家参数量大时开启`apply_efsdp_modules`，对专家参数做FSDP2全分片。 |
| `fixed_router`固定路由 | 调试手段：以固定均匀路由替代实际路由，避免负载不均带来的性能与精度问题，仅在`expert_parallel_size > 1`时生效。 |

详见[专家并行](../../features/expert_parallel.md)。

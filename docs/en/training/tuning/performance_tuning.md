# Model Performance Tuning

This document describes FSDPTurbo training performance tuning methods, covering five dimensions: performance analysis, communication optimization, memory optimization, quantization acceleration, and MoE-specific optimization.

## Performance Analysis Methods

The prerequisite for performance tuning is accurately identifying the bottleneck. FSDPTurbo has built-in PyTorch Profiler support; see [Performance Data Profiling](../../tools/profiling.md) for details. Based on the profiling results, you can determine:

- **Compute bottleneck**: Operator time accounts for a high proportion; you need to optimize compute efficiency or enable quantization.
- **Communication bottleneck**: AllGather / ReduceScatter time accounts for a high proportion; you need to adjust the prefetch strategy or the parallelism.
- **Memory bottleneck**: Frequent OOM or memory fragmentation; you need to enable memory optimizations such as recomputation / CPU offload.

## Communication Optimization

| Approach | Configuration Item | Description |
|---|---|---|
| Forward prefetch | `fsdp_plan.num_to_forward_prefetch` | AllGather the parameters of the next N layers in advance during the forward pass to hide communication latency. It is recommended to test by incrementing from 1. |
| Backward prefetch | `fsdp_plan.num_to_backward_prefetch` | Prefetch parameters in advance during the backward pass and overlap with backward computation. |
| Fine-grained communication scheduling | `fsdp_plan.hook_modules` | Specify the communication hook modules (for example, `model.layers.{*}`) to achieve layer-by-layer communication scheduling. |
| Custom FSDP2 implementation | `fsdp_plan.fsdp_implementation: custom` | Use the FSDPTurbo custom FSDP2 path, which provides finer-grained multi-stream communication scheduling capability. |

## Memory Optimization and Throughput Trade-offs

Memory optimization usually trades compute overhead for memory. Choose comprehensively based on the available memory and the throughput target:

| Approach | Memory Benefit | Time Impact | Configuration |
|---|---|---|---|
| Activation recomputation | High (reduces activation memory) | Medium (recomputes the forward pass) | `memory.recompute: true` |
| Batch chunking | Medium (reduces per-layer peak) | Low-medium | `memory.chunk_batch: true` |
| Activation swapping | Medium (offloads activations to CPU) | Low-medium (PCIe transfer) | `memory.swap_activation: true` |
| Optimizer state swapping | High (offloads optimizer states) | Low | `optimizer.optimizer_type: "AdamWSwap"` |
| CPU Offload | Medium (offloads parameters/gradients) | Medium (PCIe transfer) | `fsdp_plan.cpu_offload: true` |

For details, see the feature documents: [Activation Recomputation](../../features/recompute.md), [Batch Chunking](../../features/chunk_batch.md), [Activation Swapping](../../features/swap_activation.md), [Optimizer State Swapping](../../features/swap_optimizer.md), and [Fully Sharded Data Parallelism](../../features/fully_shard_parallel.md).

Recommended combined strategy:

1. Enable recomputation first (the most commonly used approach with the most stable benefit).
2. If memory is still insufficient, add activation swapping or `AdamWSwap` (prioritize when optimizer state usage is large).
3. Enable CPU offload only when overall memory is tight.

## Quantization Acceleration

MXFP8 quantization training can both reduce memory usage and improve compute throughput:

- Enable `quantization.quantization_plan.quant_format: E4M3` or `HIF8`.
- Combine with `enable_fsdp_low_precision_all_gather: true` to reduce communication volume.
- For MoE models, you can additionally enable `quant_gmm: true` to speed up expert computation.

For details, see [Quantization Training](../../features/quantization.md).

## MoE-Specific Optimization

| Approach | Description |
|---|---|
| Dispatcher selection | `fused` / `mc2` outperform `eager`; use them first. `domino` suits specific load patterns. |
| EFSDP parallelism | Enable `apply_efsdp_modules` when the expert parameter count is large to apply FSDP2 full sharding to the expert parameters. |
| `fixed_router` fixed routing | A debugging tool: replaces the actual routing with fixed uniform routing to avoid performance and accuracy issues caused by uneven load. It takes effect only when `expert_parallel_size > 1`. |

For details, see [Expert Parallelism](../../features/expert_parallel.md).

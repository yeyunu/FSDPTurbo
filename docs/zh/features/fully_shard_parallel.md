# 全分片数据并行（FSDP2）

FSDPTurbo的核心是基于PyTorch FSDP2的全分片数据并行：模型参数、梯度和优化器状态均按DeviceMesh分片存储，前向时按需AllGather参数、反向时ReduceScatter梯度，大幅降低单卡显存占用。

## 基础用法

通过Python API配置并行计划，并将FSDPTurbo应用到模型：

```python
from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import (
    DistributedConfig,
    FSDPPlanConfig,
    FSDPTurboConfig,
)

config = FSDPTurboConfig(
    distributed=DistributedConfig(
        fully_shard_parallel_size=8,
        fsdp_plan=FSDPPlanConfig(
            param_dtype="bf16",
            reduce_dtype="fp32",
            num_to_forward_prefetch=1,
            num_to_backward_prefetch=1,
        ),
    ),
)
model = FSDPTurbo(config, model)
```

对应YAML配置：

```yaml
distributed:
  fully_shard_parallel_size: 8
  fsdp_plan:
    apply_modules:
      model.layers.{*}: {}
      model.embed_tokens: {}
      lm_head: {}
    param_dtype: "bf16"
    reduce_dtype: "fp32"
    num_to_forward_prefetch: 1
    num_to_backward_prefetch: 1
```

## 主要能力

### 混合精度

通过`param_dtype` / `reduce_dtype` / `output_dtype`分别控制前向计算、梯度归约与输出精度。典型配置为`param_dtype: bf16` + `reduce_dtype: fp32`，兼顾训练速度与数值稳定性。

### 通信预取

- `num_to_forward_prefetch`：前向时提前预取后续N层的全量参数，隐藏AllGather通信。
- `num_to_backward_prefetch`：反向时提前预取后续N层参数，隐藏通信。
- `prefetch_modules`：可选，按前向实际执行顺序排列的`apply_modules`模式列表，须将每个模式恰好列出一次。配置初始化时检查列表类型及模式是否完整、无重复；校验不会改变用户指定的顺序。模式内按模块注册顺序展开，重叠匹配的模块只保留首次出现；反向使用展开列表的逆序。未配置时沿用模块注册顺序。
- `hook_modules`：指定通信挂钩模块（如`model.language_model.layers.{*}`），实现细粒度通信调度。

多模态模型的执行顺序可能与模块注册顺序不同。例如DeepSeek-V4.1先运行视觉塔和对齐器，再进入语言层，应在其`fsdp_plan`中配置：

```yaml
prefetch_modules:
  - model.embed
  - model.vision
  - model.aligner
  - model.layers.{*}
  - model.head
```

`examples/deepseek_v41/config_A5.yaml` 使用上述整塔预取顺序，前后向预取深度均为 1。`config_A3.yaml` 和 `config_tnd.yaml` 对视觉塔分块包装，预取顺序为 `model.embed`、`model.vision.patch_embed`、`model.vision.blocks.{*}`、`model.aligner`、`model.layers.{*}`、`model.head`，前后向预取深度也均为 1。

该顺序只影响参数预取，不改变FSDP包装顺序或模型计算。各rank必须遵循一致的通信顺序；若模型反向顺序不是前向逆序，应关闭显式反向预取。

### CPU Offload

显存不足时，可开启参数与梯度的CPU卸载：

```yaml
fsdp_plan:
  cpu_offload: true    # 参数/梯度卸载到 CPU，进一步降低显存占用
  pin_memory: true
```

### 选择性分片

- `apply_modules`：仅对指定模块做FSDP2分片，支持`model.layers.{*}`层级通配符。
- `ignored_modules`：排除指定模块（如冻结的视觉编码器）。

### 自定义实现路径

`fsdp_implementation: native`使用PyTorch原生FSDP2；`fsdp_implementation: custom`使用FSDPTurbo自定义实现，提供更细的通信调度能力。

## 断点续训

通过`checkpoint`配置段控制权重与优化器状态的保存/恢复，详见[断点续训](./checkpoint_resume.md)。

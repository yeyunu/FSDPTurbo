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
- `num_to_backward_prefetch`：反向时提前预取后续N层参数，隐藏通信。设为`0`时关闭反向预取，包括FSDP2自带的一层默认预取。
- 专家参数前向预取：`num_to_forward_prefetch > 0`且配置`ep_plan.apply_efsdp_modules`时，每层forward入口除预取后续层外，还会预取本层及后续层的experts参数，使专家AllGather与attention计算重叠，无需额外开关；experts单元自身不持有forward预取列表，避免激活重算期从experts的pre-forward带出错误方向的预取。`custom`实现还通过`hook_modules`在层入口发射本层experts。
- `prefetch_modules`：可选，按前向实际执行顺序排列的`apply_modules`模式列表，须将每个模式恰好列出一次。配置初始化时检查列表类型及模式是否完整、无重复；校验不会改变用户指定的顺序。模式内按模块注册顺序展开，重叠匹配的模块只保留首次出现；反向使用展开列表的逆序。未配置时沿用模块注册顺序。
- `hook_modules`：指定通信挂钩模块（如`model.language_model.layers.{*}`），实现细粒度通信调度。
- `prefetch_wait_a2a`（默认`false`）：将每层forward预取的AllGather（含上述层入口发射的专家参数AllGather）排到上一层专家a2a（dispatch/combine）完成事件之后，避免预取AG与专家a2a抢占带宽。要求配置`ep_plan.apply_efsdp_modules`；未开启`backward_prefetch_on_moe_input`时还要求`num_to_forward_prefetch > 0`。
- `backward_prefetch_on_moe_input`（默认`false`）：MoE层的反向参数预取发射点从FSDP pre-backward hook挪到MoE forward输入tensor的grad hook——该梯度就绪即本层dispatch a2a反向刚完成，预取AllGather与层内纯计算反向重叠，不再与a2a冲突；dense层与专家单元通过`[self]`占位阻断FSDP2默认反向预取。仅支持`fsdp_implementation: native`，且要求`num_to_backward_prefetch > 0`并配置`ep_plan.apply_efsdp_modules`。与`prefetch_wait_a2a`组合使用时，预取AllGather流会先等待本层a2a反向完成事件。当前仅支持粗粒度切分：layer整体作为一个FSDP单元（如`model.layers.{*}`）、experts由`apply_efsdp_modules`独立包裹；层内`self_attn`、`mlp`等子单元独立包裹的细粒度配置不支持。

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

### 分片后物化参数

DeepSeek-V4.1 示例先通过 `prepare_deepseek_v41_model_for_fsdp` 将普通参数转为 meta，
再应用 EP/FSDP 切分。加载 HF 初始权重前，调用 `initialize_deepseek_v41_model(..., initialize_weights=False)`
只分配本地参数 shard 的真实存储，不进行随机初始化，避免权重复制到 meta 参数成为 no-op。
随机初始化路径使用默认的 `initialize_weights=True`，完成物化与随机初始化。

CPU Offload 开启时本地参数在 CPU 物化，否则在当前加速器物化；共享参数别名、已有参数值
和 buffer 保持不变。Host Engram 表仍由自己的 EP hook 分配本地行段。
优化器在物化和权重加载完成后创建，完整调用示例见
[DeepSeek-V4.1 README](../../../examples/deepseek_v41/README.md)。

### 选择性分片

- `apply_modules`：仅对指定模块做FSDP2分片，支持`model.layers.{*}`层级通配符。
- `ignored_modules`：排除指定模块（如冻结的视觉编码器）。

### 自定义实现路径

`fsdp_implementation: native`使用PyTorch原生FSDP2；`fsdp_implementation: custom`使用FSDPTurbo自定义实现，提供更细的通信调度能力。

## 断点续训

通过`checkpoint`配置段控制权重与优化器状态的保存/恢复，详见[断点续训](./checkpoint_resume.md)。

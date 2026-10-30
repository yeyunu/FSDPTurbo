# Fully Sharded Data Parallel (FSDP2)

The core of FSDPTurbo is fully sharded data parallelism based on PyTorch FSDP2: model parameters, gradients, and optimizer states are all stored sharded across the DeviceMesh, parameters are AllGathered on demand during the forward pass, and gradients are ReduceScattered during the backward pass, greatly reducing the per-card memory usage.

## Basic Usage

Configure the parallel plan through the Python API and apply FSDPTurbo to the model:

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

The corresponding YAML configuration:

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

## Key Capabilities

### Mixed Precision

`param_dtype`/`reduce_dtype`/`output_dtype` control the precision of the forward computation, gradient reduction, and output, respectively. A typical configuration is `param_dtype: bf16` + `reduce_dtype: fp32`, which balances training speed and numerical stability.

### Communication Prefetch

- `num_to_forward_prefetch`: prefetches the full parameters of the next N layers ahead of time during the forward pass, hiding the AllGather communication.
- `num_to_backward_prefetch`: prefetches the parameters of the next N layers ahead of time during the backward pass, hiding the communication.
- `hook_modules`: specifies the communication hook modules (such as `model.language_model.layers.{*}`), enabling fine-grained communication scheduling.

### CPU Offload

When memory is insufficient, parameter and gradient offloading to the CPU can be enabled:

```yaml
fsdp_plan:
  cpu_offload: true    # Offloads parameters/gradients to the CPU to further reduce memory usage
  pin_memory: true
```

### Selective Sharding

- `apply_modules`: applies FSDP2 sharding only to the specified modules, supporting `model.layers.{*}` level wildcards.
- `ignored_modules`: excludes the specified modules (such as a frozen vision encoder).

### Custom Implementation Path

`fsdp_implementation: native` uses PyTorch's native FSDP2; `fsdp_implementation: custom` uses FSDPTurbo's custom implementation, which provides finer-grained communication scheduling capabilities.

## Checkpoint-based Resumable Training

The `checkpoint` configuration section controls the saving/restoring of weights and optimizer states. See [Checkpoint-based Resumable Training](./checkpoint_resume.md) for details.

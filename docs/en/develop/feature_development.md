# Feature Development Guide

This document describes how to develop a new FSDPTurbo feature, including architecture description, development tutorial, and development practice guide.

## Overview

FSDPTurbo uses a **configuration-driven** feature management architecture: features are enabled through YAML configuration and applied as one-time transformations (sharding/wrapping/conversion/replacement) to the `nn.Module` tree during model construction, while the training loop itself is unaware of features. Core components:

| Component | Location | Description |
|---|---|---|
| `FSDPTurboConfig` | `fsdp_turbo/fsdp_turbo_config.py` | Feature configuration definitions, YAML parsing, and validation |
| `FSDPTurbo` | `fsdp_turbo/fsdp_turbo.py` | Main entry point; applies features to the model in a fixed order |
| `apply_module_patches`/`patch_namespace_members` | `fsdp_turbo/utils/patch.py` | Function/method-level replacement with idempotent installation |
| `module_name_match` | `fsdp_turbo/utils/str_match.py` | Module name wildcard matching |
| `register_op`/`dispatch_op` | `fsdp_turbo/ops/registry.py` | Operator registration and dispatch across multiple backends (CPU/NPU) |

Developers can add a new feature by simply adding one configuration item and one module transformation function, and integrating them into `FSDPTurbo`.

### Feature Application Order

When constructing `FSDPTurbo(config, model)`, transformations are applied to the model in the following order:

| Order | Stage | Description |
|---|---|---|
| 1 | `init_parallel_state` | Initialize process groups and the device mesh |
| 2 | Quantization conversion | Weight format conversion such as MXFP8/W8A16 |
| 3 | Tensor parallelism (TP) | Column/row-parallel sharding of Linear |
| 4 | Post-TP capture | Restore mesh and placements on quantized Linear after sharding |
| 5 | Context parallelism (CP) | Ulysses/KV-AllGather attention, packed-TND runtime and loss patch |
| 6 | `module_patches` | User-configured function replacements |
| 7 | Expert parallelism (EP/EFSDP) | MoE expert sharding and dispatcher |
| 8 | Vocabulary parallelism (VP/V-FSDP) | Vocabulary-dim sharding and V-FSDP |
| 9 | Pre-FSDP module hooks | Nonstandard storage prepares itself and joins the FSDP ignore list |
| 10 | Activation recomputation | Wrap forward with checkpoint |
| 11 | Activation swapping | Wrap forward with swap_activation |
| 12 | Batch chunking | Wrap forward with chunk_batch |
| 13 | FSDP2 sharding | Wrap with `fully_shard`; must be executed last |

> [!NOTE]
> When adding a new feature, confirm its insertion position: all transformations that depend on the original module structure must be applied before FSDP2 sharding; `module_patches` are applied before FSDP, and a failed target match raises an error directly.

## Development Process

The ChunkBatch (batch chunking) feature is used as an example.

### Defining the Configuration

Add the feature switch and plan configuration in `fsdp_turbo/fsdp_turbo_config.py`:

```python
# fsdp_turbo/fsdp_turbo_config.py
@dataclass
class ChunkBatchPlanConfig:
    chunk_mbs: int = 1                 # Number of samples per micro-batch
    apply_modules: List[str] = None    # Module name patterns, e.g. "model.layers.{*}"
    batch_dim: int = 0                 # Dimension of the batch
    chunk_arg_indexes: Optional[List[int]] = None   # Indexes of positional arguments to chunk
    chunk_kwarg_names: Optional[List[str]] = None  # Names of keyword arguments to chunk


@dataclass
class MemoryConfig:
    recompute: bool = False
    recompute_plan: List[str] = None
    chunk_batch: bool = False                                  # Feature switch
    chunk_batch_plan: Optional[ChunkBatchPlanConfig] = None    # Feature plan
```

The following notes apply:

- The switch (`chunk_batch`) and the plan (`chunk_batch_plan`) are separated, so there is zero overhead when the feature is disabled.
- Configuration is mounted in domain-based groups (`distributed`/`memory`/`quantization`).
- Module names use wildcards uniformly: `{*}` matches a single-layer index (e.g. `model.layers.{*}`), `{0-3}` matches a numeric range, and `*` matches any prefix (e.g. `*.q_proj`).

### Adding Configuration Validation

Implement the `validate_xxx_config` validation method and register it for invocation in `__post_init__`:

```python
# fsdp_turbo/fsdp_turbo_config.py
def validate_chunk_batch_config(self):
    if not self.memory.chunk_batch:
        return
    self.memory.chunk_batch_plan = (
        ChunkBatchPlanConfig() if self.memory.chunk_batch_plan is None else self.memory.chunk_batch_plan
    )
    if self.memory.chunk_batch_plan.chunk_mbs <= 0:
        raise ValueError("chunk_mbs must be positive.")
    if not self.memory.chunk_batch_plan.apply_modules:
        raise ValueError("chunk_batch_plan.apply_modules must not be empty when chunk_batch is enabled.")
    if not self.memory.chunk_batch_plan.chunk_arg_indexes and not self.memory.chunk_batch_plan.chunk_kwarg_names:
        raise ValueError(
            "chunk_batch_plan must specify chunk_arg_indexes or chunk_kwarg_names when chunk_batch is enabled."
        )


# Register it in __post_init__
def __post_init__(self):
    ...
    self.validate_chunk_batch_config()
```

The following notes apply:

- The method returns immediately when the feature is disabled, without affecting other configuration.
- Validation error messages include the full field path (e.g. `chunk_batch_plan.apply_modules`) to help locate YAML issues.
- If a new top-level group is added, its parsing must also be integrated into `load_config_from_yaml`.

### Implementing the Module Transformation Function

Implement the module transformation function in the feature directory, matching the target modules and applying the transformation:

```python
# fsdp_turbo/memory/chunk_batch/chunk_batch.py
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match


def chunk_batch_modules(model, plan: ChunkBatchPlanConfig):
    chunk_mbs_modules = get_chunkmbs_modules(model, plan.apply_modules)
    apply_chunkmbs_module(chunk_mbs_modules, plan)
    return model


def get_chunkmbs_modules(modules, plan):
    matched_modules = []
    for plan_name in plan:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                matched_modules.append((name, module))
    if len(matched_modules) == 0:
        raise RuntimeError(f'[ChunkMBS] No module named {plan}.')
    return matched_modules
```

The following notes apply:

- The transformation function only wraps the module's `forward` and does **not replace the module object itself**, keeping parameters, buffers, and fully qualified module names unchanged to avoid breaking weight loading and FSDP traversal.
- A `RuntimeError` is raised when no module matches, preventing the configuration from silently failing.
- Use `print_rank(logger.info, ...)` to print application logs; output is restricted to the main process.

### Integrating the Feature into FSDPTurbo

Add an integration method in `FSDPTurbo` and call it in `__init__` according to the application order:

```python
# fsdp_turbo/fsdp_turbo.py
from fsdp_turbo.memory.chunk_batch.chunk_batch import chunk_batch_modules


class FSDPTurbo(torch.nn.Module):
    def __init__(self, config: FSDPTurboConfig, model: torch.nn.Module):
        ...
        self.apply_recompute_modules()
        self.apply_chunk_batch_modules()   # Insert before FSDP sharding
        self.apply_fsdp_modules()

    def apply_chunk_batch_modules(self):
        if not self.config.memory.chunk_batch:
            return
        self.model = chunk_batch_modules(self.model, self.config.memory.chunk_batch_plan)
```

### Using the Feature

Enable the feature in the YAML configuration, for example:

```yaml
# examples/xxx/config.yaml
memory:
  chunk_batch: true
  chunk_batch_plan:
    apply_modules: ["model.layers.{*}"]
    chunk_mbs: 1
    batch_dim: 0
    chunk_kwarg_names: ["hidden_states"]
```

## Replacing Third-Party Functions (module_patches)

When a feature needs to replace Hugging Face/PyTorch functions (such as attention implementations, loss computation, or model forward), this can be done through the `module_patches` configuration without modifying the feature code:

```yaml
module_patches:
  - target: transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward
    replacement: fsdp_turbo.models.qwen.qwen3_5.qwen3_5_model_forward
```

The following notes apply:

- `target` supports two forms: `module.member` (module-level functions) and `module.Class.member` (class methods/attributes, effective for all instances of the class).
- `replacement` must be a dotted path (the `module:name` colon syntax is not supported).
- `module_patches` are applied before FSDP2 sharding; a `RuntimeError` is raised when the target does not match any module/namespace member, preventing typos from silently failing.
- When patching is needed inside feature code, use `fsdp_turbo.utils.patch.patch_namespace_members`: installation is idempotent (the same replacement is not installed twice), and already-loaded `from module import member` aliases are refreshed synchronously.

## Multi-Backend Operator Development

Features that involve device-related operators (such as MoE grouped matmul) use the operator registry to support multiple backends (CPU/NPU):

```python
# fsdp_turbo/ops/npu/grouped_matmul.py
from fsdp_turbo.ops.registry import register_op

@register_op('grouped_matmul', 'npu')
def grouped_matmul_npu(inputs, m_split, weights):
    ...  # NPU-accelerated implementation
```

```python
# Call side decoupled from the device
from fsdp_turbo.ops.registry import dispatch_op

output = dispatch_op('grouped_matmul', inputs, m_split, weights)
```

The following notes apply:

- Operators are organized by device under `fsdp_turbo/ops/{cpu,npu,triton}/`, with one implementation per directory for operators with the same name.
- **A CPU implementation is required**: it serves both as the correctness reference implementation (the unit test baseline) and as the fallback when no accelerated backend is available — `get_op` automatically falls back to the CPU implementation and prints a warning once when the target device is not registered.
- Custom autograd operators (such as `AllToAllGroupedMatmul` in `fsdp_turbo/ops/npu/grouped_matmul_mc2.py`) should inherit from `torch.autograd.Function` and then expose a unified dispatch entry point through the registry.

## Development Practice Guide

### Development Practice Recommendations

- Disabled by default: new feature switches default to `False`/`None` and must not affect basic functionality when not enabled.
- Complete validation: validate module name patterns, value ranges, and feature combination constraints in `validate_xxx_config` within `__post_init__`, with field paths included in error messages.
- Fail fast: raise an error immediately when module matching or patch target matching fails; silent skipping is forbidden.
- Keep module instances: transformations only wrap `forward` or modify attributes without replacing module objects, ensuring compatibility with weight loading and FSDP traversal.
- Application order: all transformations that depend on the original module structure must run before `apply_fsdp_modules`; FSDP2 sharding must be executed last.
- Deferred imports: import heavy dependencies (such as NPU extensions) only inside the feature branch, avoiding side effects when the feature is not enabled.

### Checklist for Creating a New Feature

After development, go through the checklist item by item:

```text
Feature Creation Checklist
├── Configuration definition
│   ├── [ ] Add a plan dataclass (switch separated from plan)
│   ├── [ ] Mount to the corresponding group (distributed/memory/quantization)
│   └── [ ] Integrate parsing into load_config_from_yaml (when adding a group)
├── Parameter validation
│   ├── [ ] Implement validate_xxx_config and register it in __post_init__
│   ├── [ ] Validate module name patterns and value ranges, with field paths in error messages
│   └── [ ] Return immediately when the feature is disabled
├── Module transformation
│   ├── [ ] Implement xxx_modules(model, plan)
│   ├── [ ] Raise RuntimeError when no module matches
│   └── [ ] Do not replace module objects; only wrap forward/modify attributes
├── Application integration
│   ├── [ ] Add apply_xxx_modules in FSDPTurbo; return directly when disabled
│   └── [ ] Confirm the insertion position is before FSDP2 sharding
├── Device operators (if involved)
│   ├── [ ] Register the CPU reference implementation
│   └── [ ] Register the NPU-accelerated implementation
└── Testing and validation
    ├── [ ] Unit tests for config parsing and validation
    ├── [ ] Unit tests for module matching/transformation (CPU implementation as the baseline)
    └── [ ] Feature on/off comparison: loss consistency + memory/performance gains
```

### Common Issues

- **How do I troubleshoot a feature that is not taking effect?**

  1. Check that the YAML keys are spelled correctly — unknown configuration items are ignored and an `Ignoring unknown ... config key` warning is printed.
  2. Look for the `Applying ... to module: xxx` output in the logs (printed by the main process only).
  3. Confirm that the feature switch field is `true` and that the module name patterns in the plan match the `model.named_modules()` output.

- **What if no module name matches?**

  Print `model.named_modules()` to view the actual module paths; check the wildcard syntax: `{*}` matches a single-layer index (`model.layers.{*}`), `{0-3}` matches a numeric range, and `*` matches any prefix (`*.q_proj`).

- **How do I choose between module_patches and patching inside feature code?**

  | Scenario | Recommended Approach |
  |---|---|
  | Users replacing third-party functions per model (attention/loss/forward) | YAML `module_patches` |
  | Functions that the feature's own logic must replace | `patch_namespace_members` in the feature code |

- **Are patches idempotent?**

  Yes. `fsdp_turbo.utils.patch` records installed replacements via `_fsdp_turbo_patch_ids`, so repeated installation of the same replacement is skipped; module-level replacements also synchronously refresh loaded `from module import member` aliases, ensuring that `from ... import` usage also sees the patch.

- **How do I choose the patch target form?**

  - `module.member`: module-level functions, e.g. `transformers.loss.loss_utils.ForCausalLMLoss`.
  - `module.Class.member`: class methods or attributes, effective for all instances of the class, e.g. `transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward`.

# Expert Parallelism

FSDPTurbo provides expert parallelism (EP) for MoE (Mixture of Experts) models, sharding the experts across multiple devices along the expert dimension. Tokens are dispatched through All-to-All communication to the devices holding the corresponding experts for computation.

## Configuration

Configure the expert parallel degree and the expert plan in the `distributed` section:

```yaml
distributed:
  expert_parallel_size: 8
  expert_fully_shard_parallel_size: 1
  ep_plan:
    apply_modules:
      - model.language_model.layers.{*}.mlp.experts
    dispatcher: "fused"
    apply_efsdp_modules:
      - model.language_model.layers.{*}.mlp.experts
```

## Parameter Descriptions

| Parameter | Description |
|---|---|
| `expert_parallel_size` | Expert parallel degree; shards the experts module along the expert dimension across the EP group |
| `expert_fully_shard_parallel_size` | Expert FSDP2 (EFSDP) parallel degree; further fully shards the expert parameters |
| `ep_plan.apply_modules` | List of modules to which EP is applied, supporting the `model.language_model.layers.{*}.mlp.experts` wildcard |
| `ep_plan.dispatcher` | Dispatcher for expert computation; see the table below |
| `ep_plan.fixed_router` | Whether to use a fixed router |
| `ep_plan.apply_efsdp_modules` | List of modules to which EFSDP is applied |
| `ep_plan.gradient_divide_factor` | Gradient divide factor for the experts (must be specified manually under special parallelism combinations) |

## Dispatcher Types

| dispatcher | Description |
|---|---|
| `eager` | Per-expert dispatch; has the best generality |
| `fused` | Grouped GEMM (grouped_matmul); computes in batches after tokens are aggregated, with better performance |
| `mc2` | Communication-computation fusion implementation accelerated by Ascend MC2 |
| `domino` | Domino dispatch implementation |
| `custom_ep_forward_eager` | Custom EP forward (eager semantics); the model must implement the `ep_forward_eager` interface |
| `custom_ep_forward_fused` | Custom EP forward (fused semantics); the model must implement the `ep_forward_fused` interface |

## Custom EP Forward

When a model's expert computation semantics are non-standard, a custom dispatcher can be used, implemented on the experts module:

```python
def ep_forward_eager(self, hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router):
    ...

def ep_forward_fused(self, hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router):
    ...
```

MoE models with the standard GLU/SwiGLU structure should prefer the built-in `eager`/`fused`/`mc2`/`domino` dispatchers.

## Model Structure Requirements

The EP path of FSDPTurbo requires the experts module to hold **3D tensor parameters**, rather than independent per-expert submodules in ModuleList form. Specific requirements:

- The experts module must expose two 3D tensor parameters, `gate_up_proj` and `down_proj`, with the shape `[num_experts, out_dim, in_dim]`.
- Parameters in the 2D stacked format `[num_experts * out_dim, in_dim]` must be reshaped to 3D via `view(num_experts, -1, in_dim)` before they can be used in grouped_matmul.
- **ModuleList-form experts are not supported**: per-expert Linear submodules cannot directly participate in EP's All-to-All dispatch and grouped GEMM; they must first be converted into a 3D tensor aggregated structure.

## Notes

- **Parallelism combination constraint**: `expert_parallel_size` must evenly divide the total number of experts per layer, and `expert_parallel_size × expert_fully_shard_parallel_size ≤ total number of devices`.
- **Expert load balancing**: during training, the routing distribution (the frequency with which experts are selected by top-k) can be observed. If severe imbalance occurs, consider adjusting the model-side routing temperature, or use `fixed_router` (a debugging measure: replaces the actual router with a fixed uniform router to avoid the performance and precision issues caused by imbalanced load; it only takes effect when `expert_parallel_size > 1`).
- **EFSDP**: when the expert parameter count is large, enable `apply_efsdp_modules` to further fully shard the expert parameters with FSDP2, reducing the per-card memory usage.

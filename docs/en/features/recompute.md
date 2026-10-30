# Activation Recomputation

Activation recomputation (gradient checkpointing) does not save the activations of certain modules during the forward pass but recomputes them during the backward pass, trading time for memory.

## Configuration

Enable recomputation in the `memory` section and specify the target modules:

```yaml
memory:
  recompute: true
  recompute_plan:
    - model.language_model.layers.{*}
    - model.visual.blocks.{*}
```

## Parameter Descriptions

| Parameter | Description |
|---|---|
| `recompute` | Master switch for activation recomputation |
| `recompute_plan` | List of modules to which recomputation applies, supporting `model.layers.{*}` hierarchical wildcards. It can be enabled only on selected layers to balance memory usage and time cost |

## Implementation Mechanism

FSDPTurbo's recomputation is implemented based on `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`, and `context_fn` is used to control the random number state consistency within the recomputation region. The recomputation wrapper automatically handles the following scenarios:

- **KV Cache handling**: If a module's forward accepts `past_key_values`/`past_key_value` arguments, they are set to `None` and `use_cache` is set to `False` before recomputation, avoiding dtype mismatches or CheckpointError caused by the KV cache in the recomputation path.
- **Recursion protection**: The two flags `is_grad_enabled` and `_in_checkpoint_forward` prevent infinite recursion during recomputation.

## Notes

- **Combining with context parallelism (CP)**: Recomputation and Ulysses CP can be enabled at the same time. Recomputation applies to the local rank's sequence segment after CP sharding, and the two do not conflict.
- **Combining with CPU offload**: Recomputation reduces activation memory, while CPU Offload reduces parameter/gradient memory. Combining the two can further compress memory peaks but increases computation and data transfer overhead.
- **Granularity selection**: It is recommended to use Transformer layers (`model.layers.{*}`) as the recomputation unit. Enabling it layer by layer strikes a balance between memory usage and time cost. Enabling it everywhere saves the most memory but adds the most time.

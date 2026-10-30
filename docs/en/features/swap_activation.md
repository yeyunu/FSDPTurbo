# Activation Swapping

Activation swapping asynchronously offloads activations to CPU memory during computation and transfers them back during the backward pass, trading PCIe bandwidth for device memory.

## How It Works

FSDPTurbo provides `TensorSwapManager` to manage the asynchronous transfer of activations. After swapping is applied to a specified module, its forward is wrapped:

1. After the forward computation completes, the activations are asynchronously offloaded from the device to the host (pinned CPU memory) via `SwapTensor`.
2. When the backward computation needs an activation, it is asynchronously transferred back from the host to the device.
3. Multi-stream asynchronous execution overlaps data transfer with computation, reducing memory peaks.

## Configuration

Enable it in the `memory` section of `config.yaml`; the module matching rules are the same as those for recomputation (`recompute_plan`):

```yaml
memory:
  swap_activation: true
  swap_activation_plan:
    apply_modules:
      - model.layers.{*}
```

| Parameter | Description |
|---|---|
| `swap_activation` | Master switch for activation swapping |
| `swap_activation_plan.apply_modules` | List of modules to which swapping applies, supporting `model.layers.{*}` hierarchical wildcards. The list must not be empty when swapping is enabled; otherwise, an error is raised at startup |

Once configured, it is applied automatically by the `FSDPTurbo` initialization flow, with no additional code required.

## Applicable Scenarios

- Long-sequence training (activations account for a high proportion of memory usage)
- Large batch size scenarios
- Scenarios with tight memory that need to further compress activation memory

## Notes

- Swapping introduces PCIe data transfer overhead; ensure that computation and transfer overlap effectively; otherwise, throughput may decrease.
- It is recommended to enable it only on modules with high activation memory usage (e.g., decoder layers).
- It can be combined with other memory optimization techniques such as recomputation and CPU Offload.

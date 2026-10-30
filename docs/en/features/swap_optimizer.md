# Optimizer State Swapping

Optimizer state swapping offloads optimizer states to CPU memory and transfers them back in batches according to the available memory at update time, reducing Ascend NPU memory usage.

## How It Works

The states of optimizers such as AdamW (first-order momentum `exp_avg`, second-order momentum `exp_avg_sq`) usually occupy as much memory as the parameters, or even more. Optimizer state swapping is implemented through the `AdamWSwap` optimizer:

1. At initialization, the optimizer states are copied from the device to pinned CPU memory.
2. When `step()` is executed, the states are transferred back to the device in batches according to the currently available memory, and are swapped back to the host after the parameter updates are computed.
3. By transferring states back in batches, the peak memory is reduced from "all states" to "a single batch of states".

## Configuration

Set the optimizer type to `AdamWSwap` in the `optimizer` section of `config.yaml`:

```yaml
optimizer:
  optimizer_type: "AdamWSwap"
  lr: 3.0e-4
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1.0e-8
  mem_fraction_static: 0.9                    # Optional: fraction of free memory used for batched H2D transfer-back, default 0.9
  state_keys: ["exp_avg", "exp_avg_sq"]       # Optional: state keys involved in swapping; defaults to the internal AdamWSwap defaults when omitted
  param_prefetch: false                        # Optional: whether to preload one parameter involved in swapping to reduce bandwidth blocking; defaults to the internal AdamWSwap default (disabled)
  clip_grad_keep_on_device: false              # Optional: when cpu_offload is enabled, keeps gradients on the device side after clip_grad updates them, saving one H2D and one D2H transfer; takes effect only when both cpu_offload and AdamWSwap are enabled; defaults to the internal AdamWSwap default (disabled)
```

| Parameter | Description |
|---|---|
| `optimizer_type` | Setting it to `AdamWSwap` enables optimizer state swapping |
| `mem_fraction_static` | Fraction of free memory allowed for batched transfer-back, default `0.9` |
| `state_keys` | List of optimizer state keys involved in swapping; when omitted, the internal AdamWSwap defaults are used (for AdamW, `exp_avg` and `exp_avg_sq`) |
| `param_prefetch` | Configuration for preloading the parameters involved in swapping; when omitted, the internal AdamWSwap default is used (no preloading) |
| `clip_grad_keep_on_device` | When both cpu_offload and AdamWSwap are enabled, keeps gradients on the device side after clip_grad updates them; when omitted, the internal AdamWSwap default is used (gradients on the host side) |

`AdamWSwap` is registered with the optimizer factory (OptimizerFactory), and the other optimizer arguments (lr, betas, weight_decay, etc.) are the same as those of standard AdamW. See the example: `examples/deepseekv4/config.yaml`.

## Differences from FSDP2 CPU Offload

| Feature | Optimizer State Swapping | FSDP2 `cpu_offload` |
|---|---|---|
| Offload target | Optimizer states only | Parameters + gradients + optimizer states |
| Offload granularity | Transfer back in batches per batch | Offload/transfer back as a whole |
| Communication overhead | Lower (states only) | Higher (parameters + gradients) |
| Applicable scenarios | Optimizer states dominate memory usage | Overall memory is tight |

The two can be combined, but note the additional data transfer overhead when combined.

## Applicable Scenarios

- Optimizer states account for a high proportion of memory usage (large models + AdamW)
- Parameters are already sharded via FSDP2, but optimizer states still occupy a large amount of memory

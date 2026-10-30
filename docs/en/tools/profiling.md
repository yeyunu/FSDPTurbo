# Performance Data Profiling

FSDPTurbo has built-in performance data profiling based on PyTorch Profiler, which can be enabled through YAML configuration.

## Configuration

Configure the performance data profiling parameters in the `run.profile` section:

```yaml
run:
  profile:
    enabled: true
    wait_steps: 0
    warmup_steps: 0
    active_steps: 10
    repeat: 1
    skip_first: 0
    output_dir: "./profile"
    record_shapes: false
    profile_memory: false
    with_stack: false
```

## Parameter Description

| Parameter | Description |
|---|---|
| `enabled` | Profiling switch |
| `wait_steps` / `warmup_steps` / `active_steps` / `repeat` | Profiler scheduling: number of wait/warmup/active steps and number of repeats |
| `skip_first` | Number of initial steps to skip |
| `output_dir` | Output directory for profiling data |
| `record_shapes` | Whether to record operator input shapes |
| `profile_memory` | Whether to record memory allocations |
| `with_stack` | Whether to record call stacks |

## Usage Recommendations

- Complete the `warmup_steps` warmup before profiling, to avoid interference from the compilation/initialization phase.
- Enable `profile_memory` when analyzing memory issues; enable `record_shapes` when analyzing operator time hotspots.
- The profiling results can be viewed with `tensorboard --logdir ./profile` or Chrome tracing tools.

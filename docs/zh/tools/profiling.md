# 性能数据采集

FSDPTurbo内置基于PyTorch Profiler的性能数据采集能力，通过YAML配置即可开启。

## 配置方法

在`run.profile`段配置性能数据采集参数：

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

## 参数说明

| 参数 | 说明 |
|---|---|
| `enabled` | Profiling开关 |
| `wait_steps` / `warmup_steps` / `active_steps` / `repeat` | Profiler调度：等待/预热/采集步数与重复轮数 |
| `skip_first` | 跳过起始步数 |
| `output_dir` | Profiling数据输出目录 |
| `record_shapes` | 是否记录算子输入形状 |
| `profile_memory` | 是否记录显存分配 |
| `with_stack` | 是否记录调用栈 |

## 使用建议

- 先完成`warmup_steps`预热再采集，避免编译/初始化阶段干扰。
- 分析显存问题时开启`profile_memory`；分析算子耗时热点时开启`record_shapes`。
- 采集结果可使用`tensorboard --logdir ./profile`或Chrome tracing工具查看。

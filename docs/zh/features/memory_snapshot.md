# 显存快照

显存快照（Memory Snapshot）采集设备分配器（NPU 的 `torch.npu.memory` / CUDA 的 `torch.cuda.memory`）的分配历史与快照 pickle，用于显存占用分析与 OOM 定位。离线分析可用 `tools/analyze_memory_snapshot.py` 查看每个设备的活动块、最大分配栈等。

该能力位于 trainer 层，**适用于所有模型**（框架内各 example 均走同一套训练循环钩子，配置写法一致）；本页的示例块也已提供给各模型的 example 配置。

## 配置方法

在`config.yaml`的`run`段配置：

```yaml
run:
  memory_snapshot:
    enable: false
    start_step: 1
    end_step: 2
    save_path: ./memory_snapshot
    dump_ranks: [0]
    stacks: all
    max_entries: null
    mem_info: false
```

| 参数 | 说明 |
|---|---|
| `enable` | 快照采集总开关。`true`时从`start_step`开始记录分配器历史，`end_step`步结束落盘一份快照后停止；`end_step`缺省时从`start_step`起每步落盘一份 |
| `start_step` | 采集窗口起点（0 起计的微步序号，每个 dataloader batch 记一步；`gradient_accumulation_steps=1`时即优化步） |
| `end_step` | 采集窗口终点（含）。给定该值时整个窗口只落一份快照（在`end_step`步结束时），落盘后停止记录；`null`表示不设终点、逐步落盘 |
| `save_path` | 快照落盘目录；`null`时默认`<checkpoint.output_dir>/memory_snapshot`。相对路径按启动目录解析 |
| `dump_ranks` | 记录与落盘的全局 rank 列表；`null`缺省时全部 rank。显式空列表表示不落盘。未选中的 rank 无任何采集开销 |
| `stacks` | 分配栈记录范围：`all`或`python` |
| `max_entries` | 分配器历史条数上限；`null`表示不限。长跑建议配合采集窗口或显式调小 |
| `mem_info` | 每步打印一行显存统计（当前/本步峰值/累计峰值 allocated 与 reserved，单位 GiB；每步开始重置峰值统计），独立于`enable`，可用于只做统计不开快照 |

例如`start_step: 1, end_step: 2`：记录第 1~2 步的分配器历史，在第 2 步结束于`dump_ranks`指定的 rank 上落盘**一份**快照后停止记录。

## 落盘内容

- 快照文件命名：`rank<r>_{step|error}_micro<m>_global<g>_<时间>.pkl`（时间格式`YYYY-MM-DD-HH-MM`）。
- YAML 采集**只覆盖正常运行**：运行中抛异常（含 OOM）时不落盘、不归置原生文件、不停止记录；异常/OOM 现场由下方的 OOM 陷阱负责。
- 部分 torch_npu 版本的`_dump_snapshot`会忽略传入路径、把`device_<id>_memusage*`系列文件（pickle 快照及`device_<id>_memUsageINFO*.csv`统计伴生文件）写到启动目录——框架会在每次落盘后按`LOCAL_RANK`匹配把它们统一搬运到`save_path`。

## 分析与验证

分析快照 pickle：

```bash
python tools/analyze_memory_snapshot.py [--top 15] snapshot.pkl [more.pkl ...]
python tools/analyze_memory_snapshot.py 'memory_snapshot/rank*_error_*.pkl'
```

输出每个设备的段/活动字节汇总、OOM 事件（含失败分配的调用栈）以及按分配栈归组的活动块排行。

## OOM 陷阱（环境变量控制）

除按窗口采集外，还提供"静默记录、只在真 OOM 时落档"的陷阱模式，通过环境变量控制（不在 YAML 配置内）：

| 环境变量 | 取值 | 说明 |
|---|---|---|
| `OOM_SNAPSHOT_ENABLE` | `0`（默认）、`1` | OOM 快照陷阱：静默开启分配器历史记录（无每步日志、无每步落盘），仅当某步抛出`torch.OutOfMemoryError`时落一份带分配栈的快照并停止记录。与 YAML 采集共用窗口、`dump_ranks`、`stacks`、`max_entries`，可叠加（每步统计行 + OOM 陷阱）。历史记录本身有轻微开销，仅在排查 OOM 时开启 |
| `OOM_SNAPSHOT_PATH` | 路径，默认不设 | 原生 OOM 快照（`oom_snapshot_<pid>_<时间戳>.pickle`，分配器在抛 OOM 瞬间自动写出、内容含完整分配栈）的保存目录。torch_npu 新版本会原生读取本变量直接写往该路径；对仍写往当前工作目录的旧版本运行时，框架的 OOM 陷阱会在异常处理时把本进程的文件搬运到该目录并加`rank<r>_native_`前缀（优先级高于`save_path`；两者都不设时默认`<checkpoint.output_dir>/memory_snapshot/`）。框架自己的`rank<r>_error_*.pkl`不受本变量影响 |

```bash
# 平时零感知，只在真 OOM 时自动留一份带分配栈的现场
OOM_SNAPSHOT_ENABLE=1 bash run.sh
```

OOM 现场**完全由该陷阱负责**：YAML 采集（`memory_snapshot.enable`）只覆盖正常运行，即使发生 OOM 也不会额外落盘或归置原生文件——需要在 OOM 时留下现场，请开启 `OOM_SNAPSHOT_ENABLE`（可配合 `OOM_SNAPSHOT_PATH`）。只开 YAML 时，分配器在 OOM 瞬间写出的原生快照文件可能残留在启动目录（新版本 torch_npu 会写往 `OOM_SNAPSHOT_PATH`），不会被自动搬运。

## 注意事项

- 快照采集（历史记录与逐步落盘）有额外开销，仅调试显存时开启；`mem_info`的逐步统计开销较小。
- 采集是显式开启的能力：在无法采集的后端上（如 CPU-only、或缺少显存统计/落盘接口的环境），训练启动时会**直接报错（fail fast）**而不是静默降级——请在这些环境关闭`memory_snapshot.enable`并去掉`OOM_SNAPSHOT_ENABLE`。
- 快照目录会被所有采集 rank 共享，多 rank 各写各的文件，建议指向共享盘或节点本地盘按需选择。
- OOM 常发生在个别 rank 上，若该 rank 不在`dump_ranks`内则不会留下快照；不确定时保持默认全采。

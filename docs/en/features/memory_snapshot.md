# Memory Snapshot

Memory snapshot collects the allocation history and the snapshot pickle of the device allocator (`torch.npu.memory` for NPU/`torch.cuda.memory` for CUDA). It is used for memory usage analysis and OOM localization. For offline analysis, use `tools/analyze_memory_snapshot.py` to inspect the active blocks, the largest allocation stacks, and more for each device.

This capability resides in the trainer layer and **applies to all models** (all examples in the framework go through the same set of training loop hooks, so the configuration is written in the same way); the example block on this page has also been provided to the example configuration of each model.

## Configuration

Configure it in the `run` section of `config.yaml`:

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

| Parameter | Description |
|---|---|
| `enable` | Master switch for snapshot collection. When `true`, the allocator history is recorded starting from `start_step`, and recording stops after one snapshot is dumped at the end of the `end_step` step; when `end_step` is not specified, one snapshot is dumped per step starting from `start_step` |
| `start_step` | Start of the collection window (a 0-based micro-step index; each dataloader batch counts as one step; with `gradient_accumulation_steps=1` it is the optimizer step) |
| `end_step` | End of the collection window (inclusive). When set, only one snapshot is dumped for the whole window (at the end of the `end_step` step) and recording stops after the dump; `null` means no end point and one snapshot is dumped per step |
| `save_path` | Directory to which snapshots are dumped; when `null`, defaults to `<checkpoint.output_dir>/memory_snapshot`. Relative paths are resolved against the launch directory |
| `dump_ranks` | List of global ranks to record and dump; when `null` (default), all ranks. An explicit empty list means no dump. Ranks not selected incur no collection overhead |
| `stacks` | Scope of allocation stack recording: `all` or `python` |
| `max_entries` | Upper limit on the number of allocator history entries; `null` means unlimited. For long runs, it is recommended to combine it with a collection window or set it to a smaller explicit value |
| `mem_info` | Print one line of memory statistics per step (current/this-step peak/cumulative peak allocated and reserved, in GiB; the peak statistics are reset at the beginning of each step). It is independent of `enable` and can be used for statistics only without enabling snapshots |

For example, with `start_step: 1, end_step: 2`: the allocator history of steps 1~2 is recorded, and after **one** snapshot is dumped on the ranks specified by `dump_ranks` at the end of step 2, recording stops.

## Dumped Content

- Snapshot file naming: `rank<r>_{step|error}_micro<m>_global<g>_<time>.pkl` (time format `YYYY-MM-DD-HH-MM`).
- YAML collection **covers normal execution only**: when an exception is thrown during the run (including OOM), no snapshot is dumped, native files are not relocated, and recording is not stopped; exception/OOM scenes are handled by the OOM trap described below.
- In some torch_npu versions, `_dump_snapshot` ignores the passed-in path and writes the `device_<id>_memusage*` family of files (the pickle snapshot and the accompanying `device_<id>_memUsageINFO*.csv` statistics files) to the launch directory — after each dump, the framework matches them by `LOCAL_RANK` and moves them all to `save_path`.

## Analysis and Validation

To analyze a snapshot pickle:

```bash
python tools/analyze_memory_snapshot.py [--top 15] snapshot.pkl [more.pkl ...]
python tools/analyze_memory_snapshot.py 'memory_snapshot/rank*_error_*.pkl'
```

It outputs the segment/active byte summary of each device, the OOM events (including the call stack of the failed allocation), and a ranking of active blocks grouped by allocation stack.

## OOM Trap (Controlled by Environment Variables)

In addition to window-based collection, a trap mode that "records silently and dumps a file only on a real OOM" is also provided, controlled by environment variables (not part of the YAML configuration):

| Environment Variable | Value | Description |
|---|---|---|
| `OOM_SNAPSHOT_ENABLE` | `0` (default), `1` | OOM snapshot trap: silently enables allocator history recording (no per-step logs, no per-step dumps); a snapshot with allocation stacks is dumped and recording stops only when a step throws `torch.OutOfMemoryError`. It shares the window, `dump_ranks`, `stacks`, and `max_entries` with YAML collection, and the two can be combined (per-step statistics line + OOM trap). History recording itself has a slight overhead, so enable it only when troubleshooting OOM |
| `OOM_SNAPSHOT_PATH` | Path, unset by default | Directory for saving native OOM snapshots (`oom_snapshot_<pid>_<timestamp>.pickle`, written out automatically by the allocator at the moment OOM is thrown, with content containing the full allocation stack). New torch_npu versions read this variable natively and write directly to that path; for older runtime versions that still write to the current working directory, the framework's OOM trap moves this process's files to that directory during exception handling and adds the `rank<r>_native_` prefix (this takes precedence over `save_path`; when neither is set, the default is `<checkpoint.output_dir>/memory_snapshot/`). The framework's own `rank<r>_error_*.pkl` files are not affected by this variable |

```bash
# Zero awareness in normal operation; a scene with allocation stacks is automatically preserved only on a real OOM
OOM_SNAPSHOT_ENABLE=1 bash run.sh
```

OOM scenes are **handled entirely by this trap**: YAML collection (`memory_snapshot.enable`) covers normal execution only; even if an OOM occurs, it does not dump anything extra or relocate native files — to preserve a scene on OOM, enable `OOM_SNAPSHOT_ENABLE` (optionally together with `OOM_SNAPSHOT_PATH`). When only YAML collection is enabled, the native snapshot files written by the allocator at the moment of OOM may remain in the launch directory (new torch_npu versions write to `OOM_SNAPSHOT_PATH`) and will not be moved automatically.

## Notes

- Snapshot collection (history recording and per-step dumping) has extra overhead; enable it only when debugging memory. The per-step statistics of `mem_info` have a relatively small overhead.
- Collection is an explicitly enabled capability: on backends where collection is not possible (such as CPU-only, or environments lacking memory statistics/dump interfaces), training **raises an error directly (fail fast)** at startup instead of silently degrading — in these environments, disable `memory_snapshot.enable` and remove `OOM_SNAPSHOT_ENABLE`.
- The snapshot directory is shared by all collecting ranks, and each rank writes its own files; it is recommended to point it to a shared disk or a node-local disk as needed.
- OOM often occurs on individual ranks; if such a rank is not included in `dump_ranks`, no snapshot is left. When unsure, keep the default of collecting on all ranks.

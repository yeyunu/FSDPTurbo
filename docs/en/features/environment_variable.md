# Environment Variables

This document describes the environment variables involved in FSDPTurbo training, including variables related to cluster configuration, the runtime environment, and NPU performance tuning.

## Cluster Configuration Environment Variables

The training launch script `run.sh` supports overriding the cluster configuration through environment variables:

| Environment Variable | Default Value | Description |
|---|---|---|
| `NNODES` | `1` | Number of nodes; 1 for a single machine |
| `GPUS_PER_NODE` | `16` | Number of devices per node |
| `MASTER_ADDR` | `127.0.0.1` | IP address of the master node; in multi-node setups, all nodes are configured with the master node IP |
| `MASTER_PORT` | `29500` | Port of the master node |
| `CONFIG_FILE` | `config.yaml` | Path to the training configuration file |

Multi-node training example:

```bash
# Run on all nodes (MASTER_ADDR is the master node IP)
NNODES=2 GPUS_PER_NODE=16 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 bash run.sh
```

## Runtime Environment Variables

| Environment Variable | Description |
|---|---|
| `HCCL_NPU_SOCKET_SOCKET_PORT_RANGE`/`HCCL_NPU_SOCKET_PORT_RANGE` | Port range of the NPU communication socket (set to `auto` in the example script) |
| `ASCEND_LAUNCH_BLOCKING` | Switch for synchronous NPU execution (set to `1` in the example script for easier debugging) |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | Output NPU logs to standard output (for debugging; disabled by default) |

Kimi routing diagnostics now live on `ep_plan.router_debug` (`apply_custom_router`, `target_max_vio`, `router_mode`, `record_load_balance`) and are off by default. See [Expert Parallel](./expert_parallel.md).

> [!NOTE]
>
> In an NPU environment, the Ascend environment must be loaded correctly before training:
>
> ```bash
> source /usr/local/Ascend/cann/set_env.sh
> ```

## NPU Performance Tuning Environment Variables

The following environment variables are used for Ascend NPU training performance tuning and can be set in the launch script as appropriate:

| Environment Variable | Recommended Value | Description |
|---|---|---|
| `HCCL_CONNECT_TIMEOUT` | `1800` | HCCL communication connection timeout in seconds; increase it for large multi-node clusters |
| `TASK_QUEUE_ENABLE` | `2` | Task queue mode, improving the overlap efficiency of communication and computation |
| `CPU_AFFINITY_CONF` | `1` | CPU affinity configuration, reducing process scheduling overhead |
| `MULTI_STREAM_MEMORY_REUSE` | `2` | Multi-stream memory reuse, reducing memory usage |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | NPU memory allocator configuration, reducing memory fragmentation |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Workspace configuration for matrix operations |

Environment variable configuration in the example `run.sh`:

```bash
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export ASCEND_LAUNCH_BLOCKING=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=1   # Enable during debugging
```

> [!NOTE]
>
> The NPU performance tuning variables above are recommended configurations; the actual effects vary with the model scale, hardware model, and parallelism strategy. It is recommended to tune them in conjunction with [performance data profiling](../tools/profiling.md).

## MoonEP Debug

`trace_enabled`, `trace_sync_enabled`, `vmm_probe_enabled`, and `npu_async_comm` now live on `ep_plan.moonep_config` and are off by default. See [Expert Parallel](./expert_parallel.md).

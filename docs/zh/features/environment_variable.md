# 环境变量说明

本文档介绍FSDPTurbo训练过程中涉及的环境变量，包括集群配置、运行环境与NPU性能调优相关变量。

## 集群配置环境变量

训练启动脚本`run.sh`支持通过环境变量覆盖集群配置：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `NNODES` | `1` | 节点数，单机为1 |
| `GPUS_PER_NODE` | `16` | 单节点卡数 |
| `MASTER_ADDR` | `127.0.0.1` | 主节点IP，多机时所有节点配置为主节点IP |
| `MASTER_PORT` | `29500` | 主节点端口 |
| `CONFIG_FILE` | `config.yaml` | 训练配置文件路径 |

多机训练示例：

```bash
# 所有节点执行（MASTER_ADDR 为主节点 IP）
NNODES=2 GPUS_PER_NODE=16 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 bash run.sh
```

## 运行环境变量

| 环境变量 | 说明 |
|---|---|
| `HCCL_NPU_SOCKET_SOCKET_PORT_RANGE` / `HCCL_NPU_SOCKET_PORT_RANGE` | NPU通信socket端口范围（示例脚本中设置为`auto`） |
| `ASCEND_LAUNCH_BLOCKING` | NPU同步执行开关（示例脚本中设置为`1`，便于调试定位） |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | NPU日志输出到标准输出（调试用，默认关闭） |

> [!NOTE]
> 
> NPU环境下，训练前需正确加载昇腾环境：
>
> ```bash
> source /usr/local/Ascend/cann/set_env.sh
> ```

## NPU性能调优环境变量

以下环境变量用于昇腾NPU训练性能调优，可根据实际情况在启动脚本中设置：

| 环境变量 | 推荐值 | 说明 |
|---|---|---|
| `HCCL_CONNECT_TIMEOUT` | `1800` | HCCL通信连接超时时间（秒），多机大集群时建议增大 |
| `TASK_QUEUE_ENABLE` | `2` | 任务队列模式，提升通信与计算重叠效率 |
| `CPU_AFFINITY_CONF` | `1` | CPU亲和性配置，减少进程调度开销 |
| `MULTI_STREAM_MEMORY_REUSE` | `2` | 多流内存复用，降低显存占用 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | NPU显存分配器配置，减少显存碎片 |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | 矩阵运算workspace配置 |

示例`run.sh`中的环境变量配置：

```bash
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export ASCEND_LAUNCH_BLOCKING=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=1   # 调试时开启
```

> [!NOTE]
>
> 以上NPU性能调优变量为推荐配置，实际效果因模型规模、硬件型号与并行策略而异，建议结合 [性能数据采集](../tools/profiling.md) 进行调优。

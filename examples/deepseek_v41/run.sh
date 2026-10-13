#!/bin/bash
# ============================================================
# DeepSeek-V4.1-Flash Distributed Training Launch Script
# ============================================================

set -e

export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export TASK_QUEUE_ENABLE=2
# export ASCEND_LAUNCH_BLOCKING=1
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

export CPU_AFFINITY_CONF=1,npu0:144-155,npu1:156-167,npu2:48-59,npu3:60-71,npu4:72-83,npu5:84-95,npu6:168-179,npu7:180-191 # 116 124
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRIPT_DIR"

# --- Cluster Configuration ---
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# --- Training Arguments ---
CONFIG_FILE=${CONFIG_FILE:-"config_A5.yaml"}

echo "============================================================"
echo "DeepSeek-V4.1-Flash Multimodal Training"
echo "  NNODES: $NNODES"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  CONFIG: $CONFIG_FILE"
echo "============================================================"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config "$CONFIG_FILE" "$@" | tee dsv41_8p.log

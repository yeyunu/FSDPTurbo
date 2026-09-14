#!/bin/bash
# ============================================================
# Qwen3-30B-A3B Distributed Training Launch Script
# ============================================================

set -euo pipefail

export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export ASCEND_LAUNCH_BLOCKING=1
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Cluster Configuration ---
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}

# --- Training Arguments ---
CONFIG_FILE=${CONFIG_FILE:-"config.yaml"}

echo "============================================================"
echo "Qwen3-30B-A3B Training"
echo "  NNODES: $NNODES"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NODE_RANK: $NODE_RANK"
echo "  CONFIG: $CONFIG_FILE"
echo "============================================================"

mkdir -p logs

LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config "$CONFIG_FILE"  2>&1 | tee "$LOG_FILE"

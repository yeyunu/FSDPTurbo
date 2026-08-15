#!/bin/bash
# ============================================================
# Qwen3.5 moe Distributed Training Launch Script
# ============================================================

set -e

export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export ASCEND_LAUNCH_BLOCKING=1
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Cluster Configuration ---
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-16}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# --- Training Arguments ---
CONFIG_FILE=${CONFIG_FILE:-"config.yaml"}
EXTRA_ARGS=()
if [[ -n "${NUM_HIDDEN_LAYERS:-}" ]]; then
    EXTRA_ARGS+=(--num-hidden-layers "$NUM_HIDDEN_LAYERS")
fi

echo "============================================================"
echo "Qwen3.5 moe Training"
echo "  NNODES: $NNODES"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  CONFIG: $CONFIG_FILE"
echo "  NUM_HIDDEN_LAYERS: ${NUM_HIDDEN_LAYERS:-full}"
echo "============================================================"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config "$CONFIG_FILE" \
    "${EXTRA_ARGS[@]}"

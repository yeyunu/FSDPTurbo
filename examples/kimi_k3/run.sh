#!/bin/bash
# ============================================================
# Qwen3.5 moe Distributed Training Launch Script
# ============================================================

set -e

CANN_DIR=/usr/local/Ascend
source $CANN_DIR/ascend-toolkit/set_env.sh
source $CANN_DIR/nnal/atb/set_env.sh

export HCCL_NPU_SOCKET_PORT_RANGE="auto"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Cluster Configuration ---
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-16}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}

# --- Training Arguments ---
CONFIG_FILE=${CONFIG_FILE:-"config.yaml"}
EXTRA_ARGS=()
if [[ -n "${NUM_HIDDEN_LAYERS:-}" ]]; then
    EXTRA_ARGS+=(--num-hidden-layers "$NUM_HIDDEN_LAYERS")
fi

echo "============================================================"
echo "KimiK3 moe Training"
echo "  NNODES: $NNODES"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NODE_RANK: $NODE_RANK"
echo "  CONFIG: $CONFIG_FILE"
echo "  NUM_HIDDEN_LAYERS: ${NUM_HIDDEN_LAYERS:-full}"
echo "============================================================"

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config "$CONFIG_FILE" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee $LOG_PATH

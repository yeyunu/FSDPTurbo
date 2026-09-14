#!/bin/bash
# ============================================================
# Qwen3.5 moe Distributed Training Launch Script
# ============================================================
# dependency: GPU torch==2.10.0, torchvision==0.25.0, transformers==5.15.0
# dependency: NPU torch==2.10.0, torch_npu==2.10.0, torchvision==0.25.0, transformers==5.15.0

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
NODE_RANK=${NODE_RANK:-0}

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
    "${EXTRA_ARGS[@]}"

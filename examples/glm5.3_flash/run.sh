#!/bin/bash
# ============================================================
# GLM5.3 flash Distributed Training Launch Script
# ============================================================

set -e

CANN_DIR=/usr/local/Ascend
source $CANN_DIR/ascend-toolkit/set_env.sh
source $CANN_DIR/nnal/atb/set_env.sh

export HCCL_NPU_SOCKET_PORT_RANGE="auto"

# Allocator: the indexer's chunked pipeline allocates/frees variable-size
# temporaries every layer, which fragments the NPU caching allocator badly
# (a 512k run died with 6 GiB reserved-but-unusable unable to serve a
# 2 GiB request). Expandable segments let the allocator grow segments
# instead of carving a fixed arena, eliminating that trap.
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p "$SCRIPT_DIR/log"
LOGFILE="$SCRIPT_DIR/log/train_glm53flash.log"

# --- Cluster Configuration ---
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-16}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}

# --- Training Arguments ---
# Model reduction (layers/experts) is a standalone offline step now -- run
# python examples/glm5.3_flash/convert_checkpoint.py once (pass --num-hidden-layers
# / --n-routed-experts explicitly; no defaults), then point
# config.yaml's model.model_name_or_path at the reduced checkpoint.
CONFIG_FILE=${CONFIG_FILE:-"config.yaml"}

echo "============================================================"
echo "  glm5.3-flash Training"
echo "  NNODES: $NNODES"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NODE_RANK: $NODE_RANK"
echo "  CONFIG: $CONFIG_FILE"
echo "============================================================"

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config "$CONFIG_FILE" 2>&1 | tee "$LOGFILE"

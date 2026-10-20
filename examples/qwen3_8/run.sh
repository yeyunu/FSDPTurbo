#!/bin/bash
# ============================================================
# Qwen3.8-Flash-Next (qwen4_exp) FSDP2 Training Launch Script
# ============================================================
# Hardware: 64x Ascend910 (4 NPU nodes × 16 chips)
# Model:    336 GB bf16, 48 layers, 512 experts
# ============================================================

set -e

# --- NPU environment (auto-detected; skipped on CUDA) ---
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export HCCL_NPU_SOCKET_PORT_RANGE="22000,22999"
    export ASCEND_LAUNCH_BLOCKING=1
    export ACLNN_CACHE_LIMIT=100000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_CONNECT_TIMEOUT=7200
fi

# --- Fix NVRTC builtins for CUDA JIT (vision encoder triggers nvrtc) ---
for _nvrtc_dir in \
    "$(python -c 'import nvidia.cu13; print(nvidia.cu13.__path__[0]+"/lib")' 2>/dev/null)" \
    "$(python -c 'import nvidia.cuda_nvrtc; print(nvidia.cuda_nvrtc.__path__[0]+"/lib")' 2>/dev/null)" \
    /usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib \
    /usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib; do
    [[ -n "$_nvrtc_dir" && -f "$_nvrtc_dir/libnvrtc-builtins.so.13.0" ]] && export LD_LIBRARY_PATH="$_nvrtc_dir:$LD_LIBRARY_PATH" && break
    [[ -n "$_nvrtc_dir" && -f "$_nvrtc_dir/libnvrtc-builtins.so.12.8" ]] && export LD_LIBRARY_PATH="$_nvrtc_dir:$LD_LIBRARY_PATH" && break
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Cluster Configuration ---
NNODES=${NNODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-16}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# --- Training Arguments ---
CONFIG_FILE=${CONFIG_FILE:-"config.yaml"}
EXTRA_ARGS=()
if [[ -n "${NUM_HIDDEN_LAYERS:-}" ]]; then
    EXTRA_ARGS+=(--num-hidden-layers "$NUM_HIDDEN_LAYERS")
fi
if [[ -n "${ULYSSES_SIZE:-}" ]]; then
    EXTRA_ARGS+=(--ulysses-size "$ULYSSES_SIZE")
fi

echo "============================================================"
echo "Qwen3.8-Flash-Next Training"
echo "  NNODES: $NNODES"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  CONFIG: $CONFIG_FILE"
echo "  NUM_HIDDEN_LAYERS: ${NUM_HIDDEN_LAYERS:-full (48)}"
echo "  NGRAM_VOCAB_SIZE_BASE: ${NGRAM_VOCAB_SIZE_BASE:-full (20000000)}"
echo "  MODEL: /mnt/weight/Qwen3.8-Flash-Next (336 GB bf16)"
echo "============================================================"

LOG_DIR=logs/qwen38
LOG_FILE=qwen38_flash_next_$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p $LOG_DIR

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config "$CONFIG_FILE" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee ${LOG_DIR}/${LOG_FILE}.log

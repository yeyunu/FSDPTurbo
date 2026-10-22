#!/bin/bash
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at LICENSE (Apache-2.0) in the repo root.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================
# DeepSeek-V4-Flash training launcher (GPU / Ascend NPU, same script;
# every machine-specific value is an overridable environment variable).
#
# Random-init quick check (default):
#   bash run.sh
# Train from real released weights, 2000 steps:
#   MAX_STEPS=2000 CKPT=output/v4_real_4l16e.pt bash run.sh
# Or straight from the released safetensors directory:
#   MAX_STEPS=2000 WEIGHTS=/data/DeepSeek-V4-Flash-BF16 bash run.sh
#
# Overridable env (all optional):
#   CONFIG_FILE, MAX_STEPS, CKPT, WEIGHTS, OUTPUT_DIR,
#   NNODES, NODE_RANK, GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT,
#   DSV4_MODEL_PATH, DSV4_OUTPUT_DIR, DSV4_DATA_PATH,
#   CANN_SET_ENV_FILE, PYTHON_BIN
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Working directory: ${SCRIPT_DIR}"

# --- Python interpreter (default python; point at a conda env if needed) ---
PYTHON_BIN=${PYTHON_BIN:-python}

# --- CANN (Ascend NPU only). Sourced when present; GPU hosts simply skip it. ---
CANN_SET_ENV_FILE=${CANN_SET_ENV_FILE:-}
if [ -n "${CANN_SET_ENV_FILE}" ] && [ -f "${CANN_SET_ENV_FILE}" ]; then
    source "${CANN_SET_ENV_FILE}"
    # NPU runtime knobs inherited from the original run_npu.sh (no-op on GPU).
    export HCCL_NPU_SOCKET_PORT_RANGE="auto"
    export ASCEND_LAUNCH_BLOCKING=1
elif [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # NPU runtime knobs inherited from the original run_npu.sh (no-op on GPU).
    export HCCL_NPU_SOCKET_PORT_RANGE="auto"
    export ASCEND_LAUNCH_BLOCKING=1
else
    echo "WARN: CANN set_env.sh not found (fine on GPU hosts; set CANN_SET_ENV_FILE on NPU)."
fi
export TASK_QUEUE_ENABLE=2

# --- Repo root on PYTHONPATH so train.py can import fsdp_turbo. ---
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Machine-specific paths: placeholders in config_train.yaml are filled
# through os.path.expandvars in train.py; defaults are provided here so the
# yaml itself stays free of machine paths. ---
export DSV4_MODEL_PATH=${DSV4_MODEL_PATH:-"DSV4_MODEL_PATH"}
export DSV4_OUTPUT_DIR=${DSV4_OUTPUT_DIR:-"${SCRIPT_DIR}/output"}
export DSV4_DATA_PATH=${DSV4_DATA_PATH:-"DSV4_DATA_PATH"}

# --- Cluster configuration ---
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29536}
NODE_RANK=${NODE_RANK:-0}

# --- Training arguments ---
# Random initialization by default (fast smoke test); set CKPT (prepared .pt)
# or WEIGHTS (released safetensors dir) to train from the real weights.
CONFIG_FILE=${CONFIG_FILE:-"config_train.yaml"}
MAX_STEPS=${MAX_STEPS:-20}
CKPT=${CKPT:-""}
WEIGHTS=${WEIGHTS:-""}

echo "============================================================"
echo "DeepSeek-V4-Flash Training"
echo "  CONFIG: $CONFIG_FILE   MAX_STEPS: $MAX_STEPS"
echo "  CKPT: $CKPT   WEIGHTS: $WEIGHTS"
echo "  Nodes: $NNODES   Procs/Node: $GPUS_PER_NODE   Port: $MASTER_PORT   Node Rank: $NODE_RANK"
echo "  DSV4_MODEL_PATH: $DSV4_MODEL_PATH"
echo "============================================================"

mkdir -p logs

"${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config "$CONFIG_FILE" --max-steps "$MAX_STEPS" \
    ${CKPT:+--ckpt "$CKPT"} \
    ${WEIGHTS:+--weights "$WEIGHTS"} 2>&1 | tee ./logs/dsv4_train.log

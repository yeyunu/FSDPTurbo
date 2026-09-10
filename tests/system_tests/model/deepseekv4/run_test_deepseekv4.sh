#!/bin/bash
# ============================================================
# DeepSeekV4 CI baseline system test
#
# Runs a single merged config with all compatible features enabled
# (recompute, chunk_batch, chunk_loss, swap_activation,
#  swap_optimizer) on a 4-layer / 16-expert DeepSeekV4
# demo model.
#
# Usage:
#   bash run_test_deepseekv4.sh                # default 10 steps
#   NUM_STEPS=5 bash run_test_deepseekv4.sh    # custom steps
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export NUM_STEPS="${NUM_STEPS:-10}"

NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29505}"

echo "============================================================"
echo "DeepSeekV4 CI Baseline ST"
echo "  NUM_STEPS: $NUM_STEPS"
echo "============================================================"
echo ""

torchrun \
    --nproc_per_node="$NUM_GPUS_PER_NODE" \
    --nnodes="$NUM_NODES" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    test_deepseekv4.py

TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "DeepSeekV4 CI baseline: PASSED."
else
    echo "DeepSeekV4 CI baseline: FAILED (exit code: $TEST_RESULT)."
fi
exit $TEST_RESULT

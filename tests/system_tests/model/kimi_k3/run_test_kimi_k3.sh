#!/bin/bash
# ============================================================
# Run Kimi-K3 complete system tests with FSDPTurbo
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# MODEL_PATH only needs config/tokenizer/processor and remote-code files.
# The language model is reduced to NUM_HIDDEN_LAYERS and randomly initialized.
# DATASET_PATH points to multimodal JSON data (conversations + images).

mkdir -p /home/dataset/Kimi-K3-tmp
cp -r /home/dataset/Kimi-K3/* /home/dataset/Kimi-K3-tmp
cp -r "$SCRIPT_DIR/../../../../fsdp_turbo/models/kimi/"* /home/dataset/Kimi-K3-tmp

export MODEL_PATH=${MODEL_PATH:-/home/dataset/Kimi-K3-tmp}
export DATASET_PATH=${DATASET_PATH:-/home/dataset/coco/llava_instruct_150k_fsdp_turbo_100.json}
export NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-4}

# Match the 8-NPU Kimi-K3 example topology.
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
export FULLY_SHARD_PARALLEL_SIZE=${FULLY_SHARD_PARALLEL_SIZE:-8}
export EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-8}
export EXPERT_FULLY_SHARD_PARALLEL_SIZE=${EXPERT_FULLY_SHARD_PARALLEL_SIZE:-1}

NUM_NODES=${NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29503}

echo "Starting Kimi-K3 complete test suite..."
echo "This test will run the following configuration:"
echo "  FSDP + EP + TP + Recompute + Mixed Precision"
echo ""

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    test_kimi_k3.py
TEST_RESULT=$?

rm -rf /home/dataset/Kimi-K3-tmp

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "All Kimi-K3 tests completed."
else
    echo "Kimi-K3 tests failed."
fi
exit $TEST_RESULT

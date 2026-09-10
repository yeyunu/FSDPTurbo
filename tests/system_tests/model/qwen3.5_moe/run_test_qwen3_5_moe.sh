#!/bin/bash
# ============================================================
# Run Qwen3.5-MoE complete system tests with FSDPTurbo
# ============================================================

# MODEL_PATH only needs config.json + tokenizer/processor files (no weights).
# The model is built via from_config with NUM_HIDDEN_LAYERS and random init.
# DATASET_PATH points to a multimodal json (conversations + images).
export MODEL_PATH=/home/dataset/Qwen3.5-35B-A3B
export DATASET_PATH=/home/dataset/coco/llava_instruct_150k_fsdp_turbo_100.json
export NUM_HIDDEN_LAYERS=4

# st enables Ulysses CP: FSP=4, TP=1, ULYSSES=2, EP=8, EFP=1 on 8 NPUs (example uses ULYSSES=1).
export TENSOR_PARALLEL_SIZE=1
export FULLY_SHARD_PARALLEL_SIZE=4
export ULYSSES_PARALLEL_SIZE=2
export EXPERT_PARALLEL_SIZE=8
export EXPERT_FULLY_SHARD_PARALLEL_SIZE=1

NUM_NODES=1
NUM_GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29502

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Qwen3.5-MoE complete test suite..."
echo "This test will run the following configuration.:"
echo "  FSDP + EP + TP + Ulysses CP (GDN) + Module Patches + Recompute + Mixed Precision"
echo ""

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    test_qwen3_5_moe.py
TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "All Qwen3.5-MoE tests completed."
else
    echo "Qwen3.5-MoE tests failed."
fi
exit $TEST_RESULT

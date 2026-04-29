#!/bin/bash
# ============================================================
# Run Qwen3 complete system tests with FSDPTurbo
# ============================================================

NUM_NODES=1
NUM_GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29500

echo "Starting Qwen3 complete test suite..."
echo "This test will run the following configuration.:"
echo "  FSDP2 + EP + TP + Recompute + Mixed Precision + Hook Module "
echo ""

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    tests/system_tests/qwen3/test_qwen3.py

echo ""
echo "All Qwen3 tests completed."

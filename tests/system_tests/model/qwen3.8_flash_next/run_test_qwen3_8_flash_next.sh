#!/bin/bash
#=============================================
# Author: FSDPTurbo Team
# Date: 2026-11-11
# Description: Qwen3.8-Flash-Next (qwen4_exp) reduced-layer smoke system test on 8 NPUs.
#              FSP=4, TP=1, EP=8, EFP=1, post-fusion block-aligned FSDP, MoE all2all
#              dispatch with E-FSDP, PLE n-gram embedding CPU offload, and recompute.
# Remarks: When MODEL_PATH points at a real checkpoint dir (config.json + tokenizer/
#          processor, no weights) the test loads the production path with PLE CPU
#          offload and a multimodal dataloader. When MODEL_PATH is unset or absent
#          (the CI default for a brand-new model) the test self-bootstraps a tiny
#          Qwen4ExpConfig (PLE/MTP disabled) and synthetic text batches, exercising
#          the fused decoder + MoE dispatch + QSA indexer with no external asset —
#          a function-only smoke (training runs = PASS).
#=============================================

# MODEL_PATH may point at a real checkpoint dir (config.json + tokenizer/processor).
# When unset/absent the test self-bootstraps a synthetic config + synthetic batches,
# so the case runs on CI even before the real checkpoint is staged.
export MODEL_PATH=${MODEL_PATH:-/home/dataset/Qwen3.8-Flash-Next}
export DATASET_PATH=${DATASET_PATH:-/home/dataset/coco/llava_instruct_150k_fsdp_turbo_100.json}
export NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-4}
# Shrink the PLE n-gram table so the smoke test fits host memory (full ~51B table
# is only materialized in the production example); only used on the real path.
export NGRAM_VOCAB_SIZE_BASE=${NGRAM_VOCAB_SIZE_BASE:-200000}

# 8-NPU smoke topology: FSP=4, EP=8, EFP=1.
export FULLY_SHARD_PARALLEL_SIZE=${FULLY_SHARD_PARALLEL_SIZE:-4}
export EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-8}
export EXPERT_FULLY_SHARD_PARALLEL_SIZE=${EXPERT_FULLY_SHARD_PARALLEL_SIZE:-1}

NUM_NODES=${NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29506}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Qwen3.8-Flash-Next complete test suite..."
echo "This test will run the following configuration:"
echo "  FSDP + EP + E-FSDP + Recompute + Mixed Precision + Hook Module + Prefetch"
echo "  FSP=${FULLY_SHARD_PARALLEL_SIZE}, EP=${EXPERT_PARALLEL_SIZE}, EFP=${EXPERT_FULLY_SHARD_PARALLEL_SIZE}"
echo ""

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    test_qwen3_8_flash_next.py
TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "All Qwen3.8-Flash-Next tests completed."
else
    echo "Qwen3.8-Flash-Next tests failed."
fi
exit $TEST_RESULT

#!/bin/bash
# ============================================================
# Run GLM-5.3-Flash complete system tests with FSDPTurbo
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# MODEL_PATH supplies config/tokenizer/processor files (preprocessor_config.json,
# tokenizer*.json, chat template); modeling is imported from the fsdp_turbo
# package, so no remote-code population step is needed.
# The language model is reduced to NUM_HIDDEN_LAYERS and randomly initialized.
# DATASET_PATH points to multimodal JSON data (conversations + images).

# Fail before launching multiple ranks if the repo-local model package is incomplete.
python - <<'PY'
from importlib.util import find_spec

for module_name in (
    'fsdp_turbo.models.glm5_next.configuration_glm5_next',
    'fsdp_turbo.models.glm5_next.modeling_glm5_next',
    'fsdp_turbo.models.glm5_next.context_parallel',
):
    if find_spec(module_name) is None:
        raise ModuleNotFoundError(f'Required local model module is missing: {module_name}')
PY
if [ $? -ne 0 ]; then
    exit 1
fi

export MODEL_PATH=${MODEL_PATH:-/home/dataset/GLM-5.3-Flash}
export DATASET_PATH=${DATASET_PATH:-/home/dataset/coco/llava_instruct_150k_fsdp_turbo_100.json}
export NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-4}

# ST baseline: dense attention + eager indexer. Set DSA_IMPL=sfa and
# INDEXER_IMPL=triton to exercise the production sparse path.
export DSA_IMPL=${DSA_IMPL:-dense}
export INDEXER_IMPL=${INDEXER_IMPL:-eager}

# Match the 8-NPU GLM-5.3-Flash topology (16-way example scaled down).
export FULLY_SHARD_PARALLEL_SIZE=${FULLY_SHARD_PARALLEL_SIZE:-8}
export EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-8}
export EXPERT_FULLY_SHARD_PARALLEL_SIZE=${EXPERT_FULLY_SHARD_PARALLEL_SIZE:-1}

NUM_NODES=${NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29505}

echo "Starting GLM-5.3-Flash complete test suite..."
echo "This test will run the following configuration:"
echo "  FSDP + EP + Recompute + Mixed Precision"
echo "  dsa_implementation=$DSA_IMPL indexer_implementation=$INDEXER_IMPL"
echo ""

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    test_glm5_3_flash.py
TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "All GLM-5.3-Flash tests completed."
else
    echo "GLM-5.3-Flash tests failed."
fi
exit $TEST_RESULT

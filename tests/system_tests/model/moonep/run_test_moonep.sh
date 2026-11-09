#!/bin/bash
# ============================================================
# MoonEP on Ascend NPU: end-to-end numerical verification.
#
# Compares the MoonEP dispatcher against a replicated reference MoE, checking
# the forward output plus the hidden-state, router-probability and expert-weight
# gradients. A non-zero exit status means a numerical mismatch, not just a crash.
#
# Prerequisites:
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   npu_patch installed (local npu_patch/scripts/build_base.sh or image; not on any pip index)
#
#   bash tests/system_tests/model/moonep/run_test_moonep.sh
# ============================================================
set -euo pipefail

FSDP_TURBO_ROOT="${FSDP_TURBO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "${FSDP_TURBO_ROOT}"
export PYTHONPATH="${FSDP_TURBO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is unset; source the CANN environment first:" >&2
    echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh" >&2
    exit 1
fi

if ! python -c "import npu_patch" >/dev/null 2>&1; then
    echo "SKIP: npu_patch is not installed (not published on the CI index)." >&2
    echo "Build it in the MoonEP repo with bash npu_patch/scripts/build_base.sh, or bake it into the image." >&2
    exit 0
fi

NUM_NPUS_PER_NODE="${NUM_NPUS_PER_NODE:-8}"
NUM_NODES="${NUM_NODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Model/EP shape. E = MOONEP_EXPERTS_PER_RANK * world_size, and hidden == ffn
# keeps the per-rank expert block a multiple of the 2 MiB VMM granularity.
export MOONEP_HIDDEN="${MOONEP_HIDDEN:-512}"
export MOONEP_FFN="${MOONEP_FFN:-512}"
export MOONEP_EXPERTS_PER_RANK="${MOONEP_EXPERTS_PER_RANK:-4}"
export MOONEP_TOKENS="${MOONEP_TOKENS:-512}"
export MOONEP_TOP_K="${MOONEP_TOP_K:-4}"
export MOONEP_STEPS="${MOONEP_STEPS:-3}"

# The npu_patch backend reports no asynchronous communication support, so this
# flag is masked off before it reaches MoonEP and is inert on NPU.
export MOONEP_ASYNC_FINISH="${MOONEP_ASYNC_FINISH:-0}"

# CANN SHMEM bootstraps its config store one port above the torch rendezvous.
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/moonep_npu_$(date +%Y%m%d_%H%M%S).log"

echo "MoonEP Ascend end-to-end verification"
echo "  world           : ${NUM_NODES} node(s) x ${NUM_NPUS_PER_NODE} NPU(s)"
echo "  experts         : ${MOONEP_EXPERTS_PER_RANK} per rank"
echo "  hidden / ffn    : ${MOONEP_HIDDEN} / ${MOONEP_FFN}"
echo "  tokens / top-k  : ${MOONEP_TOKENS} / ${MOONEP_TOP_K}"
echo "  log             : ${LOG_FILE}"
echo ""

set +e
torchrun \
    --nproc_per_node="${NUM_NPUS_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    tests/system_tests/model/moonep/test_moonep.py 2>&1 | tee "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"
set -e

echo ""
if [[ "${STATUS}" -eq 0 ]]; then
    echo "PASS: MoonEP on Ascend matches the replicated reference."
else
    echo "FAIL: see ${LOG_FILE}"
fi
exit "${STATUS}"

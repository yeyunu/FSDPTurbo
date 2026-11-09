#!/bin/bash
# D-2 probe on Ascend: world=4, EP=2, efsdp=1 → edp=2.
# Checks that the two expert copies agree after sync_replica_gradients.
# Do not write this up as supporting edp>1.
#
#   bash tests/system_tests/model/moonep/run_test_moonep_edp2.sh
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

# D-2 is the 4-card replica mesh. Do not inherit the 8-card default from D-1.
NUM_NPUS_PER_NODE=4
NUM_NODES=1
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29501}"

export MOONEP_HIDDEN="${MOONEP_HIDDEN:-512}"
export MOONEP_FFN="${MOONEP_FFN:-512}"
export MOONEP_EXPERTS_PER_RANK="${MOONEP_EXPERTS_PER_RANK:-4}"
export MOONEP_TOKENS="${MOONEP_TOKENS:-512}"
export MOONEP_TOP_K="${MOONEP_TOP_K:-4}"
export MOONEP_STEPS="${MOONEP_STEPS:-3}"
export MOONEP_ASYNC_FINISH="${MOONEP_ASYNC_FINISH:-0}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/moonep_edp2_$(date +%Y%m%d_%H%M%S).log"

echo "MoonEP D-2 replica-consistency probe"
echo "  world / EP / efsdp / edp : 4 / 2 / 1 / 2"
echo "  log                      : ${LOG_FILE}"
echo ""

set +e
torchrun \
    --nproc_per_node="${NUM_NPUS_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    tests/system_tests/model/moonep/test_moonep_edp_replica.py 2>&1 | tee "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"
set -e

echo ""
if [[ "${STATUS}" -eq 0 ]]; then
    echo "PASS: D-2 replica home_grad agreed (not a support claim)."
else
    echo "FAIL: see ${LOG_FILE}"
fi
exit "${STATUS}"

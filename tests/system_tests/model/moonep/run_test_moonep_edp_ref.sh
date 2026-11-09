#!/bin/bash
# ============================================================
# MoonEP replica reduction at edp>1, against an absolute reference.
#
# D-2 only showed the two expert copies agree with each other, which a uniform
# scale error survives untouched. This launcher runs the world-summed reference
# check instead, and doubles as the A/B for what the reduction costs.
#
# Prerequisites:
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   npu_patch installed (local npu_patch/scripts/build_base.sh or image; not on any pip index)
#
# Ports: D-1 uses 29500 and D-2 uses 29501, so this one takes 29502. All three
# must be run serially.
#
# V-2  absolute reference at edp=2 (the default):
#   bash tests/system_tests/model/moonep/run_test_moonep_edp_ref.sh
#
# V-1  config invariance, E pinned so both runs are the same model:
#   MOONEP_NUM_EXPERTS=16 MOONEP_EDP=1 MOONEP_EP=4 \
#     MOONEP_DUMP_DIR=./logs/edp1 bash tests/.../run_test_moonep_edp_ref.sh
#   MOONEP_NUM_EXPERTS=16 MOONEP_EDP=2 MOONEP_EP=2 \
#     MOONEP_DUMP_DIR=./logs/edp2 bash tests/.../run_test_moonep_edp_ref.sh
#
# V-3  replica drift over 50 steps:
#   MOONEP_STEPS=50 MOONEP_OPT_LR=0.01 bash tests/.../run_test_moonep_edp_ref.sh
#
# V-5  gradient accumulation:
#   MOONEP_ACCUM_STEPS=2 bash tests/.../run_test_moonep_edp_ref.sh
#
# V-6 / V-7  cost of the reduction, same config both arms, twice each:
#   MOONEP_MEASURE=1 MOONEP_STEPS=12 MOONEP_EDP_SYNC=1 bash tests/.../run_test_moonep_edp_ref.sh
#   MOONEP_MEASURE=1 MOONEP_STEPS=12 MOONEP_EDP_SYNC=0 bash tests/.../run_test_moonep_edp_ref.sh
#
# V-8  edp=4 on 8 cards, and the shard round trip:
#   MOONEP_NUM_EXPERTS=16 MOONEP_EDP=4 MOONEP_EP=2 bash tests/.../run_test_moonep_edp_ref.sh
#   MOONEP_CKPT=1 bash tests/.../run_test_moonep_edp_ref.sh
#
# A pass is not a 7.8 upgrade and must not be written up as supporting edp>1.
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

# The mesh drives the card count, so world is never silently wrong: edp * ep.
export MOONEP_EDP="${MOONEP_EDP:-2}"
export MOONEP_EP="${MOONEP_EP:-2}"
NUM_NPUS_PER_NODE="${NUM_NPUS_PER_NODE:-$((MOONEP_EDP * MOONEP_EP))}"
NUM_NODES="${NUM_NODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29502}"

if [[ "$((MOONEP_EDP * MOONEP_EP))" -ne "$((NUM_NPUS_PER_NODE * NUM_NODES))" ]]; then
    echo "MOONEP_EDP(${MOONEP_EDP}) * MOONEP_EP(${MOONEP_EP}) must equal the world size" >&2
    echo "  (NUM_NPUS_PER_NODE=${NUM_NPUS_PER_NODE} x NUM_NODES=${NUM_NODES})" >&2
    exit 1
fi

export MOONEP_HIDDEN="${MOONEP_HIDDEN:-512}"
export MOONEP_FFN="${MOONEP_FFN:-512}"
export MOONEP_TOKENS="${MOONEP_TOKENS:-512}"
export MOONEP_TOP_K="${MOONEP_TOP_K:-4}"
export MOONEP_STEPS="${MOONEP_STEPS:-3}"
export MOONEP_ASYNC_FINISH="${MOONEP_ASYNC_FINISH:-0}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"

# E defaults to the per-rank convention of the other two scripts. Pin it only when
# comparing configs, where both runs must describe the same model.
if [[ -n "${MOONEP_NUM_EXPERTS:-}" ]]; then
    export MOONEP_NUM_EXPERTS
    EXPERTS_DESC="${MOONEP_NUM_EXPERTS} (pinned, E/ep=$((MOONEP_NUM_EXPERTS / MOONEP_EP)))"
else
    export MOONEP_EXPERTS_PER_RANK="${MOONEP_EXPERTS_PER_RANK:-4}"
    EXPERTS_DESC="$((MOONEP_EXPERTS_PER_RANK * MOONEP_EP)) (${MOONEP_EXPERTS_PER_RANK} per rank)"
fi

export MOONEP_EDP_SYNC="${MOONEP_EDP_SYNC:-1}"
export MOONEP_ACCUM_STEPS="${MOONEP_ACCUM_STEPS:-1}"
export MOONEP_OPT_LR="${MOONEP_OPT_LR:-0}"
export MOONEP_DRIFT_EVERY="${MOONEP_DRIFT_EVERY:-10}"
export MOONEP_MEASURE="${MOONEP_MEASURE:-0}"
export MOONEP_WARMUP="${MOONEP_WARMUP:-2}"

# A timing or memory run must not carry the naive reference: it costs more than the
# kernel under test and allocates its own FP32 buffers, so it would dilute the very
# gap the A/B is looking for. Correctness runs keep it on.
if [[ -z "${MOONEP_REF_CHECK:-}" ]]; then
    if [[ "${MOONEP_MEASURE}" == "1" ]]; then
        MOONEP_REF_CHECK=0
    else
        MOONEP_REF_CHECK=1
    fi
fi
export MOONEP_REF_CHECK
export MOONEP_CKPT="${MOONEP_CKPT:-0}"
[[ -n "${MOONEP_DUMP_DIR:-}" ]] && export MOONEP_DUMP_DIR

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/moonep_edp_ref_edp${MOONEP_EDP}_ep${MOONEP_EP}_sync${MOONEP_EDP_SYNC}_$(date +%Y%m%d_%H%M%S).log"

echo "MoonEP edp>1 absolute-reference check"
echo "  world / edp / efsdp / ep : $((NUM_NPUS_PER_NODE * NUM_NODES)) / ${MOONEP_EDP} / 1 / ${MOONEP_EP}"
echo "  replica reduction        : ${MOONEP_EDP_SYNC}"
echo "  steps / accum            : ${MOONEP_STEPS} / ${MOONEP_ACCUM_STEPS}"
echo "  experts                  : ${EXPERTS_DESC}"
echo "  measure / ckpt / lr      : ${MOONEP_MEASURE} / ${MOONEP_CKPT} / ${MOONEP_OPT_LR}"
echo "  reference check          : ${MOONEP_REF_CHECK}"
echo "  dump dir                 : ${MOONEP_DUMP_DIR:-<none>}"
echo "  log                      : ${LOG_FILE}"
echo ""

set +e
torchrun \
    --nproc_per_node="${NUM_NPUS_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    tests/system_tests/model/moonep/test_moonep_edp_ref.py 2>&1 | tee "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"
set -e

echo ""
if [[ "${STATUS}" -eq 0 ]]; then
    echo "PASS: the replica reduction matched the world-summed reference (not a support claim)."
else
    echo "FAIL: see ${LOG_FILE}"
fi
exit "${STATUS}"

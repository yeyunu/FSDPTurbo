#!/bin/bash
# ============================================================
# MoonEP single-node 8-card ACCURACY suite.
#
# Chains the per-tensor and absolute-reference checks that already ship in this
# directory, pinned to the three 8-card expert layouts worth looking at:
#
#   A) EP=8              per-tensor vs a replicated reference  (test_moonep.py)
#   B) EDP=2 x EP=4      absolute world-summed reference       (test_moonep_edp_ref.py)
#   C) EDP=4 x EP=2      absolute world-summed reference       (test_moonep_edp_ref.py)
#
# A proves dispatch/prefetch/GEMM/combine and the home-rank reduction are
# bit-close to a naive MoE. B and C additionally exercise the cross-replica
# (EDP>1) gradient sum on 8 cards. A non-zero exit means a numerical mismatch.
#
# This is accuracy only. Memory and performance are the qwen3 A/B described in
# README_8card.md, which drives tests/system_tests/model/qwen3/run_test_qwen3.sh
# twice with EP_DISPATCHER=all2all and EP_DISPATCHER=moonep.
#
# Prerequisites:
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   npu_patch installed (local npu_patch/scripts/build_base.sh or image; not on any pip index)
#
#   bash tests/system_tests/model/moonep/run_test_moonep_8card.sh
#
# Nothing here pushes to any remote.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSDP_TURBO_ROOT="${FSDP_TURBO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
export FSDP_TURBO_ROOT

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

# One shared shape for all three arms so the only variable is the EDP/EP split.
export MOONEP_HIDDEN="${MOONEP_HIDDEN:-512}"
export MOONEP_FFN="${MOONEP_FFN:-512}"
export MOONEP_TOKENS="${MOONEP_TOKENS:-512}"
export MOONEP_TOP_K="${MOONEP_TOP_K:-4}"
export MOONEP_STEPS="${MOONEP_STEPS:-3}"

declare -a NAMES=()
declare -a CODES=()

# A) EP=8 per-tensor check. run_test_moonep.sh reads NUM_NPUS_PER_NODE and the
#    per-rank expert count; 4 experts/rank => E=32 over 8 ranks.
#
#    Four and not two: at hidden == ffn == 512 one expert row is 512 KiB, so the
#    per-rank BF16 block is only granularity aligned when E/R is a multiple of
#    four. Two experts/rank gives a 1 MiB block against a 2 MiB granule and the
#    run stops at the chunk-alignment guard before any numbers are produced.
#    B and C keep E=16 because their E/R (4 and 8) already satisfies this.
echo ""
echo "============================================================"
echo ">>> A) EP=8 per-tensor vs replicated reference"
echo "============================================================"
NUM_NPUS_PER_NODE=8 MOONEP_EXPERTS_PER_RANK="${MOONEP_EXPERTS_PER_RANK:-4}" \
    bash "${SCRIPT_DIR}/run_test_moonep.sh"
NAMES+=("A: EP=8 per-tensor"); CODES+=("$?")

# B) EDP=2 x EP=4 absolute reference. E pinned so both EDP arms are the same model.
echo ""
echo "============================================================"
echo ">>> B) EDP=2 x EP=4 absolute reference"
echo "============================================================"
MOONEP_EDP=2 MOONEP_EP=4 MOONEP_NUM_EXPERTS="${MOONEP_NUM_EXPERTS:-16}" \
    bash "${SCRIPT_DIR}/run_test_moonep_edp_ref.sh"
NAMES+=("B: EDP=2xEP=4 abs-ref"); CODES+=("$?")

# C) EDP=4 x EP=2 absolute reference, same E.
echo ""
echo "============================================================"
echo ">>> C) EDP=4 x EP=2 absolute reference"
echo "============================================================"
MOONEP_EDP=4 MOONEP_EP=2 MOONEP_NUM_EXPERTS="${MOONEP_NUM_EXPERTS:-16}" \
    bash "${SCRIPT_DIR}/run_test_moonep_edp_ref.sh"
NAMES+=("C: EDP=4xEP=2 abs-ref"); CODES+=("$?")

echo ""
echo "============================================================"
echo "MoonEP 8-card accuracy summary"
echo "============================================================"
status=0
for i in "${!NAMES[@]}"; do
    if [[ "${CODES[$i]}" -eq 0 ]]; then
        echo "  PASS  ${NAMES[$i]}"
    else
        echo "  FAIL  ${NAMES[$i]} (exit ${CODES[$i]})"
        status=1
    fi
done
echo ""
if [[ "${status}" -eq 0 ]]; then
    echo "ALL PASS: MoonEP matches the reference on all three 8-card layouts."
else
    echo "FAILURES above; see each arm's log under ./logs."
fi
exit "${status}"

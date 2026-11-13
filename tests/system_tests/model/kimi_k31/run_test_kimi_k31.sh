#!/usr/bin/env bash
# CI runs the portable checks; npu/all also run fused DSA and packing regressions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
TEST_TARGET="tests/system_tests/model/kimi_k31/test_kimi_k31.py"
MODE="${1:-ci}"
if (( $# > 0 )); then
    shift
fi
PYTEST_ARGS=()

case "$MODE" in
    ci)
        ;;
    cpu)
        PYTEST_ARGS=("--deselect=$TEST_TARGET::TestKimiK31NPU")
        ;;
    npu)
        TEST_TARGET="$TEST_TARGET::TestKimiK31NPU"
        PYTEST_ARGS=("tests/system_tests/model/kimi_k31/test_k31_attention_npu.py"
                     "tests/system_tests/model/kimi_k31/test_k31_packing.py")
        ;;
    all)
        PYTEST_ARGS=("tests/system_tests/model/kimi_k31/test_k31_attention_npu.py"
                     "tests/system_tests/model/kimi_k31/test_k31_packing.py")
        ;;
    *)
        echo "Usage: $0 [ci|cpu|npu|all] [pytest arguments...]" >&2
        exit 2
        ;;
esac

cd "$REPO_ROOT"
exec python -m pytest -q -rs "$TEST_TARGET" "${PYTEST_ARGS[@]}" "$@"

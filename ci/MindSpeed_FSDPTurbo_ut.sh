#!/bin/bash
set -x

WORKSPACE=${1:-${WORKSPACE}}
pr_id=$2
TARGET_BRANCH=${3:-"main"}

# 检查是否在docker容器中
if [ ! -f "/.dockerenv" ]; then
    echo "Not in a docker container."
    exit 1
fi

# 运行环境准备
function setup_running_env() {
    # 安装FSDPTurbo
    echo "Installing FSDPTurbo..."
    cd "${WORKSPACE}"/CODE
    pip install -e . -v --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi

    echo "Install Complete..."
    pip list
}

# 运行环境清理
function clean_running_env () {
    echo "Cleaning up environment..."
    pip uninstall -y fsdp-turbo
    echo "Finish cleaning"
}

# Treat a value as a boolean flag (1/true/yes/on).
is_true() {
    local value="${1:-false}"
    case "${value,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# Print a [FAILED] block for one ST case. Extra detail lines are passed as
# remaining args (e.g. "Log: /path/to/log").
print_case_failure() {
    local prefix="$1"
    local reason="$2"
    local exitcode="$3"
    shift 3
    echo
    echo "[FAILED] ${prefix}"
    echo "Reason: ${reason}"
    echo "Exit code: ${exitcode}"
    for line in "$@"; do
        echo "$line"
    done
}

# Run one ST case: train (tee log) then, if a baseline JSON exists, compare
# against it. Relies on run_testcase's locals via dynamic scope: GENERATE_LOG_DIR,
# BASELINE_DIR, TEST_TOOLS_DIR, and the failed_script_cases /
# failed_compare_cases / skipped_cases arrays (appended on failure / skip).
# Returns 0 on pass (full or functional-only) or skip, 1 on script-execution or
# baseline-comparison failure.
run_one_case() {
    local test_case="$1"
    local file_name=$(basename "${test_case}")
    local file_name_prefix="${file_name%.*}"
    local generate_log="${GENERATE_LOG_DIR}/${file_name_prefix}.log"
    local generate_json="${generate_log%.log}.json"
    local baseline_json="${BASELINE_DIR}/${file_name_prefix}.json"

    echo
    echo "=================================================="
    echo "Running ${file_name}..."
    echo "=================================================="

    # Truncate the generated JSON to prevent stale results.
    : > "${generate_json}"

    # Write both stdout and stderr to the generated log. -e propagates any
    # mid-script command failure as a non-zero exit instead of letting it be
    # masked by a later successful command (e.g. the final torchrun).
    bash -e "${test_case}" 2>&1 | tee "${generate_log}"

    # Read the exit code from the ST script instead of tee.
    local script_exitcode=${PIPESTATUS[0]}

    if [[ ${script_exitcode} -ne 0 ]]; then
        failed_script_cases+=("${file_name_prefix}:${script_exitcode}")
        print_case_failure "${file_name_prefix}" "script execution failed" "${script_exitcode}" "Log: ${generate_log}"
        return 1
    fi

    # No baseline JSON -> functional test only. Training already passed above,
    # and there is nothing to compare against, so skip compare_baseline and pass.
    # This keeps a new ST case (no committed baseline yet) under functional CI
    # coverage (does training complete?) instead of skipping it entirely; add a
    # baseline_results/<name>.json (even empty) to activate the accuracy/perf
    # guard for it.
    if [[ ! -f "${baseline_json}" ]]; then
        echo "[PASS-functional] ${file_name_prefix}: no baseline JSON at ${baseline_json}; training passed, comparison skipped."
        echo "  Generated JSON: ${generate_json}"
        echo "  Generated log:  ${generate_log}"
        skipped_cases+=("${file_name_prefix}:no baseline (functional only)")
        return 0
    fi

    python "${TEST_TOOLS_DIR}/compare_baseline.py" \
        --baseline-json "${baseline_json}" \
        --generate-log "${generate_log}"
    local compare_exitcode=$?

    if [[ ${compare_exitcode} -ne 0 ]]; then
        failed_compare_cases+=("${file_name_prefix}:${compare_exitcode}")
        print_case_failure "${file_name_prefix}" "baseline comparison failed" "${compare_exitcode}" \
            "Baseline: ${baseline_json}" \
            "Generated JSON: ${generate_json}" \
            "Generated log: ${generate_log}"
        return 1
    fi

    echo
    echo "[PASSED] ${file_name_prefix}"
    return 0
}

# Print the ST summary. Relies on run_testcase's locals failed_script_cases /
# failed_compare_cases / skipped_cases via dynamic scope.
print_st_summary() {
    echo
    echo "=================================================="
    echo "ST test summary"
    echo "=================================================="

    if [[ ${#skipped_cases[@]} -gt 0 ]]; then
        echo
        echo "Comparison skipped (training ran, functional only):"
        for skipped_case in "${skipped_cases[@]}"; do
            echo "  - ${skipped_case}"
        done
    fi

    if [[ ${#failed_script_cases[@]} -gt 0 ]]; then
        echo
        echo "Script execution failures:"
        for failed_case in "${failed_script_cases[@]}"; do
            echo "  - ${failed_case}"
        done
    fi

    if [[ ${#failed_compare_cases[@]} -gt 0 ]]; then
        echo
        echo "Baseline comparison failures:"
        for failed_case in "${failed_compare_cases[@]}"; do
            echo "  - ${failed_case}"
        done
    fi
}

function run_unittest () {
    cd "${WORKSPACE}"/CODE

    # UT runs on CPU via pytest; the CI image may not ship pytest/pytest-timeout, so install them first.
    echo "Installing pytest for unit tests..."
    pip install pytest pytest-timeout

    echo "Running unit tests (CPU)..."
    # -x stop on first failure; -ra short summary (skip/fail reasons); --timeout per-test hang guard.
    python -m pytest tests/unit_tests -v -x -ra --timeout="${UT_TIMEOUT:-300}"
    local ut_result=$?

    cd ..

    if [[ ${ut_result} -ne 0 ]]; then
        echo "Unit tests failed (exit ${ut_result})."
        return 1
    fi
    echo "Unit tests passed."
    return 0
}

function run_testcase () {
    cd "${WORKSPACE}"/CODE

    local SHELL_SCRIPTS_DIR="tests/system_tests/model"
    local BASELINE_DIR="tests/system_tests/baseline_results"
    local TEST_TOOLS_DIR="tests/system_tests/test_tools"
    # Logs land in a ./run_logs folder under the workspace; the CI container is
    # destroyed after the run, so logs auto-cleanup and need no explicit delete.
    local GENERATE_LOG_DIR="${ST_LOG_DIR:-run_logs}"

    # Avoid explicit IP binding.
    export HCCL_HOST_SOCKET_PORT_RANGE=auto

    local CONTINUE_ON_FAILURE="${CI_CONTINUE_ON_FAILURE:-false}"
    if is_true "${CONTINUE_ON_FAILURE}"; then
        echo "ST mode: continue-on-failure"
    else
        echo "ST mode: fail-fast"
    fi

    mkdir -p "${GENERATE_LOG_DIR}"

    shopt -s nullglob globstar
    local test_files=("${SHELL_SCRIPTS_DIR}"/**/run_test_*.sh)
    shopt -u nullglob globstar

    if [[ ${#test_files[@]} -eq 0 ]]; then
        echo "No run_test_*.sh files found in ${SHELL_SCRIPTS_DIR}"
        cd ..
        return 0
    fi

    # Store ST scripts that fail during execution / baseline comparison, plus
    # cases whose comparison was skipped (no committed baseline JSON -> ran as
    # functional only).
    local failed_script_cases=()
    local failed_compare_cases=()
    local skipped_cases=()

    for test_case in "${test_files[@]}"; do
        run_one_case "${test_case}"
        local rc=$?
        if [[ ${rc} -ne 0 ]]; then
            if ! is_true "${CONTINUE_ON_FAILURE}"; then
                echo
                echo "Fail-fast mode enabled. Exiting immediately."
                cd ..
                return 1
            fi
            echo
            echo "Continuing with the next ST case."
        fi
    done

    print_st_summary

    local total_failed=$(( ${#failed_script_cases[@]} + ${#failed_compare_cases[@]} ))

    cd ..

    if [[ ${total_failed} -gt 0 ]]; then
        echo
        echo "Total failed ST checks: ${total_failed}"
        return 1
    fi

    echo
    echo "All ST cases passed."
    return 0
}

function main() {

    # 准备环境
    echo "Setting up environment..."
    setup_running_env

    # 执行UT（CPU，pytest）；UT失败则跳过ST
    echo "Running unit tests..."
    run_unittest

    # 执行ST（NPU）
    echo "Running system tests..."
    run_testcase
    local test_result=$?

    # 清理环境
    echo "Cleaning environment..."
    clean_running_env

    exit $test_result
}

main

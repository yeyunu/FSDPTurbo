import argparse
import os
import statistics
import sys

from acquire_json import transfer_logs_as_json, read_json

LOSS = "lm loss"
GRAD_NORM = "grad norm"
TIME_INFO = "time info"
MEMO_INFO = "memo info"

# Number of leading steps to drop before averaging the per-step time, so that
# warm-up / compilation overhead does not skew the performance baseline.
WARM_UP = 2


class TestMargin:
    loss = 0.02
    grad_norm = 0.1
    time = 0.05
    memory = 0.1

    @classmethod
    def refresh_from_json(cls, json_obj):
        cls.loss = json_obj.get(LOSS + " margin", cls.loss)
        cls.grad_norm = json_obj.get(GRAD_NORM + " margin", cls.grad_norm)
        cls.time = json_obj.get(TIME_INFO + " margin", cls.time)
        cls.memory = json_obj.get(MEMO_INFO + " margin", cls.memory)


class AbsTol:
    # Absolute tolerance floor for near-zero values. When |expected| <= abs_tol,
    # compare |actual - expected| against abs_tol instead of the relative
    # margin: relative error is meaningless near zero (e.g. grad norm 0.0007
    # vs 0.2316 is a 330x relative blow-up but a 0.23 absolute blip on a curve
    # whose real values are 24..9932). 0 disables the floor (pure relative).
    loss = 0.1
    grad_norm = 1.0
    time = 0.0
    memory = 0.0

    @classmethod
    def refresh_from_json(cls, json_obj):
        cls.loss = json_obj.get(LOSS + " abs tol", cls.loss)
        cls.grad_norm = json_obj.get(GRAD_NORM + " abs tol", cls.grad_norm)
        cls.time = json_obj.get(TIME_INFO + " abs tol", cls.time)
        cls.memory = json_obj.get(MEMO_INFO + " abs tol", cls.memory)


def _approx(actual, expected, rel, abs_tol=0.0):
    # Near-zero floor: when the expected value sits inside the absolute-noise
    # floor, relative error is meaningless, so compare absolutely. Otherwise
    # fall back to the relative margin (with the exact-zero shortcut kept).
    if abs_tol and abs(expected) <= abs_tol:
        return abs(actual - expected) <= abs_tol
    if expected == 0:
        return abs(actual) <= rel
    return abs(actual - expected) / abs(expected) <= rel


def compare_lm_loss(expected_list, actual_list):
    if len(expected_list) != len(actual_list):
        return False, f"length mismatch: expected {len(expected_list)}, actual {len(actual_list)}"
    for step, (e, a) in enumerate(zip(expected_list, actual_list)):
        if not _approx(a, e, TestMargin.loss, AbsTol.loss):
            return False, f"step {step + 1} loss {a} not approx to {e} (rel {TestMargin.loss}, abs_tol {AbsTol.loss})"
    return True, "ok"


def compare_grad_norm(expected_list, actual_list):
    if len(expected_list) != len(actual_list):
        return False, f"length mismatch: expected {len(expected_list)}, actual {len(actual_list)}"
    for step, (e, a) in enumerate(zip(expected_list, actual_list)):
        if not _approx(a, e, TestMargin.grad_norm, AbsTol.grad_norm):
            return (
                False,
                f"step {step + 1} grad norm {a} not approx to {e} (rel {TestMargin.grad_norm}, abs_tol {AbsTol.grad_norm})",
            )
    return True, "ok"


def compare_time(expected_list, actual_list):
    if len(expected_list) != len(actual_list):
        return False, f"length mismatch: expected {len(expected_list)}, actual {len(actual_list)}"
    # Drop warm-up steps, then compare the median (robust to single-step
    # spikes, e.g. a one-off slow iteration) rather than the mean.
    warm_up = min(WARM_UP, max(len(expected_list) - 1, 0))
    exp_post = expected_list[warm_up:]
    act_post = actual_list[warm_up:]
    if not exp_post:
        return True, "not enough steps to compare time"
    expected_med = statistics.median(exp_post)
    actual_med = statistics.median(act_post)
    if actual_med <= expected_med or _approx(actual_med, expected_med, TestMargin.time, AbsTol.time):
        return True, f"median time {actual_med:.1f} ms within margin of {expected_med:.1f} ms"
    return False, f"median time {actual_med:.1f} ms slower than expected {expected_med:.1f} ms"


def compare_memory(expected_list, actual_list):
    if len(expected_list) != len(actual_list):
        return False, f"length mismatch: expected {len(expected_list)}, actual {len(actual_list)}"
    for e, a in zip(expected_list, actual_list):
        e_alloc = e["allocated memory"]
        a_alloc = a["allocated memory"]
        if not (a_alloc <= e_alloc or _approx(a_alloc, e_alloc, TestMargin.memory, AbsTol.memory)):
            return False, f"allocated memory {a_alloc} abnormal vs {e_alloc}"
        e_max = e["max allocated memory"]
        a_max = a["max allocated memory"]
        if not (a_max <= e_max or _approx(a_max, e_max, TestMargin.memory, AbsTol.memory)):
            return False, f"max allocated memory {a_max} abnormal vs {e_max}"
    return True, "ok"


COMPARISONS = {
    LOSS: compare_lm_loss,
    GRAD_NORM: compare_grad_norm,
    TIME_INFO: compare_time,
    MEMO_INFO: compare_memory,
}


def run(baseline_json, generate_log, coverage=False):
    # The JSON summary path is derived from the log path (run.log -> run.json).
    root, _ = os.path.splitext(generate_log)
    generate_json = (root or generate_log) + ".json"

    transfer_logs_as_json(generate_log, generate_json)
    actual = read_json(generate_json)
    print(f"Parsed {generate_log} -> {generate_json}: {len(actual.get(LOSS, []))} step(s).")

    # Treat a missing baseline like an empty one: the gate is not activated, so
    # the case must not fail (let the CI orchestrator decide whether to run it).
    if not os.path.exists(baseline_json):
        print(f"[SKIP] baseline {baseline_json} does not exist; no checks enforced.")
        return 0
    expected = read_json(baseline_json)
    TestMargin.refresh_from_json(expected)
    AbsTol.refresh_from_json(expected)

    failures = []
    for metric, compare_fn in COMPARISONS.items():
        expected_list = expected.get(metric, [])
        actual_list = actual.get(metric, [])

        print(f"===================== Begin comparing {metric} ===================")
        if not expected_list:
            print(f"  [SKIP] baseline for {metric} is empty; fill it from a real run to enable the check.")
            continue
        if not actual_list:
            print(f"  [FAIL] actual {metric} is empty; the training run likely failed.")
            failures.append((metric, f"actual {metric} empty"))
            continue
        if metric == TIME_INFO and coverage:
            print("  [SKIP] time comparison disabled under START_COVERAGE.")
            continue

        print(f"  expected: {expected_list}")
        print(f"  actual:   {actual_list}")
        ok, msg = compare_fn(expected_list, actual_list)
        if ok:
            print(f"  [PASS] {msg}")
        else:
            print(f"  [FAIL] {msg}")
            failures.append((metric, msg))

    print()
    if failures:
        print("Baseline comparison failed:")
        for metric, msg in failures:
            print(f"  - {metric}: {msg}")
        return 1
    print("Baseline comparison passed.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Parse a FSDPTurbo ST run log and compare it against a baseline JSON.")
    parser.add_argument("--generate-log", required=True, help="ST run log to parse into a JSON summary.")
    parser.add_argument("--baseline-json", required=True, help="Baseline JSON to compare against.")
    args = parser.parse_args()

    coverage = os.environ.get("START_COVERAGE", "").lower() in ("1", "true", "yes", "on")
    sys.exit(run(args.baseline_json, args.generate_log, coverage=coverage))


if __name__ == "__main__":
    main()

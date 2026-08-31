import argparse
import json
import os
import re
import statistics


LOSS = "lm loss"
GRAD_NORM = "grad norm"
TIME_INFO = "time info"
MEMO_INFO = "memo info"

# Numeric tokens use ``\S+`` rather than a digit-only character class so that
# nan / inf / -inf (printed by f-strings when a float is non-finite) are
# captured instead of silently dropping the whole step, which would leave the
# per-step metric lists with mismatched lengths and misalign them against the
# baseline. float() then parses nan/inf natively.
_step_pattern = re.compile(
    r"\[Rank\s+(\d+)\]\s+step\s+(\d+)\s+step time:\s+([\d.]+)s\s+"
    r"loss:\s+(\S+)\s+grad norm:\s+(\S+)\s+"
    r"memory:\s+(\S+)\s+GB\s+max memory:\s+(\S+)\s+GB"
)


def _parse_log(log_file):
    """Parse one ST run log into a metric dict (no file written)."""
    data = {LOSS: [], GRAD_NORM: [], TIME_INFO: [], MEMO_INFO: []}
    with open(log_file, "r", encoding="utf-8-sig") as f:
        log_content = f.read()
    matches = _step_pattern.findall(log_content)
    if matches:
        data[LOSS] = [float(m[3]) for m in matches]
        data[GRAD_NORM] = [float(m[4]) for m in matches]
        # step time is printed in seconds; store as ms to match CI convention.
        data[TIME_INFO] = [float(m[2]) * 1000.0 for m in matches]
        # Keep the last reported memory snapshot per rank.
        last_memo = {}
        for m in matches:
            rank = int(m[0])
            last_memo[rank] = {
                "rank": rank,
                "allocated memory": float(m[5]),
                "max allocated memory": float(m[6]),
            }
        data[MEMO_INFO] = [last_memo[r] for r in sorted(last_memo)]
    return data


def _median_per_step(series_list):
    """Per-step median across runs; tolerates differing list lengths."""
    n = min((len(s) for s in series_list if isinstance(s, list)), default=0)
    return [statistics.median([s[i] for s in series_list if i < len(s)]) for i in range(n)]


def _median_memo(runs):
    """Per-rank median of allocated / max allocated memory across runs."""
    per_rank = {}
    for run in runs:
        for entry in run.get(MEMO_INFO, []):
            rank = entry.get("rank", 0)
            bucket = per_rank.setdefault(rank, {"allocated memory": [], "max allocated memory": []})
            bucket["allocated memory"].append(entry.get("allocated memory", 0))
            bucket["max allocated memory"].append(entry.get("max allocated memory", 0))
    merged = []
    for rank in sorted(per_rank):
        b = per_rank[rank]
        merged.append(
            {
                "rank": rank,
                "allocated memory": statistics.median(b["allocated memory"]) if b["allocated memory"] else 0,
                "max allocated memory": statistics.median(b["max allocated memory"])
                if b["max allocated memory"]
                else 0,
            }
        )
    return merged


def merge_logs_as_baseline(log_files, output_json_file):
    """Parse multiple ST run logs and write a median baseline JSON.

    ``lm loss`` / ``grad norm`` / ``time info`` are aggregated as the per-step
    median across runs (robust to a single divergent run); ``memo info`` as the
    per-rank median of allocated / max allocated memory. A single log is
    written verbatim (identical to transfer_logs_as_json).
    """
    runs = [_parse_log(log) for log in log_files]
    if len(runs) == 1:
        data = runs[0]
    else:
        data = {
            LOSS: _median_per_step([r.get(LOSS, []) for r in runs]),
            GRAD_NORM: _median_per_step([r.get(GRAD_NORM, []) for r in runs]),
            TIME_INFO: _median_per_step([r.get(TIME_INFO, []) for r in runs]),
            MEMO_INFO: _median_memo(runs),
        }
    with open(output_json_file, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4)
    return data


def transfer_logs_as_json(log_file, output_json_file):
    """Parse a FSDPTurbo ST run log into a baseline-comparable JSON file.

    Args:
        log_file: Path to the captured stdout log of the ST run.
        output_json_file: Path of the JSON summary to be written.
    """
    data = _parse_log(log_file)
    with open(output_json_file, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4)
    return data


def read_json(file):
    """Read a baseline or generated JSON file."""
    if os.path.exists(file):
        with open(file, encoding="utf-8-sig") as f:
            return json.load(f)
    raise FileNotFoundError(f"The file does not exist: {file}")


def main():
    """CLI: parse one or more ST run logs into a baseline JSON.

    With a single --log the run is written verbatim; with multiple --log values
    a median baseline is written (per-step median for loss/grad/time, per-rank
    median for memory). Use the multi-log form to build a stable baseline from
    several CI runs and dampen run-to-run noise (esp. step time).

    Usage:
        python acquire_json.py --log run.log --json out.json
        python acquire_json.py --log run1.log run2.log run3.log --json baseline.json
    """
    parser = argparse.ArgumentParser(
        description="Parse FSDPTurbo ST run log(s) into a baseline-comparable JSON summary."
    )
    parser.add_argument("--log", required=True, nargs="+", help="One or more captured ST run logs.")
    parser.add_argument("--json", dest="json_file", required=True, help="Path of the output JSON summary to write.")
    args = parser.parse_args()

    data = merge_logs_as_baseline(args.log, args.json_file)
    label = "verbatim" if len(args.log) == 1 else f"median of {len(args.log)} runs"
    print(f"Wrote {args.json_file}: {len(data[LOSS])} step(s), {label}.")


if __name__ == "__main__":
    main()

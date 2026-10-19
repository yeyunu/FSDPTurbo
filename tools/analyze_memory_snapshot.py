#!/usr/bin/env python3
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Analyze FSDPTurbo memory snapshot pickles (torch_npu / CUDA allocator dumps).

Works on the pickles written by the trainer's memory snapshot feature
(``run.memory_snapshot`` with ``enable: true``, or the always-on OOM trap
``OOM_SNAPSHOT_ENABLE=1``). Prints, per file:

1. A per-device summary of segment vs active bytes.
2. OOM events found in the device traces (the failed allocation with its
   stack), when the backend records them.
3. Active blocks grouped by allocation stack, largest first -- the stack that
   owns the most live memory, and the biggest single blocks.

Usage:
    python tools/analyze_memory_snapshot.py [--top 15] snapshot.pkl [more.pkl ...]
    python tools/analyze_memory_snapshot.py 'snapshots/rank*_error_*.pkl'
"""

import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict


def _fmt_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:,.2f} {unit}"
        size /= 1024
    return f"{size:,.2f} GiB"


def _block_frames(block):
    """Best-effort extraction of a block's allocation frames.

    The CUDA snapshot format nests them under ``history[-1].frames``; some
    builds put ``frames`` directly on the block.
    """
    frames = block.get("frames")
    if frames:
        return frames
    for record in reversed(block.get("history") or []):
        frames = record.get("frames") if isinstance(record, dict) else None
        if frames:
            return frames
    return []


def _stack_text(frames, max_frames=6):
    if not frames:
        return "<no python frames>"
    parts = []
    for frame in frames[-max_frames:]:
        name = frame.get("name") or frame.get("function") or "?"
        filename = os.path.basename(frame.get("filename") or frame.get("file") or "?")
        parts.append(f"{name} ({filename}:{frame.get('line', '?')})")
    return " <- ".join(parts)


def _is_active(block):
    # Snapshot states are e.g. "active_allocated" / "active_pending_free" /
    # "inactive"; a substring match would wrongly count "inactive".
    return str(block.get("state", "")).lower().startswith("active")


def analyze(path, top):
    with open(path, "rb") as f:
        snapshot = pickle.load(f)  # nosec B301 - trusted local snapshot produced by the trainer
    if not isinstance(snapshot, dict):
        print(f"== {path}: unexpected snapshot type {type(snapshot).__name__}, skipping ==")
        return

    print(f"===== {os.path.basename(path)} =====")

    # 1. Per-device segment/active summary.
    per_device = defaultdict(lambda: {"segments": 0, "total": 0, "active": 0})
    for seg in snapshot.get("segments", []):
        stats = per_device[seg.get("device", "?")]
        stats["segments"] += 1
        stats["total"] += seg.get("total_size", 0)
        for block in seg.get("blocks", []):
            if _is_active(block):
                stats["active"] += block.get("size", 0)
    for device, stats in sorted(per_device.items(), key=lambda kv: str(kv[0])):
        print(
            f"[device {device}] segments={stats['segments']} "
            f"total={_fmt_size(stats['total'])} active={_fmt_size(stats['active'])}"
        )

    # 2. OOM events recorded in the traces (the failing allocation itself).
    oom_events = []
    for traces in snapshot.get("device_traces", []):
        for event in traces or []:
            if "oom" in str(event.get("action", "")).lower():
                oom_events.append(event)
    if oom_events:
        for event in oom_events[-3:]:
            print(f"\n[OOM] requested {_fmt_size(event.get('size', 0))}")
            print(f"  stack: {_stack_text(event.get('frames') or [])}")
    else:
        print("\n[OOM] no oom events in traces (backend did not record them; see top stacks below)")

    # 3. Active bytes grouped by allocation stack, plus biggest single blocks.
    by_stack = defaultdict(lambda: [0, 0])  # bytes, count
    biggest = []
    for seg in snapshot.get("segments", []):
        for block in seg.get("blocks", []):
            if not _is_active(block):
                continue
            size = block.get("size", 0)
            by_stack[_stack_text(_block_frames(block))][0] += size
            by_stack[_stack_text(_block_frames(block))][1] += 1
            biggest.append((size, _stack_text(_block_frames(block), max_frames=3)))
    print(f"\nTop {top} allocation stacks by live bytes:")
    for stack, (total, count) in sorted(by_stack.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"  {_fmt_size(total):>14}  x{count:<5} {stack}")
    print(f"\nTop {top} single live blocks:")
    for size, stack in sorted(biggest, key=lambda item: -item[0])[:top]:
        print(f"  {_fmt_size(size):>14}  {stack}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshots", nargs="+", help="snapshot .pkl file(s) or glob")
    parser.add_argument("--top", type=int, default=15, help="number of stacks/blocks to print (default 15)")
    args = parser.parse_args()

    paths = []
    for pattern in args.snapshots:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded if expanded else [pattern])
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        parser.error(f"snapshot file(s) not found: {', '.join(missing)}")

    for path in paths:
        try:
            analyze(path, args.top)
        except Exception as error:  # noqa: BLE001 - report per-file, keep analyzing the rest
            print(f"===== {path}: failed to analyze: {error} =====", file=sys.stderr)


if __name__ == "__main__":
    main()

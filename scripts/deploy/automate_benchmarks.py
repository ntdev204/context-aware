#!/usr/bin/env python3
"""Automated Benchmark Tracker for Jetson

Runs pipeline benchmarks and appends results to a CSV history file.
Can be triggered manually on the Jetson.

Usage:
  python scripts/automate_benchmarks.py --note "Added temporal GRU k=3"
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
HISTORY_FILE = ROOT / "benchmarks" / "history.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--note", type=str, default="", help="Note for this benchmark run")
    return p.parse_args()


def run_benchmark(frames: int) -> dict:
    cmd = [sys.executable, "scripts/benchmark.py", "--frames", str(frames)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Benchmark failed:\n{result.stderr}")
        sys.exit(1)

    # Parse standard format from benchmark.py output
    lines = result.stdout.splitlines()
    metrics = {}
    for line in lines:
        if "Mean latency :" in line:
            metrics["mean_ms"] = float(line.split(":")[1].split("ms")[0].strip())
        elif "P99 latency  :" in line:
            metrics["p99_ms"] = float(line.split(":")[1].split("ms")[0].strip())
        elif "Mean FPS     :" in line:
            metrics["fps"] = float(line.split(":")[1].strip())

    return metrics


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running benchmark ({args.frames} frames)...")
    metrics = run_benchmark(args.frames)

    # Check if we need to write header
    write_header = not HISTORY_FILE.exists()

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_hash": get_git_hash(),
        "fps": metrics.get("fps", 0),
        "mean_ms": metrics.get("mean_ms", 0),
        "p99_ms": metrics.get("p99_ms", 0),
        "note": args.note,
    }

    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    print("\nBenchmark finished.")
    print(
        f"FPS: {record['fps']:.1f} | Mean: {record['mean_ms']:.1f}ms | P99: {record['p99_ms']:.1f}ms"
    )
    print(f"Saved to {HISTORY_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

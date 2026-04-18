#!/usr/bin/env python3
"""System health check — memory, GPU, temperature on Jetson.

Usage:
    python scripts/health_check.py          # one-shot report
    python scripts/health_check.py --watch  # continuous 5s updates
"""

from __future__ import annotations

import argparse
import subprocess
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true", help="Refresh every 5s")
    return p.parse_args()


def run_cmd(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return "N/A"


def get_memory_gb() -> tuple[float, float]:
    """Return (used_gb, total_gb) from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, v = line.split(":")
                info[k.strip()] = int(v.strip().split()[0])
        total_gb = info["MemTotal"] / 1024 / 1024
        free_gb = (info["MemFree"] + info["Buffers"] + info["Cached"]) / 1024 / 1024
        used_gb = total_gb - free_gb
        return used_gb, total_gb
    except Exception:
        return 0.0, 0.0


def get_gpu_usage() -> str:
    return run_cmd("cat /sys/devices/gpu.0/load 2>/dev/null || echo N/A")


def get_temp() -> str:
    return run_cmd("cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1")


def report() -> None:
    used, total = get_memory_gb()
    gpu = get_gpu_usage()
    temp = get_temp()
    ts = time.strftime("%H:%M:%S")

    print(f"\n[{ts}] === Jetson Health ===")
    print(f"  RAM : {used:.2f} GB / {total:.2f} GB  ({used / total * 100:.1f}%)")
    print(f"  GPU : {gpu}")
    print(f"  Temp: {temp}")
    print(f"  RAM budget: {'OK' if used < 3.5 else 'WARNING: > 3.5GB target'}")


def main():
    args = parse_args()
    if args.watch:
        print("Health check — Ctrl+C to stop")
        while True:
            report()
            time.sleep(5.0)
    else:
        report()


if __name__ == "__main__":
    main()

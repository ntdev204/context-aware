#!/usr/bin/env python3
"""Pipeline FPS and latency benchmark — runs on Jetson or dev machine.

Usage:
    python scripts/benchmark.py                    # 100 frames, USB cam
    python scripts/benchmark.py --frames 300       # longer run
    python scripts/benchmark.py --no-camera        # synthetic frames only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.navigation import ContextBuilder, HeuristicPolicy, SafetyMonitor
from src.perception import IntentCNN, ROIExtractor, Tracker, YOLODetector


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline benchmark")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--config", type=str, default=None)
    return p.parse_args()


def synthetic_frame(w=640, h=480):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def run_benchmark(args) -> None:
    cfg = load_config(args.config)

    per_cfg = cfg.section("perception")
    yolo = YOLODetector(
        model_path=per_cfg.get("yolo.model_path", "models/yolo/yolo11s.pt"),
        use_tensorrt=per_cfg.get("yolo.use_tensorrt", False),
    )
    yolo.load()

    tracker = Tracker()
    roi_ex = ROIExtractor()
    cnn = IntentCNN(model_path=per_cfg.get("cnn_intent.model_path", None))
    cnn.load()
    ctx_bld = ContextBuilder()
    policy = HeuristicPolicy()
    safety = SafetyMonitor()

    latencies = []

    print(f"Benchmarking {args.frames} frames...")
    for i in range(args.frames):
        frame = synthetic_frame()
        t0 = time.perf_counter()

        frame_det = yolo.detect(frame, frame_id=i)
        frame_det = tracker.update(frame_det, frame.shape)
        rois = roi_ex.extract(frame, frame_det)
        preds = cnn.predict_batch(rois)
        obs = ctx_bld.build(frame_det, preds)
        cmd = policy.decide(obs, frame_det, preds)
        cmd = safety.check(cmd, frame_det, preds)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    latencies_arr = np.array(latencies)
    print("\n=== Benchmark Results ===")
    print(f"Frames       : {args.frames}")
    print(f"Mean latency : {latencies_arr.mean():.1f} ms")
    print(f"P50 latency  : {np.percentile(latencies_arr, 50):.1f} ms")
    print(f"P95 latency  : {np.percentile(latencies_arr, 95):.1f} ms")
    print(f"P99 latency  : {np.percentile(latencies_arr, 99):.1f} ms")
    print(f"Max latency  : {latencies_arr.max():.1f} ms")
    print(f"Mean FPS     : {1000 / latencies_arr.mean():.1f}")
    print(f"Target: < 33.3ms (30 FPS) -- {'PASS' if latencies_arr.mean() < 33.3 else 'FAIL'}")


if __name__ == "__main__":
    run_benchmark(parse_args())

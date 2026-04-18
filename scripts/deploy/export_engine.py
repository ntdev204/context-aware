"""Export YOLO .pt to TensorRT .engine using Ultralytics built-in export.

Uses Ultralytics' Python TRT bindings directly -- no intermediate files required.

Usage:
    python scripts/export_engine.py models/yolo/yolo11s.pt [--fp16] [--workspace 2]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("export_engine")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO .pt → TensorRT .engine")
    parser.add_argument("model", help="Path to .pt file (e.g. models/yolo/yolo11s.pt)")
    parser.add_argument("--fp16", action="store_true", default=True, help="FP16 precision (default: True)")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[480, 640],
                        help="Input size as H W (default: 480 640 for 640x480 camera)")
    parser.add_argument("--workspace", type=int, default=2, help="TensorRT workspace in GB (default: 2)")
    args = parser.parse_args()

    pt_path = Path(args.model)
    if not pt_path.exists():
        log.error("Model not found: %s", pt_path)
        sys.exit(1)

    imgsz = args.imgsz if len(args.imgsz) > 1 else args.imgsz[0]   # [H, W] or scalar
    engine_path = pt_path.with_suffix(".engine")
    log.info("Exporting %s -> %s  [FP16=%s, workspace=%dGB, imgsz=%s]",
             pt_path.name, engine_path.name, args.fp16, args.workspace, imgsz)

    # Suppress Ultralytics' emoji-filled stdout (library prints directly, no logging API)
    os.environ.setdefault("YOLO_OFFLINE", "True")
    os.environ.setdefault("ULTRALYTICS_TELEMETRY", "0")
    import logging as _logging
    _logging.getLogger("ultralytics").setLevel(_logging.ERROR)

    from ultralytics import YOLO
    model = YOLO(str(pt_path))

    log.info("Running TensorRT export (this takes 3-10 min on Jetson Orin)...")
    model.export(
        format="engine",
        device=0,
        half=args.fp16,
        imgsz=imgsz,
        workspace=args.workspace,
        simplify=False,
        verbose=False,
    )

    if engine_path.exists():
        size_mb = engine_path.stat().st_size / (1024 ** 2)
        log.info("Engine saved: %s (%.1f MB)", engine_path, size_mb)
        os._exit(0)  # bypass TRT GC cleanup to avoid SIGABRT
    else:
        log.error("Export failed -- engine not found at %s", engine_path)
        os._exit(1)


if __name__ == "__main__":
    main()

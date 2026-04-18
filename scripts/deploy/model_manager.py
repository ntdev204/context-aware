#!/usr/bin/env python3
"""Model version registry: register, promote, and track model files.

Usage:
  python scripts/model_manager.py register --type cnn_intent --file models/cnn_intent/best.pt --version 1.0.0 --metrics '{"accuracy": 0.85}'
  python scripts/model_manager.py promote  --type cnn_intent --version 1.0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
MODEL_TYPES = ["yolo", "cnn_intent", "rl_policy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model registry manager")
    subs = parser.add_subparsers(dest="command", required=True)

    reg = subs.add_parser("register", help="Register a new model version")
    reg.add_argument("--type", choices=MODEL_TYPES, required=True)
    reg.add_argument("--file", type=Path, required=True, help="Path to .pt or .engine file")
    reg.add_argument("--version", required=True, help="e.g. 1.0.0 or v2-temporal")
    reg.add_argument("--metrics", type=str, default="{}", help="JSON string of metrics")

    prom = subs.add_parser("promote", help="Set a version as the active model")
    prom.add_argument("--type", choices=MODEL_TYPES, required=True)
    prom.add_argument("--version", required=True)

    return parser.parse_args()


def md5_file(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
    return md5.hexdigest()


def cmd_register(args: argparse.Namespace) -> None:
    if not args.file.exists():
        log.error("File not found: %s", args.file)
        sys.exit(1)

    model_dir = MODELS_DIR / args.type / args.version
    model_dir.mkdir(parents=True, exist_ok=True)

    dest = model_dir / args.file.name
    shutil.copy2(args.file, dest)

    metadata = {
        "model_type": args.type,
        "version": args.version,
        "filename": dest.name,
        "registered_at": datetime.now().isoformat(),
        "md5_hash": md5_file(dest),
        "metrics": json.loads(args.metrics),
    }

    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("Registered %s v%s -> %s (md5: %s)", args.type, args.version,
             dest.relative_to(ROOT), metadata["md5_hash"])


def cmd_promote(args: argparse.Namespace) -> None:
    target_dir = MODELS_DIR / args.type / args.version
    if not target_dir.exists():
        log.error("Version '%s' not found for type '%s'", args.version, args.type)
        sys.exit(1)

    with open(target_dir / "metadata.json") as f:
        meta = json.load(f)

    src = target_dir / meta["filename"]
    dest = MODELS_DIR / args.type / meta["filename"]

    for pattern in ("*.pt", "*.engine"):
        for old in (MODELS_DIR / args.type).glob(pattern):
            if old.is_file() and old != dest:
                old.unlink()

    shutil.copy2(src, dest)
    log.info("Promoted %s v%s -> %s", args.type, args.version, dest.relative_to(ROOT))


def main() -> None:
    args = parse_args()
    {"register": cmd_register, "promote": cmd_promote}[args.command](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export a trained temporal intent model for phase-2 deployment.

The training checkpoint remains the source of truth. This script writes a
TorchScript artifact plus a JSON metadata sidecar carrying ontology,
calibration, temporal-window, and optimization settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_intent_cnn import build_model  # noqa: E402
from src.perception.intent_labels import INTENT_NAMES, TRAINABLE_INTENT_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export temporal Intent CNN checkpoint to TorchScript + metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("models/cnn_intent/intent_v1.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/cnn_intent/intent_v1.ts"))
    parser.add_argument("--temporal-window", type=int, default=15)
    parser.add_argument("--quantize-dynamic", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError("Expected checkpoint dict with model_state_dict")
    return state


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    ckpt = _load_checkpoint(args.checkpoint, device)
    model = build_model(freeze_backbone_blocks=0).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    if args.quantize_dynamic:
        if device.type != "cpu":
            raise ValueError("--quantize-dynamic is CPU-only")
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    example = torch.zeros(1, args.temporal_window, 3, 256, 128, device=device)
    traced = torch.jit.trace(model, example)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))

    ckpt_meta = ckpt.get("metadata", {})
    metadata = {
        "exported_at": datetime.now().isoformat(),
        "source_checkpoint": str(args.checkpoint),
        "artifact": str(args.output),
        "format": "torchscript",
        "architecture": ckpt_meta.get("architecture", "mobilenetv3_small_tcn"),
        "temporal_window": args.temporal_window,
        "trainable_intent_names": list(TRAINABLE_INTENT_NAMES),
        "runtime_intent_names": list(INTENT_NAMES),
        "temperature": ckpt.get("temperature", ckpt_meta.get("temperature", 1.0)),
        "confidence_threshold": ckpt_meta.get("confidence_threshold", 0.55),
        "margin_threshold": ckpt_meta.get("margin_threshold", 0.12),
        "quantized_dynamic": bool(args.quantize_dynamic),
        "label_policy": "No FOLLOW/FOLLOWING; UNCERTAIN is abstain/review class.",
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported: {args.output}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()

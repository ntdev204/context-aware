#!/usr/bin/env python3
"""Build a phase-2 intent dataset manifest.

The manifest is the gate artifact before model export. It records ontology,
class counts, pending review items, track/session coverage, and whether the
dataset is ready for temporal Intent CNN training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.intent_labels import (  # noqa: E402
    INTENT_NAMES,
    REVIEW_ACCEPTED_STATUSES,
    TRAINABLE_INTENT_NAMES,
    canonical_label,
    is_trainable_label,
    needs_human_review,
)


def _load_metadata(dataset: Path) -> dict[str, dict]:
    meta_path = dataset / "metadata.jsonl"
    index: dict[str, dict] = {}
    if not meta_path.exists():
        return index
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            file_value = row.get("file")
            if file_value:
                index[str((dataset / file_value).resolve())] = row
    return index


def build_manifest(dataset: Path, temporal_window: int = 15) -> dict:
    meta_index = _load_metadata(dataset)
    images = [
        p
        for p in list(dataset.rglob("*.jpg")) + list(dataset.rglob("*.png"))
        if "reports" not in p.parts and "review_queue" not in p.parts
    ]

    class_counts: Counter[str] = Counter()
    trainable_counts: Counter[str] = Counter()
    review_pending: Counter[str] = Counter()
    tracks: dict[str, list[int]] = defaultdict(list)
    legacy_follow_count = 0

    for path in images:
        sidecar = meta_index.get(str(path.resolve()), {})
        raw_label = sidecar.get("label", path.parent.name)
        if str(raw_label).strip().upper() in {"FOLLOW", "FOLLOWING"}:
            legacy_follow_count += 1
        label = canonical_label(raw_label)
        class_counts[label] += 1

        review_status = str(sidecar.get("review_status") or "")
        if needs_human_review(label) and review_status not in REVIEW_ACCEPTED_STATUSES:
            review_pending[label] += 1
            continue
        if is_trainable_label(label):
            trainable_counts[label] += 1
            track_uid = str(
                sidecar.get("track_uid")
                or sidecar.get("session_id")
                or sidecar.get("tid")
                or path.stem
            )
            tracks[track_uid].append(int(sidecar.get("frame_id", 0)))

    short_tracks = {tid: len(frames) for tid, frames in tracks.items() if len(frames) < temporal_window}
    ready = (
        legacy_follow_count == 0
        and review_pending.get("ERRATIC", 0) == 0
        and sum(trainable_counts.values()) >= 500
        and len(tracks) >= 2
    )

    return {
        "generated_at": int(time.time()),
        "dataset": str(dataset),
        "ontology": {
            "runtime_intents": list(INTENT_NAMES),
            "trainable_intents": list(TRAINABLE_INTENT_NAMES),
            "removed": ["FOLLOW", "FOLLOWING"],
        },
        "temporal_window": temporal_window,
        "total_images": len(images),
        "class_counts": dict(class_counts),
        "trainable_counts": dict(trainable_counts),
        "review_pending": dict(review_pending),
        "legacy_follow_count": legacy_follow_count,
        "track_count": len(tracks),
        "short_track_count": len(short_tracks),
        "ready_for_phase2_training": ready,
        "gates": {
            "no_legacy_follow": legacy_follow_count == 0,
            "erratic_review_done": review_pending.get("ERRATIC", 0) == 0,
            "min_trainable_images_500": sum(trainable_counts.values()) >= 500,
            "track_split_possible": len(tracks) >= 2,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build intent dataset manifest")
    parser.add_argument("--dataset", type=Path, default=Path("intent_dataset"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--temporal-window", type=int, default=15)
    args = parser.parse_args()

    manifest = build_manifest(args.dataset, args.temporal_window)
    output = args.output or args.dataset / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {output}")
    print(f"Ready: {manifest['ready_for_phase2_training']}")


if __name__ == "__main__":
    main()

"""Auto-label ROI images using depth-aware and lateral displacement heuristics.

Applies robot motion compensation before labeling to handle moving robot cases.
"""

import argparse
import json
import logging
import math
import shutil
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
try:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.perception.intent_labels import INTENT_NAMES, canonical_label, needs_human_review
except Exception:  # pragma: no cover - keep standalone script runnable
    INTENT_NAMES = [
        "STATIONARY",
        "APPROACHING",
        "DEPARTING",
        "CROSSING",
        "ERRATIC",
        "UNCERTAIN",
    ]

    def canonical_label(label: str | None) -> str:
        label_up = str(label or "UNCERTAIN").strip().upper()
        return "UNCERTAIN" if label_up in {"FOLLOW", "FOLLOWING"} else label_up

    def needs_human_review(label: str | None) -> bool:
        return canonical_label(label) in {"UNCERTAIN", "ERRATIC"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants & Thresholds
FOCAL_LENGTH_PX = 554.0  # approx for AstraS depth camera
CAMERA_W = 640.0  # reference image width
CAMERA_H = 480.0

# Thresholds — ALL in per-second rates to be FPS/throttle independent
STATIONARY_THRESHOLD = 100   # mm/s  (total speed |d_depth/dt| + |d_cx_to_mm/dt|)
APPROACHING_THRESHOLD = 150  # mm/s
DEPARTING_THRESHOLD = 150    # mm/s

CROSSING_THRESHOLD = 50      # px/s
STAT_CX_THRESHOLD = 20       # px/s
ERRATIC_VAR_THRESHOLD = 25000  # (mm/s)^2 variance

WINDOW_SIZE = 5
LABELS = tuple(INTENT_NAMES)


def _empty_stats() -> dict[str, int]:
    stats = {label: 0 for label in LABELS}
    stats["short_track_skipped"] = 0
    return stats


def calculate_angle_to_person(cx: float, frame_w: float = CAMERA_W, fx: float = FOCAL_LENGTH_PX) -> float:
    """Angle (radians) from camera optical axis to person centre.

    Uses fx (actual scaled focal length) so the result is consistent with
    the cx-compensation that also scales fx by frame_w / CAMERA_W.
    """
    return math.atan2(cx - (frame_w / 2.0), fx)


def compensate_motion(
    dt: float,
    delta_depth_raw: float,
    delta_cx_raw: float,
    cx: float,
    vx: float,
    vy: float,
    vtheta: float,
    frame_w: float = CAMERA_W,
):
    """
    Remove ego-motion effect from raw depth and cx deltas.
    Returns rates: (depth_rate_mm_per_s, cx_rate_px_per_s)
    """
    # Scale focal length proportionally when camera runs at non-reference width.
    fx = FOCAL_LENGTH_PX * (frame_w / CAMERA_W)

    # 1. Depth compensation (convert robot speed m/s → mm/s via *1000)
    # Pass the scaled fx so angle and cx-compensation use the same intrinsics.
    angle_to_person = calculate_angle_to_person(cx, frame_w, fx)
    robot_depth_contrib = (
        (vx * math.cos(angle_to_person) + vy * math.sin(angle_to_person)) * dt * 1000.0
    )
    delta_depth_true = delta_depth_raw + robot_depth_contrib

    # 2. CX compensation (vtheta > 0 turns left → objects drift right)
    robot_cx_contrib = fx * vtheta * dt
    delta_cx_true = delta_cx_raw - robot_cx_contrib

    # Normalise to per-second rates so thresholds are FPS/throttle independent.
    rate_depth = delta_depth_true / dt   # mm/s
    rate_cx = delta_cx_true / dt         # px/s

    return rate_depth, rate_cx


def _safe_dataset_filename(row: dict[str, Any], src_path: Path) -> str:
    session_id = str(row.get("session_id") or row.get("_session_id") or src_path.parent.name)
    safe_session = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)
    tid = int(row.get("tid", row.get("track_id", -1)))
    frame_id = int(row.get("frame_id", 0))
    return f"{safe_session}_t{tid}_f{frame_id:06d}{src_path.suffix.lower()}"


def _bbox_distance_proxy(row: dict[str, Any]) -> float:
    bh = float(row.get("bh") or 0.0)
    frame_h = float(row.get("frame_h") or CAMERA_H)
    if bh <= 0 or frame_h <= 0:
        estimate = row.get("distance_estimate")
        if estimate is None:
            return 0.0
        return float(estimate)
    return max(0.0, 1.0 - (bh / frame_h))


def _write_labeled_sample(row: dict[str, Any], out_dir: Path) -> bool:
    src_path = row.get("_src_path")
    if not isinstance(src_path, Path) or not src_path.exists():
        return False

    label = canonical_label(row["label"])
    dest_dir = out_dir / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_name = _safe_dataset_filename(row, src_path)
    dest_path = dest_dir / dest_name
    shutil.copy2(src_path, dest_path)

    meta = {
        key: value
        for key, value in row.items()
        if not key.startswith("_") and key not in {"label"}
    }
    meta.update(
        {
            "file": f"{label}/{dest_name}",
            "source_file": row.get("file", src_path.name),
            "label": label,
            "label_source": row.get("label_source", "heuristic"),
            "review_status": row.get(
                "review_status",
                "needs_review" if needs_human_review(label) else "auto_accepted",
            ),
            "review_required": bool(needs_human_review(label)),
            "sequence_window": WINDOW_SIZE,
            "track_uid": row.get("_track_uid"),
            "d_depth": round(float(row.get("d_depth", 0.0)), 4),
            "d_cx": round(float(row.get("d_cx", 0.0)), 4),
            "depth_valid": bool(row.get("depth_valid", False)),
            "bbox_depth_delta": bool(row.get("bbox_depth_delta", False)),
        }
    )

    with open(out_dir / "metadata.jsonl", "a", encoding="utf-8") as jf:
        jf.write(json.dumps(meta, ensure_ascii=False) + "\n")

    if needs_human_review(label):
        review_dir = out_dir / "review_queue" / label
        review_dir.mkdir(parents=True, exist_ok=True)
        review_path = review_dir / dest_name
        if not review_path.exists():
            shutil.copy2(dest_path, review_path)
            review_ref = {
                "file": f"review_queue/{label}/{dest_name}",
                "dataset_file": f"{label}/{dest_name}",
                "label": label,
                "track_uid": row.get("_track_uid"),
                "frame_id": row.get("frame_id", 0),
                "reason": "low_certainty" if label == "UNCERTAIN" else "erratic_candidate",
                "review_status": "needs_review",
            }
            with open(out_dir / "review_queue" / "metadata.jsonl", "a", encoding="utf-8") as jf:
                jf.write(json.dumps(review_ref, ensure_ascii=False) + "\n")
    return True


def process_track(track_id: str, frames: list[dict[str, Any]], out_dir: Path) -> dict[str, int]:
    """Process a sequence of frames for a single track_id."""
    stats = _empty_stats()
    # Ensure ordered by timestamp
    frames.sort(key=lambda x: x["ts"])

    if len(frames) < WINDOW_SIZE:
        logger.debug("Track %s too short (%d frames), skipping.", track_id, len(frames))
        stats["short_track_skipped"] += len(frames)
        return stats
    # Build per-frame delta features
    for i in range(len(frames)):
        if i == 0:
            frames[i]["d_depth"] = 0.0
            frames[i]["d_cx"] = 0.0
            frames[i]["depth_valid"] = False  # no previous frame to diff against
            frames[i]["label"] = "UNCERTAIN"
            continue

        curr = frames[i]
        prev = frames[i - 1]

        dt = (curr["ts"] - prev["ts"]) / 1000.0
        if dt <= 0:
            dt = 0.033  # fallback to 30fps

        delta_cx_raw = curr["cx"] - prev["cx"]
        frame_w = float(curr.get("frame_w") or CAMERA_W)

        # Depth delta is only meaningful when BOTH samples have valid sensor readings.
        # dist_mm == 0 means the depth sensor returned no valid pixel for that ROI.
        curr_depth_ok = curr.get("dist_mm", 0) > 0
        prev_depth_ok = prev.get("dist_mm", 0) > 0
        depth_pair_valid = curr_depth_ok and prev_depth_ok

        if depth_pair_valid:
            delta_depth_raw = curr["dist_mm"] - prev["dist_mm"]
            vx = curr.get("vx", 0.0)
            vy = curr.get("vy", 0.0)
            vtheta = curr.get("vtheta", 0.0)
            # Returns (mm/s, px/s) — already divided by dt inside compensate_motion
            d_depth, d_cx = compensate_motion(
                dt, delta_depth_raw, delta_cx_raw, curr["cx"], vx, vy, vtheta, frame_w
            )
        else:
            # Bbox-height proxy: normalized [0,1] difference projected into a
            # heuristic depth-like scale for fallback labeling only. This is
            # not physically comparable to sensor mm/s and should be treated
            # as lower-confidence metadata when bbox_depth_delta=True.
            bbox_delta = _bbox_distance_proxy(prev) - _bbox_distance_proxy(curr)
            d_depth = (bbox_delta / dt) * 1000.0  # heuristic proxy, not true mm/s
            vtheta = curr.get("vtheta", 0.0)
            fx = FOCAL_LENGTH_PX * (frame_w / CAMERA_W)
            robot_cx_contrib = fx * vtheta * dt
            d_cx = (delta_cx_raw - robot_cx_contrib) / dt  # px/s

        curr["d_depth"] = d_depth   # mm/s (or bbox-proxy mm/s)
        curr["d_cx"] = d_cx         # px/s
        curr["depth_valid"] = depth_pair_valid
        curr["bbox_depth_delta"] = not depth_pair_valid

    # Apply labeling over sliding windows
    for i in range(WINDOW_SIZE - 1, len(frames)):
        window = frames[i - WINDOW_SIZE + 1 : i + 1]

        d_cxs = [f["d_cx"] for f in window]

        # Count how many frames in this window have valid depth pairs.
        # If the majority lack valid depth, depth-dependent labels are unreliable.
        valid_depth_rates = [
            float(f["d_depth"])
            for f in window
            if f.get("depth_valid", False)
        ]
        valid_depth_count = len(valid_depth_rates)
        depth_window_ok = valid_depth_count >= (WINDOW_SIZE // 2 + 1)  # strict majority

        # Never mix bbox proxy deltas with real sensor-depth deltas inside
        # depth-dependent rules. The proxy is only a fallback signal for future
        # heuristics when depth is unavailable.
        mean_delta_depth = float(np.mean(valid_depth_rates)) if valid_depth_rates else 0.0
        var_delta_depth = float(np.var(valid_depth_rates)) if valid_depth_rates else 0.0
        mean_delta_cx = np.mean([abs(dx) for dx in d_cxs])

        # Default: UNCERTAIN when depth is insufficient for reliable classification.
        label = "UNCERTAIN"

        if depth_window_ok:
            # Full rule set — depth-based signals are trustworthy.
            if var_delta_depth > ERRATIC_VAR_THRESHOLD:
                label = "ERRATIC"
            elif mean_delta_depth < -APPROACHING_THRESHOLD:
                label = "APPROACHING"
            elif mean_delta_depth > DEPARTING_THRESHOLD:
                label = "DEPARTING"
            elif mean_delta_cx > CROSSING_THRESHOLD:
                label = "CROSSING"
            elif abs(mean_delta_depth) < STATIONARY_THRESHOLD and mean_delta_cx < STAT_CX_THRESHOLD:
                label = "STATIONARY"
            else:
                # No FOLLOW/FOLLOWING class in the current system. Ambiguous
                # residual motion is held for review instead of becoming a
                # trainable label.
                label = "UNCERTAIN"
        else:
            # Depth unavailable — only lateral-motion rules are valid.
            if mean_delta_cx > CROSSING_THRESHOLD:
                label = "CROSSING"
            # STATIONARY cannot be confirmed without depth — leave as UNCERTAIN.

        frames[i]["label"] = label

    # Save to output directories based on label and persist sidecar metadata.
    for f in frames[WINDOW_SIZE - 1 :]:
        label = canonical_label(f["label"])
        f["label"] = label
        if _write_labeled_sample(f, out_dir):
            stats[label] += 1

    return stats


def extract_archives(archive_path: Path, output_dir: Path) -> None:
    """Extract a .zip or .tar.gz dataset archive into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()

    def safe_target(name: str) -> Path:
        target = (output_dir / name).resolve()
        if target != output_root and output_root not in target.parents:
            raise ValueError(f"Unsafe archive member path: {name}")
        return target

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                safe_target(member)
            archive.extractall(output_dir)
        return
    if archive_path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.issym() or member.islnk():
                    raise ValueError(f"Unsafe archive link member: {member.name}")
                safe_target(member.name)
            archive.extractall(output_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def run_autolabel(batch_dir: Path, output_dir: Path) -> dict[str, int]:
    """Main entrypoint for labeling a batch of ROIs."""
    logger.info(f"Scanning batch dir: {batch_dir}")

    tracks = defaultdict(list)
    jsonl_files = list(batch_dir.glob("**/metadata.jsonl"))
    stats = _empty_stats()

    if not jsonl_files:
        logger.warning(f"No metadata.jsonl found in {batch_dir}")
        return stats

    for jf in jsonl_files:
        parent = jf.parent
        fallback_session_id = parent.name
        with open(jf, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

                # Check path
                img_path = parent / data["file"]
                if img_path.exists():
                    session_id = str(data.get("session_id") or fallback_session_id)
                    data["_session_id"] = session_id
                    data["_src_path"] = img_path
                    data["_track_uid"] = f"{session_id}:t{data['tid']}"
                    tracks[data["_track_uid"]].append(data)

    logger.info(
        f"Found {len(tracks)} separate tracks across {sum(len(t) for t in tracks.values())} ROIs"
    )

    for tid, frames in tracks.items():
        track_stats = process_track(tid, frames, output_dir)
        for key, value in track_stats.items():
            stats[key] = stats.get(key, 0) + value

    logger.info(f"Labeling completed. Output to {output_dir}")
    return stats


def autolabel(
    input_dir: Path,
    output_dir: Path,
    threshold_px: int | None = None,
    lookahead: int | None = None,
    min_track_len: int | None = None,
    move: bool = False,
) -> dict[str, int]:
    """Compatibility wrapper used by auto_watcher.py."""
    del threshold_px, lookahead, min_track_len, move
    return run_autolabel(Path(input_dir), Path(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Depth & Lateral aware Intent ROI Auto-labeling")
    parser.add_argument(
        "--batch-dir",
        type=str,
        required=True,
        help="Directory containing incoming ROIs and metadata.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output intent_dataset auto directory"
    )

    args = parser.parse_args()
    run_autolabel(Path(args.batch_dir), Path(args.output_dir))

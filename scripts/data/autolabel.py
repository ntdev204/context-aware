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
STATIONARY_THRESHOLD = 100  # mm/s  (total speed |d_depth/dt| + |d_cx_to_mm/dt|)
APPROACHING_THRESHOLD = 150  # mm/s
DEPARTING_THRESHOLD = 150  # mm/s

CROSSING_THRESHOLD = 50  # px/s
STAT_CX_THRESHOLD = 20  # px/s
ERRATIC_VAR_THRESHOLD = 25000  # (mm/s)^2 variance
ERRATIC_MIN_RATE = 150  # mm/s, ignore tiny sign flips from sensor noise
ERRATIC_MIN_SIGN_CHANGES = 2

LABELS = tuple(INTENT_NAMES)
TRACK_FRAME_DIR = "_tracks"


def label_dir_name(label: str | None) -> str:
    return canonical_label(label).lower()


def _safe_text(value: object, default: str = "unknown") -> str:
    text = str(value if value is not None else default)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    return safe or default


def _empty_stats() -> dict[str, int]:
    return {label: 0 for label in LABELS}


def calculate_angle_to_person(
    cx: float, frame_w: float = CAMERA_W, fx: float = FOCAL_LENGTH_PX
) -> float:
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
    rate_depth = delta_depth_true / dt  # mm/s
    rate_cx = delta_cx_true / dt  # px/s

    return rate_depth, rate_cx


def _safe_dataset_filename(row: dict[str, Any], src_path: Path) -> str:
    session_id = str(row.get("session_id") or row.get("_session_id") or src_path.parent.name)
    safe_session = _safe_text(session_id)
    safe_track = _safe_text(row.get("tid", row.get("track_id", row.get("_track_uid"))))
    frame_id = int(row.get("frame_id", 0))
    return f"{safe_session}_t{safe_track}_f{frame_id:06d}{src_path.suffix.lower()}"


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
    label_dir = label_dir_name(label)
    dest_dir = out_dir / label_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_name = _safe_dataset_filename(row, src_path)
    dest_path = dest_dir / dest_name
    shutil.copy2(src_path, dest_path)

    meta = {
        key: value for key, value in row.items() if not key.startswith("_") and key not in {"label"}
    }
    meta.update(
        {
            "file": f"{label_dir}/{dest_name}",
            "source_file": row.get("file", src_path.name),
            "label": label,
            "label_source": row.get("label_source", "heuristic"),
            "review_status": row.get(
                "review_status",
                "needs_review" if needs_human_review(label) else "auto_accepted",
            ),
            "review_required": bool(needs_human_review(label)),
            "sample_type": "track_representative",
            "frame_count": int(row.get("sequence_frame_count", 1)),
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
        review_dir = out_dir / "review_queue" / label_dir
        review_dir.mkdir(parents=True, exist_ok=True)
        review_path = review_dir / dest_name
        if not review_path.exists():
            shutil.copy2(dest_path, review_path)
            review_ref = {
                "file": f"review_queue/{label_dir}/{dest_name}",
                "dataset_file": f"{label_dir}/{dest_name}",
                "label": label,
                "track_uid": row.get("_track_uid"),
                "frame_id": row.get("frame_id", 0),
                "reason": "low_certainty" if label == "UNCERTAIN" else "erratic_candidate",
                "review_status": "needs_review",
            }
            with open(out_dir / "review_queue" / "metadata.jsonl", "a", encoding="utf-8") as jf:
                jf.write(json.dumps(review_ref, ensure_ascii=False) + "\n")
    return True


def _copy_sequence_frame(row: dict[str, Any], out_dir: Path) -> str | None:
    src_path = row.get("_src_path")
    if not isinstance(src_path, Path) or not src_path.exists():
        return None
    safe_track = _safe_text(row.get("_track_uid") or row.get("track_uid"))
    frame_id = int(row.get("frame_id", 0))
    dest_dir = out_dir / TRACK_FRAME_DIR / safe_track
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_name = f"f{frame_id:06d}{src_path.suffix.lower()}"
    dest_path = dest_dir / dest_name
    if not dest_path.exists():
        shutil.copy2(src_path, dest_path)
    return f"{TRACK_FRAME_DIR}/{safe_track}/{dest_name}"


def _sequence_direction(label: str, window: list[dict[str, Any]]) -> tuple[float, float]:
    if label == "APPROACHING":
        return 0.0, -0.6
    if label == "DEPARTING":
        return 0.0, 0.6
    if label == "CROSSING":
        start_cx = float(window[0].get("cx", 0.0))
        end_cx = float(window[-1].get("cx", start_cx))
        return (0.8 if end_cx >= start_cx else -0.8), 0.0
    return 0.0, 0.0


def _sequence_confidence(label: str, window: list[dict[str, Any]]) -> float:
    if label == "UNCERTAIN":
        return 0.0
    if label == "ERRATIC":
        return 0.55

    depth_valid_count = sum(1 for row in window if row.get("depth_valid", False))
    depth_ratio = depth_valid_count / max(len(window), 1)
    d_depths = [float(row.get("d_depth", 0.0)) for row in window if row.get("depth_valid", False)]
    d_cxs = [abs(float(row.get("d_cx", 0.0))) for row in window]
    mean_depth = float(np.mean(d_depths)) if d_depths else 0.0
    mean_cx = float(np.mean(d_cxs)) if d_cxs else 0.0

    if label == "APPROACHING":
        margin = max(0.0, (-mean_depth - APPROACHING_THRESHOLD) / APPROACHING_THRESHOLD)
        return min(0.98, 0.55 + 0.25 * depth_ratio + 0.20 * margin)
    if label == "DEPARTING":
        margin = max(0.0, (mean_depth - DEPARTING_THRESHOLD) / DEPARTING_THRESHOLD)
        return min(0.98, 0.55 + 0.25 * depth_ratio + 0.20 * margin)
    if label == "CROSSING":
        margin = max(0.0, (mean_cx - CROSSING_THRESHOLD) / CROSSING_THRESHOLD)
        return min(0.95, 0.50 + 0.20 * depth_ratio + 0.25 * margin)
    if label == "STATIONARY":
        return min(0.95, 0.55 + 0.30 * depth_ratio)
    return 0.5


def _write_sequence_sample(
    window: list[dict[str, Any]], row: dict[str, Any], out_dir: Path
) -> bool:
    label = canonical_label(row["label"])
    frame_paths = []
    for item in window:
        rel_path = _copy_sequence_frame(item, out_dir)
        if rel_path is None:
            return False
        frame_paths.append(rel_path)

    dx, dy = _sequence_direction(label, window)
    confidence = _sequence_confidence(label, window)
    review_required = bool(needs_human_review(label) or confidence < 0.6)
    frame_ids = [int(item.get("frame_id", 0)) for item in window]
    track_uid = row.get("_track_uid") or row.get("track_uid")
    sequence_id = f"{_safe_text(track_uid)}_f{frame_ids[0]:06d}_{frame_ids[-1]:06d}"
    depth_valid_count = sum(1 for item in window if item.get("depth_valid", False))

    meta = {
        "sequence_id": sequence_id,
        "files": frame_paths,
        "label": label,
        "label_source": row.get("label_source", "heuristic_sequence"),
        "review_status": (
            "needs_review" if review_required else row.get("review_status", "auto_accepted")
        ),
        "review_required": review_required,
        "confidence": round(confidence, 4),
        "frame_count": len(frame_paths),
        "sample_policy": "whole_track_k",
        "track_uid": track_uid,
        "session_id": row.get("session_id") or row.get("_session_id"),
        "start_frame_id": frame_ids[0],
        "end_frame_id": frame_ids[-1],
        "start_ts": window[0].get("ts"),
        "end_ts": window[-1].get("ts"),
        "depth_valid_ratio": round(depth_valid_count / max(len(window), 1), 4),
        "mean_d_depth": round(float(np.mean([item.get("d_depth", 0.0) for item in window])), 4),
        "mean_abs_d_cx": round(float(np.mean([abs(item.get("d_cx", 0.0)) for item in window])), 4),
        "dx": dx,
        "dy": dy,
        "schema": "intent_sequence_v1",
    }

    with open(out_dir / "sequence_manifest.jsonl", "a", encoding="utf-8") as jf:
        jf.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return True


def _count_motion_sign_changes(values: list[float], min_rate: float = ERRATIC_MIN_RATE) -> int:
    signs: list[int] = []
    for value in values:
        if abs(value) < min_rate:
            continue
        signs.append(1 if value > 0 else -1)
    return sum(1 for prev, curr in zip(signs, signs[1:]) if prev != curr)


def _label_whole_track(frames: list[dict[str, Any]]) -> str:
    """Assign one intent label to the full visible lifetime of a track."""
    if len(frames) < 2:
        return "UNCERTAIN"

    motion_rows = frames[1:]
    d_cxs = [float(f.get("d_cx", 0.0)) for f in motion_rows]
    valid_depth_rates = [
        float(f.get("d_depth", 0.0)) for f in motion_rows if f.get("depth_valid", False)
    ]
    valid_depth_count = len(valid_depth_rates)
    depth_needed = max(1, len(motion_rows) // 2 + 1)
    depth_window_ok = valid_depth_count >= depth_needed

    mean_delta_depth = float(np.mean(valid_depth_rates)) if valid_depth_rates else 0.0
    var_delta_depth = float(np.var(valid_depth_rates)) if valid_depth_rates else 0.0
    mean_delta_cx = float(np.mean([abs(dx) for dx in d_cxs])) if d_cxs else 0.0

    if depth_window_ok:
        sign_changes = _count_motion_sign_changes(valid_depth_rates)
        if var_delta_depth > ERRATIC_VAR_THRESHOLD and sign_changes >= ERRATIC_MIN_SIGN_CHANGES:
            return "ERRATIC"
        if mean_delta_depth < -APPROACHING_THRESHOLD:
            return "APPROACHING"
        if mean_delta_depth > DEPARTING_THRESHOLD:
            return "DEPARTING"
        if mean_delta_cx > CROSSING_THRESHOLD:
            return "CROSSING"
        if abs(mean_delta_depth) < STATIONARY_THRESHOLD and mean_delta_cx < STAT_CX_THRESHOLD:
            return "STATIONARY"
        return "UNCERTAIN"

    if mean_delta_cx > CROSSING_THRESHOLD:
        return "CROSSING"
    return "UNCERTAIN"


def process_track(track_id: str, frames: list[dict[str, Any]], out_dir: Path) -> dict[str, int]:
    """Process a sequence of frames for a single track_id."""
    stats = _empty_stats()
    # Ensure ordered by timestamp
    frames.sort(key=lambda x: x["ts"])

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

        curr["d_depth"] = d_depth  # mm/s (or bbox-proxy mm/s)
        curr["d_cx"] = d_cx  # px/s
        curr["depth_valid"] = depth_pair_valid
        curr["bbox_depth_delta"] = not depth_pair_valid

    label = canonical_label(_label_whole_track(frames))
    representative = frames[-1]
    representative["label"] = label
    representative["label_source"] = "heuristic_whole_track"
    representative["sequence_frame_count"] = len(frames)

    # Save one sequence-level sample per track. The labeled image folders keep
    # only a representative frame for quick visual exploration; training reads
    # the full K-frame file list from sequence_manifest.jsonl.
    if _write_labeled_sample(representative, out_dir):
        stats[label] += 1
    _write_sequence_sample(frames, representative, out_dir)

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


def _array_value(values: Any, idx: int, default: Any = 0) -> Any:
    if isinstance(values, list) and idx < len(values):
        return values[idx]
    return default


def _rows_from_track_meta(meta_path: Path) -> list[dict[str, Any]]:
    """Convert one track_sequence_v1 meta.json into autolabel frame rows."""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Skipping unreadable track metadata: %s", meta_path)
        return []

    track_dir = meta_path.parent
    frames = meta.get("frames") or []
    if not isinstance(frames, list):
        return []

    source_session_id = str(
        meta.get("source_session_id") or track_dir.parents[1].name
        if len(track_dir.parents) > 1
        else meta.get("session_id", "session")
    )
    session_id = str(meta.get("session_id") or track_dir.parent.name)
    track_id = str(meta.get("track_id") or track_dir.name)
    source_track_id = meta.get("source_track_id", track_id)
    track_uid = f"{source_session_id}:{session_id}:{track_id}"

    rows: list[dict[str, Any]] = []
    for idx, frame_rel in enumerate(frames):
        src_path = track_dir / str(frame_rel)
        if not src_path.exists():
            continue

        frame_id = int(_array_value(meta.get("frame_ids"), idx, idx))
        ts = float(_array_value(meta.get("timestamps"), idx, idx * 33.0))
        frame_w = int(meta.get("frame_w") or CAMERA_W)
        frame_h = int(meta.get("frame_h") or CAMERA_H)
        bh = int(_array_value(meta.get("bh"), idx, 0))
        distance_estimate = max(0.0, 1.0 - (bh / frame_h)) if frame_h > 0 and bh > 0 else 0.0

        rows.append(
            {
                "file": str(frame_rel).replace("\\", "/"),
                "tid": source_track_id,
                "track_id": track_id,
                "frame_id": frame_id,
                "ts": ts,
                "cx": float(_array_value(meta.get("cx"), idx, CAMERA_W / 2.0)),
                "cy": float(_array_value(meta.get("cy"), idx, CAMERA_H / 2.0)),
                "bw": int(_array_value(meta.get("bw"), idx, 0)),
                "bh": bh,
                "frame_w": frame_w,
                "frame_h": frame_h,
                "dist_mm": float(_array_value(meta.get("dist_mm"), idx, 0.0)),
                "distance_estimate": distance_estimate,
                "vx": float(_array_value(meta.get("vx"), idx, 0.0)),
                "vy": float(_array_value(meta.get("vy"), idx, 0.0)),
                "vtheta": float(_array_value(meta.get("vtheta"), idx, 0.0)),
                "session_id": source_session_id,
                "sequence_id": session_id,
                "track_uid": track_uid,
                "schema": "track_sequence_v1",
                "_session_id": source_session_id,
                "_src_path": src_path,
                "_track_uid": track_uid,
            }
        )
    return rows


def _load_track_sequence_rows(batch_dir: Path) -> dict[str, list[dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for meta_path in batch_dir.glob("**/track_*/meta.json"):
        for row in _rows_from_track_meta(meta_path):
            tracks[row["_track_uid"]].append(row)
    return tracks


def _load_jsonl_rows(batch_dir: Path) -> dict[str, list[dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for jf in batch_dir.glob("**/metadata.jsonl"):
        parent = jf.parent
        fallback_session_id = parent.name
        with open(jf, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

                img_path = parent / data["file"]
                if img_path.exists():
                    session_id = str(data.get("session_id") or fallback_session_id)
                    track_id = data.get("tid", data.get("track_id", "unknown"))
                    data["_session_id"] = session_id
                    data["_src_path"] = img_path
                    data["_track_uid"] = f"{session_id}:t{track_id}"
                    tracks[data["_track_uid"]].append(data)
    return tracks


def run_autolabel(batch_dir: Path, output_dir: Path) -> dict[str, int]:
    """Main entrypoint for labeling a batch of ROIs."""
    logger.info(f"Scanning batch dir: {batch_dir}")

    tracks = defaultdict(list)
    stats = _empty_stats()

    for track_uid, rows in _load_jsonl_rows(batch_dir).items():
        tracks[track_uid].extend(rows)
    for track_uid, rows in _load_track_sequence_rows(batch_dir).items():
        tracks[track_uid].extend(rows)

    if not tracks:
        logger.warning(f"No metadata.jsonl or track_sequence_v1 meta.json found in {batch_dir}")
        return stats

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
        help="Directory containing incoming ROIs as metadata.jsonl or track_sequence_v1 meta.json",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output intent_dataset auto directory"
    )

    args = parser.parse_args()
    run_autolabel(Path(args.batch_dir), Path(args.output_dir))

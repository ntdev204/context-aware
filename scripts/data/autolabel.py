"""Auto-label ROI images using depth-aware and lateral displacement heuristics.

Applies robot motion compensation before labeling to handle moving robot cases.
"""

import argparse
import json
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants & Thresholds
FOCAL_LENGTH_PX = 554.0   # approx for AstraS depth camera
CAMERA_W = 640.0          # reference image width
CAMERA_H = 480.0

STATIONARY_THRESHOLD = 100     # mm
APPROACHING_THRESHOLD = 150    # mm/s
DEPARTING_THRESHOLD = 150      # mm/s

CROSSING_THRESHOLD = 50        # px/s
STAT_CX_THRESHOLD = 20         # px/s
ERRATIC_VAR_THRESHOLD = 25000  # mm^2 variance

WINDOW_SIZE = 5


def calculate_angle_to_person(cx: float) -> float:
    """Calculate angle (radians) from camera center to person."""
    return math.atan2(cx - (CAMERA_W / 2.0), FOCAL_LENGTH_PX)


def compensate_motion(
    dt: float,
    delta_depth_raw: float,
    delta_cx_raw: float,
    cx: float,
    vx: float,
    vy: float,
    vtheta: float
):
    """
    Remove effect of robot's ego-motion from the observed delta_depth and delta_cx.
    Returns: (delta_depth_true, delta_cx_true)
    """
    # 1. Depth compensation
    angle_to_person = calculate_angle_to_person(cx)
    # Robot moving forward (vx) reduces depth.
    # We multiply by 1000 to convert robot speed (m/s) to mm/s.
    robot_depth_contrib = (vx * math.cos(angle_to_person) + vy * math.sin(angle_to_person)) * dt * 1000.0
    delta_depth_true = delta_depth_raw + robot_depth_contrib

    # 2. CX compensation
    # Robot rotating (vtheta) shifts pixels horizontally.
    # vtheta > 0 (turn left) makes objects move right (+cx).
    robot_cx_contrib = FOCAL_LENGTH_PX * vtheta * dt
    delta_cx_true = delta_cx_raw - robot_cx_contrib

    return delta_depth_true, delta_cx_true


def process_track(track_id: str, frames: list[dict[str, Any]], out_dir: Path):
    """Process a sequence of frames for a single track_id."""
    # Ensure ordered by timestamp
    frames.sort(key=lambda x: x["ts"])

    if len(frames) < WINDOW_SIZE:
        logger.debug(f"Track {track_id} too short ({len(frames)} frames), skipping.")
        return

    # To store true features and rolling windows
    for i in range(len(frames)):
        if i == 0:
            frames[i]["d_depth"] = 0.0
            frames[i]["d_cx"] = 0.0
            frames[i]["label"] = "UNCERTAIN"
            continue

        curr = frames[i]
        prev = frames[i-1]

        dt = (curr["ts"] - prev["ts"]) / 1000.0
        if dt <= 0:
            dt = 0.033  # fallback to 30fps

        delta_depth_raw = curr["dist_mm"] - prev["dist_mm"]
        delta_cx_raw = curr["cx"] - prev["cx"]

        vx = curr.get("vx", 0.0)
        vy = curr.get("vy", 0.0)
        vtheta = curr.get("vtheta", 0.0)

        d_depth, d_cx = compensate_motion(dt, delta_depth_raw, delta_cx_raw, curr["cx"], vx, vy, vtheta)

        curr["d_depth"] = d_depth
        curr["d_cx"] = d_cx

    # Apply labeling over sliding windows
    for i in range(WINDOW_SIZE - 1, len(frames)):
        window = frames[i - WINDOW_SIZE + 1 : i + 1]

        d_depths = [f["d_depth"] for f in window]
        d_cxs = [f["d_cx"] for f in window]

        mean_delta_depth = np.mean(d_depths)
        var_delta_depth = np.var(d_depths)
        mean_delta_cx = np.mean([abs(dx) for dx in d_cxs])

        label = "FOLLOWING" # Residual class

        # Rule-based classification
        if var_delta_depth > ERRATIC_VAR_THRESHOLD:
            label = "ERRATIC"  # Needs human review
        elif mean_delta_depth < -APPROACHING_THRESHOLD:
            label = "APPROACHING"
        elif mean_delta_depth > DEPARTING_THRESHOLD:
            label = "DEPARTING"
        elif mean_delta_cx > CROSSING_THRESHOLD:
            label = "CROSSING"
        elif abs(mean_delta_depth) < STATIONARY_THRESHOLD and mean_delta_cx < STAT_CX_THRESHOLD:
            label = "STATIONARY"

        frames[i]["label"] = label

    # Save to output directories based on label
    for f in frames[WINDOW_SIZE - 1:]:
        label = f["label"]
        file_path = f.get("_src_path")
        if file_path and file_path.exists():
            dest_dir = out_dir / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_dir / file_path.name)

def run_autolabel(batch_dir: Path, output_dir: Path):
    """Main entrypoint for labeling a batch of ROIs."""
    logger.info(f"Scanning batch dir: {batch_dir}")

    tracks = defaultdict(list)
    jsonl_files = list(batch_dir.glob("**/metadata.jsonl"))

    if not jsonl_files:
        logger.warning(f"No metadata.jsonl found in {batch_dir}")
        return

    for jf in jsonl_files:
        parent = jf.parent
        with open(jf, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

                # Check path
                img_path = parent / data["file"]
                if img_path.exists():
                    data["_src_path"] = img_path
                    tracks[str(data["tid"])].append(data)

    logger.info(f"Found {len(tracks)} separate tracks across {sum(len(t) for t in tracks.values())} ROIs")

    for tid, frames in tracks.items():
        process_track(tid, frames, output_dir)

    logger.info(f"Labeling completed. Output to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Depth & Lateral aware Intent ROI Auto-labeling")
    parser.add_argument("--batch-dir", type=str, required=True, help="Directory containing incoming ROIs and metadata.jsonl")
    parser.add_argument("--output-dir", type=str, required=True, help="Output intent_dataset auto directory")

    args = parser.parse_args()
    run_autolabel(Path(args.batch_dir), Path(args.output_dir))

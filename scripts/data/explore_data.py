"""Experience Data Explorer + Quality Validator.

Reads HDF5 session files produced by ExperienceBuffer and generates:
  1. Summary statistics  (frame count, duration, FPS, action distribution)
  2. Quality flags       (missing depth, zero-action, duplicate frames, outliers)
  3. Sample images       (first / last / random N frames saved as JPEG grid)
  4. Plots               (obs histogram, action distribution, distance over time)

Usage:
    # Single file
    python scripts/data/explore_data.py logs/experience/session_abc123.h5

    # Whole directory (aggregated report)
    python scripts/data/explore_data.py logs/experience/

    # Show sample images
    python scripts/data/explore_data.py logs/experience/ --samples 16

    # Auto-fix: copy clean frames into a new file (removes flagged ones)
    python scripts/data/explore_data.py logs/experience/ --export-clean logs/clean/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

# Helpers


def _load_file(path: Path) -> dict:
    """Load all frames from one HDF5 session into a list of dicts."""
    frames = []
    with h5py.File(path, "r", locking=False) as f:
        for key in sorted(f.keys()):
            grp = f[key]
            try:
                frames.append(
                    {
                        "key": key,
                        "frame_id": int(grp.attrs["frame_id"]),
                        "timestamp": float(grp.attrs["timestamp"]),
                        "wall_time": float(grp.attrs["wall_time"]),
                        "session_id": grp.attrs["session_id"],
                        "vx": float(grp.attrs["vx"]),
                        "vy": float(grp.attrs["vy"]),
                        "vtheta": float(grp.attrs["vtheta"]),
                        "battery": float(grp.attrs["battery"]),
                        "observation": grp["observation"][:],
                        "action": grp["action"][:],
                        "intent_classes": grp["intent_classes"][:],
                        "intent_confs": grp["intent_confs"][:],
                        "person_distances": grp["person_distances"][:],
                        "distance_sources": [
                            s.decode() if isinstance(s, bytes) else s
                            for s in grp["distance_sources"][:]
                        ],
                        "image_jpeg": grp["image_jpeg"][:].tobytes(),
                    }
                )
            except Exception as exc:
                print(f"  [WARN] Skipping frame {key}: {exc}")
    return frames


def _collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.h5"))


# Quality checks

ACTION_MODES = ["CRUISE", "CAUTIOUS", "AVOID", "RESERVED", "STOP"]


def _quality_flags(frames: list[dict]) -> list[dict]:
    """Return list of {frame_id, key, flags[]} for every flagged frame."""
    issues = []
    prev_ts = None
    seen_ids = set()

    for f in frames:
        flags = []

        # Duplicate frame_id
        if f["frame_id"] in seen_ids:
            flags.append("DUPLICATE_FRAME_ID")
        seen_ids.add(f["frame_id"])

        # Timestamp gap > 200 ms (dropped frames burst)
        if prev_ts is not None and (f["timestamp"] - prev_ts) > 0.2:
            flags.append(f"TIMESTAMP_GAP_{f['timestamp'] - prev_ts:.2f}s")
        prev_ts = f["timestamp"]

        # All-zero observation (sensor failure)
        if np.all(f["observation"] == 0):
            flags.append("ZERO_OBSERVATION")

        # All-zero action → robot never moved (useless for RL)
        if np.all(f["action"] == 0):
            flags.append("ZERO_ACTION")

        # Observation contains NaN/Inf
        if not np.all(np.isfinite(f["observation"])):
            flags.append("NONFINITE_OBSERVATION")

        # Battery critical (< 10%)
        if 0 < f["battery"] < 10.0:
            flags.append(f"LOW_BATTERY_{f['battery']:.1f}pct")

        # All depth sources are bbox (no real depth data collected)
        if all(s == "bbox" for s in f["distance_sources"]) and len(f["distance_sources"]) > 0:
            flags.append("NO_DEPTH_COVERAGE")

        # Person extremely close — safety stop frame (still useful, but label it)
        for d in f["person_distances"]:
            if 0 < d < 0.5:
                flags.append(f"HARD_STOP_DISTANCE_{d:.2f}m")
                break

        if flags:
            issues.append({"frame_id": f["frame_id"], "key": f["key"], "flags": flags})

    return issues


# Statistics


def _statistics(frames: list[dict]) -> dict:
    if not frames:
        return {}

    timestamps = [f["timestamp"] for f in frames]
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    fps = len(frames) / duration if duration > 0 else 0.0

    actions = np.stack([f["action"] for f in frames])  # (N, 7)
    obs = np.stack([f["observation"] for f in frames])  # (N, 104)

    # Mode one-hot → index
    mode_idxs = np.argmax(actions[:, 2:], axis=1)
    mode_counts = {ACTION_MODES[i]: int(np.sum(mode_idxs == i)) for i in range(5)}

    batteries = [f["battery"] for f in frames]
    distances = [d for f in frames for d in f["person_distances"] if d > 0]

    depth_frames = sum(1 for f in frames if any(s == "depth" for s in f["distance_sources"]))

    return {
        "total_frames": len(frames),
        "duration_s": round(duration, 1),
        "avg_fps": round(fps, 1),
        "obs_dim": obs.shape[1],
        "action_dim": actions.shape[1],
        "obs_mean": float(np.mean(obs)),
        "obs_std": float(np.std(obs)),
        "obs_has_nan": bool(not np.all(np.isfinite(obs))),
        "action_mean": float(np.mean(actions[:, 0])),  # velocity_scale
        "action_std": float(np.std(actions[:, 0])),
        "mode_distribution": mode_counts,
        "battery_min": round(min(batteries), 1) if batteries else None,
        "battery_max": round(max(batteries), 1) if batteries else None,
        "depth_frame_pct": round(depth_frames / len(frames) * 100, 1),
        "person_detections": sum(len(f["person_distances"]) for f in frames),
        "avg_person_distance_m": round(float(np.mean(distances)), 2) if distances else None,
    }


# Image sampler


def _save_sample_grid(frames: list[dict], out_path: Path, n: int = 16) -> None:
    """Save a N-image contact sheet as JPEG for quick visual inspection."""
    try:
        import cv2
    except ImportError:
        print("  [SKIP] opencv-python not installed — skipping image grid")
        return

    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    imgs = []
    th, tw = 120, 160  # thumbnail size

    for idx in indices:
        raw = np.frombuffer(frames[idx]["image_jpeg"], dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((th, tw, 3), dtype=np.uint8)
        img = cv2.resize(img, (tw, th))

        # Overlay frame id
        fid = frames[idx]["frame_id"]
        cv2.putText(img, f"#{fid}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        imgs.append(img)

    # Arrange into a grid (4 cols)
    cols = 4
    rows = (n + cols - 1) // cols
    grid_h, grid_w = rows * th, cols * tw
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        grid[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = img

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"   Sample grid saved → {out_path}")


# Clean export


def _export_clean(frames: list[dict], issues: list[dict], out_dir: Path, session_id: str) -> None:
    """Copy non-flagged frames into a new HDF5 file in out_dir."""
    import gc

    bad_keys = {
        i["key"]
        for i in issues
        if any(
            f in ("ZERO_OBSERVATION", "NONFINITE_OBSERVATION", "DUPLICATE_FRAME_ID")
            for f in i["flags"]
        )
    }
    clean = [f for f in frames if f["key"] not in bad_keys]
    if not clean:
        print("  [WARN] No clean frames to export.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"session_{session_id}_clean.h5"
    str_dtype = h5py.string_dtype(encoding="utf-8")

    gc.disable()
    try:
        with h5py.File(out_path, "w", locking=True) as f:
            for i, frame in enumerate(clean):
                grp = f.create_group(f"frame_{i:08d}")
                grp.attrs["frame_id"] = frame["frame_id"]
                grp.attrs["timestamp"] = frame["timestamp"]
                grp.attrs["wall_time"] = frame["wall_time"]
                grp.attrs["session_id"] = (
                    frame["session_id"]
                    if isinstance(frame["session_id"], bytes)
                    else frame["session_id"].encode("utf-8")
                )
                grp.attrs["vx"] = frame["vx"]
                grp.attrs["vy"] = frame["vy"]
                grp.attrs["vtheta"] = frame["vtheta"]
                grp.attrs["battery"] = frame["battery"]

                grp.create_dataset(
                    "image_jpeg",
                    data=np.frombuffer(frame["image_jpeg"], dtype=np.uint8),
                    dtype="u1",
                )
                grp.create_dataset("observation", data=frame["observation"])
                grp.create_dataset("action", data=frame["action"])
                grp.create_dataset("intent_classes", data=frame["intent_classes"])
                grp.create_dataset("intent_confs", data=frame["intent_confs"])
                grp.create_dataset("person_distances", data=frame["person_distances"])
                grp.create_dataset(
                    "distance_sources",
                    data=[
                        s.encode("utf-8") if isinstance(s, str) else s
                        for s in frame["distance_sources"]
                    ]
                    or [b"bbox"],
                    dtype=str_dtype,
                )
    finally:
        gc.enable()

    removed = len(frames) - len(clean)
    print(f"  Clean export: {len(clean)} frames kept, {removed} removed → {out_path}")


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="Experience Data Explorer")
    parser.add_argument("path", type=Path, help="HDF5 file or directory of .h5 files")
    parser.add_argument(
        "--samples", type=int, default=0, help="Save N sample images as a contact sheet"
    )
    parser.add_argument(
        "--export-clean", type=Path, default=None, help="Export clean frames to this directory"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print report as JSON (machine-readable)"
    )
    args = parser.parse_args()

    files = _collect_files(args.path)
    if not files:
        print(f"No .h5 files found at {args.path}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  Experience Data Explorer")
    print(f"  Files: {len(files)}")
    print(f"{'=' * 60}\n")

    all_frames: list[dict] = []
    all_issues: list[dict] = []

    for h5_path in files:
        print(f"+ {h5_path.name}")
        frames = _load_file(h5_path)
        issues = _quality_flags(frames)
        stats = _statistics(frames)

        all_frames.extend(frames)
        all_issues.extend(issues)

        if args.json:
            print(json.dumps({"file": h5_path.name, "stats": stats, "issues": issues}, indent=2))
        else:
            # Human-readable summary
            print(f"  Frames     : {stats.get('total_frames', 0)}")
            print(
                f"  Duration   : {stats.get('duration_s', 0):.1f}s  @ {stats.get('avg_fps', 0):.1f} FPS"
            )
            print(
                f"  Obs dim    : {stats.get('obs_dim')}   mean={stats.get('obs_mean', 0):.3f}  std={stats.get('obs_std', 0):.3f}  NaN={stats.get('obs_has_nan')}"
            )
            print(f"  Depth cov  : {stats.get('depth_frame_pct', 0):.1f}% frames with real depth")
            print(
                f"  Detections : {stats.get('person_detections', 0)} person detections  avg_dist={stats.get('avg_person_distance_m')}m"
            )
            print(f"  Battery    : {stats.get('battery_min')}% – {stats.get('battery_max')}%")
            print(f"  Modes      : {stats.get('mode_distribution')}")

            if issues:
                flag_counts: dict[str, int] = {}
                for iss in issues:
                    for flag in iss["flags"]:
                        flag_counts[flag] = flag_counts.get(flag, 0) + 1
                print(f"\n  Quality issues ({len(issues)}/{stats.get('total_frames', 0)} frames):")
                for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
                    print(f"     {flag:<40} × {count}")
            else:
                print("\n  No quality issues found")

        if args.samples > 0 and frames:
            grid_path = args.path if args.path.is_dir() else args.path.parent
            grid_path = grid_path / f"{h5_path.stem}_samples.jpg"
            _save_sample_grid(frames, grid_path, args.samples)

        if args.export_clean and frames:
            session_id = frames[0]["session_id"]
            if isinstance(session_id, bytes):
                session_id = session_id.decode()
            _export_clean(frames, issues, args.export_clean, session_id)

        print()

    # Aggregate summary when multiple files
    if len(files) > 1:
        agg_stats = _statistics(all_frames)
        print("-" * 60)
        print(f"  AGGREGATE  ({len(files)} sessions)")
        print(f"  Total frames : {agg_stats.get('total_frames', 0)}")
        print(f"  Total time   : {agg_stats.get('duration_s', 0):.1f}s")
        clean_pct = (1 - len(all_issues) / len(all_frames)) * 100 if all_frames else 0
        print(
            f"  Clean frames : {clean_pct:.1f}%  ({len(all_frames) - len(all_issues)}/{len(all_frames)})"
        )
        print(f"  Modes        : {agg_stats.get('mode_distribution')}")


if __name__ == "__main__":
    main()

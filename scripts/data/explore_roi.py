"""ROI Dataset Explorer tool.

Generates data exploration reports from the intent_dataset directory before training.
Outputs HTML report, JSON summary, and sample grids.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
try:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.perception.intent_labels import (
        REVIEW_ACCEPTED_STATUSES,
        canonical_label,
        needs_human_review,
    )
except Exception:  # pragma: no cover
    REVIEW_ACCEPTED_STATUSES = {"auto_accepted", "human_verified", "imported", "accepted"}

    def canonical_label(label: str | None) -> str:
        label_up = str(label or "UNCERTAIN").strip().upper()
        return "UNCERTAIN" if label_up in {"FOLLOW", "FOLLOWING"} else label_up

    def needs_human_review(label: str | None) -> bool:
        return canonical_label(label) in {"UNCERTAIN", "ERRATIC"}


class DatasetExplorer:
    def __init__(self, dataset_dir: Path, output_dir: Path):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.reports_dir = self.output_dir / "reports"
        self.grids_dir = self.reports_dir / "grids"

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.grids_dir.mkdir(parents=True, exist_ok=True)

        self.images: list[dict] = []
        self.classes: set[str] = set()
        self._metadata_index = self._load_metadata()

    def _load_metadata(self) -> dict[str, dict]:
        index: dict[str, dict] = {}
        meta_path = self.dataset_dir / "metadata.jsonl"
        if not meta_path.exists():
            return index

        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                file_value = row.get("file")
                if not file_value:
                    continue
                path = self.dataset_dir / file_value
                index[str(path.resolve())] = row
        return index

    def _read_image_meta(self, path: Path) -> dict | None:
        """Read ROI metadata from sidecar JSONL, with filename fallback for manual images."""
        label = path.parent.name
        sidecar = self._metadata_index.get(str(path.resolve()))
        if sidecar:
            label = canonical_label(sidecar.get("label", path.parent.name))
            return {
                "path": str(path),
                "filename": path.name,
                "label": label,
                "track_id": str(sidecar.get("track_uid") or sidecar.get("tid", "")),
                "cx_px": sidecar.get("cx"),
                "cy_px": sidecar.get("cy"),
                "dist_mm": sidecar.get("dist_mm"),
                "dsrc": "depth" if sidecar.get("dist_mm", 0) else "unknown",
                "frame_id": sidecar.get("frame_id", 0),
                "timestamp": sidecar.get("ts", 0),
                "depth_valid": sidecar.get("depth_valid", False),
                "review_status": sidecar.get("review_status", ""),
                "review_required": bool(sidecar.get("review_required", needs_human_review(label))),
            }

        stem = path.stem
        parts = stem.split("_")
        track_id = ""
        frame_id = 0
        for part in parts:
            if part.startswith("t") and part[1:].isdigit():
                track_id = part[1:]
            elif part.startswith("f") and part[1:].isdigit():
                frame_id = int(part[1:])

        label = canonical_label(label)
        return {
            "path": str(path),
            "filename": path.name,
            "label": label,
            "track_id": track_id or path.stem,
            "cx_px": None,
            "cy_px": None,
            "dist_mm": None,
            "dsrc": None,
            "frame_id": frame_id,
            "timestamp": 0,
            "depth_valid": False,
            "review_status": "",
            "review_required": needs_human_review(label),
        }

    def run(self) -> str:
        """Run full exploration and return path to JSON report."""
        print(f"[*] Scanning {self.dataset_dir} ...")
        for img_path in self.dataset_dir.rglob("*.jpg"):
            if "reports" in img_path.parts or "review_queue" in img_path.parts:
                continue
            meta = self._read_image_meta(img_path)
            if meta:
                self.images.append(meta)
                self.classes.add(meta["label"])

        if not self.images:
            print("[-] No valid ROI images found.")
            return ""

        print(f"[*] Found {len(self.images)} images across {len(self.classes)} classes.")

        # 1. Class Distribution
        class_counts = defaultdict(int)
        for img in self.images:
            class_counts[img["label"]] += 1

        # 2. Image Quality & Duplicate Check
        print("[*] Checking image quality and duplicates...")
        issues = []
        hashes = {}
        for img in self.images:
            path = Path(img["path"])
            if not path.exists() or path.stat().st_size == 0:
                issues.append(
                    {"file": img["filename"], "reason": "empty_or_missing", "path": img["path"]}
                )
                continue

            cv_img = cv2.imread(str(path))
            if cv_img is None:
                issues.append(
                    {"file": img["filename"], "reason": "corrupt_image", "path": img["path"]}
                )
                continue

            # Simple avg hash for duplicate detection
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (8, 8))
            avg = resized.mean()
            h = "".join("1" if p > avg else "0" for p in resized.flatten())

            if h in hashes:
                issues.append(
                    {
                        "file": img["filename"],
                        "reason": "duplicate",
                        "path": img["path"],
                        "duplicate_of": hashes[h],
                    }
                )
            else:
                hashes[h] = img["filename"]

        # 3. Track Length
        # Only sidecar-backed images have meaningful trajectory data.
        # Manual-import images (no sidecar) are assigned path.stem as track_id
        # which makes every image appear as a length-1 track — those are reported
        # separately so short_tracks_count reflects real trajectory quality.
        tracks_sidecar: dict[str, list] = defaultdict(list)
        no_sidecar_count = 0
        for img in self.images:
            has_sidecar = self._metadata_index.get(
                str(Path(img["path"]).resolve())
            ) is not None
            if has_sidecar:
                tracks_sidecar[img["track_id"]].append(img)
            else:
                no_sidecar_count += 1

        short_tracks = {
            tid: len(imgs)
            for tid, imgs in tracks_sidecar.items()
            if len(imgs) < 5
        }
        tracks = tracks_sidecar  # keep for total_tracks report

        # 4. Generate Grids
        print("[*] Generating sample grids...")
        for cls in self.classes:
            cls_images = [img for img in self.images if img["label"] == cls]
            np.random.shuffle(cls_images)
            sample = cls_images[:16]

            grid_imgs = []
            for img_meta in sample:
                cv_img = cv2.imread(img_meta["path"])
                if cv_img is not None:
                    grid_imgs.append(cv2.resize(cv_img, (64, 128)))

            if grid_imgs:
                # pad out
                while len(grid_imgs) < 16:
                    grid_imgs.append(np.zeros((128, 64, 3), dtype=np.uint8))

                rows = []
                for i in range(4):
                    row = np.hstack(grid_imgs[i * 4 : (i + 1) * 4])
                    rows.append(row)
                full_grid = np.vstack(rows)
                cv2.imwrite(str(self.grids_dir / f"grid_{cls}.jpg"), full_grid)

        # Build JSON Report
        report: dict[str, Any] = {
            "timestamp": int(time.time()),
            "total_images": len(self.images),
            "classes": dict(class_counts),
            "corrupt_images": [
                i
                for i in issues
                if i["reason"] == "corrupt_image" or i["reason"] == "empty_or_missing"
            ],
            "duplicates": [i for i in issues if i["reason"] == "duplicate"],
            "short_tracks_count": len(short_tracks),
            "total_tracks": len(tracks),
            # Images without sidecar JSONL (manual imports, legacy data).
            # Excluded from short_tracks_count to avoid false quality warnings.
            "no_sidecar_count": no_sidecar_count,
            "status": "PASS" if len(class_counts) > 0 else "FAIL",
        }

        # Determine class imbalance
        counts = list(class_counts.values())
        if counts and max(counts) > min(counts) * 10:
            report["imbalance_warning"] = True

        total_imgs = len(self.images)
        review_required_count = sum(1 for img in self.images if img.get("review_required"))
        report["review_required_count"] = review_required_count
        review_pending_by_class: dict[str, int] = defaultdict(int)
        for img in self.images:
            label = img["label"]
            review_status = str(img.get("review_status") or "")
            if needs_human_review(label) and review_status not in REVIEW_ACCEPTED_STATUSES:
                review_pending_by_class[label] += 1
        report["review_pending_by_class"] = dict(review_pending_by_class)

        cfe_count = class_counts.get("CROSSING", 0) + class_counts.get("ERRATIC", 0)
        cfe_ratio = cfe_count / total_imgs if total_imgs > 0 else 0
        if cfe_ratio < 0.20:
            report["diversity_warning"] = (
                f"Low diversity: CROSSING + ERRATIC combined is {cfe_ratio * 100:.1f}%, expected > 20%"
            )

        json_path = self.reports_dir / f"exploration_{report['timestamp']}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Build simple HTML Report
        html_path = self.reports_dir / f"exploration_{report['timestamp']}.html"
        self._generate_html(report, html_path)

        print(f"[+] HTML Report generated: {html_path}")
        print(f"[+] JSON Summary generated: {json_path}")

        return str(json_path)

    def _generate_html(self, report: dict, out_path: Path):
        total_images = report["total_images"]
        total_tracks = report["total_tracks"]
        no_sidecar_count = report["no_sidecar_count"]
        short_tracks_count = report["short_tracks_count"]
        review_required_count = report.get("review_required_count", 0)
        corrupt_count = len(report["corrupt_images"])
        duplicate_count = len(report["duplicates"])
        imbalance_warning_html = (
            '<p class="warn">Warning: High class imbalance detected (>10:1 ratio)</p>'
            if report.get("imbalance_warning")
            else ""
        )
        diversity_warning_html = (
            f'<p class="warn">Warning: {report["diversity_warning"]}</p>'
            if "diversity_warning" in report
            else ""
        )
        html = f"""
        <html>
        <head><title>ROI Dataset Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 40px; background: #f9f9f9; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; max-width: 600px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .warn {{ color: #d9534f; font-weight: bold; }}
            .ok {{ color: #5cb85c; font-weight: bold; }}
            .grid {{ max-width: 200px; margin: 10px; border: 1px solid #ccc; }}
        </style>
        </head>
        <body>
            <h1>ROI Dataset Exploration Report</h1>
            <div class="card">
                <h2>Summary</h2>
                <p>Total Images: {total_images}</p>
                <p>Total Tracks (sidecar): {total_tracks}</p>
                <p>Manual imports (no sidecar): {no_sidecar_count}</p>
                <p>Short Tracks &lt;5 imgs (sidecar only): {short_tracks_count}</p>
                <p>Needs Human Review: {review_required_count}</p>
                <p>Corrupt Images: {corrupt_count}</p>
                <p>Duplicates: {duplicate_count}</p>
            </div>

            <div class="card">
                <h2>Class Distribution</h2>
                <table>
                    <tr><th>Class</th><th>Count</th></tr>
        """
        for cls, count in report["classes"].items():
            html += f"<tr><td>{cls}</td><td>{count}</td></tr>"

        html += f"""
                </table>
                {imbalance_warning_html}
                {diversity_warning_html}
            </div>

            <div class="card">
                <h2>Sample Grids</h2>
                <div>
        """
        for cls in report["classes"].keys():
            grid_file = f"grids/grid_{cls}.jpg"
            html += f"""
            <div style="display:inline-block; text-align:center;">
                <h3>{cls}</h3>
                <img class="grid" src="{grid_file}" alt="Sample {cls}">
            </div>
            """

        html += """
                </div>
            </div>
        </body>
        </html>
        """
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ROI Dataset Explorer")
    parser.add_argument(
        "--dataset", type=str, default="intent_dataset", help="Path to intent_dataset directory"
    )
    parser.add_argument(
        "--output", type=str, default=".", help="Where to save the reports directory"
    )
    args = parser.parse_args()

    explorer = DatasetExplorer(Path(args.dataset), Path(args.output))
    explorer.run()

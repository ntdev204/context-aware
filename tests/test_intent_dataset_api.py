from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.train.train_intent_cnn import ROIDataset, _split_by_track


def _build_dataset_root(num_frames: int = 3) -> Path:
    root = Path(tempfile.mkdtemp(prefix="intent_dataset_api_"))
    rows = []
    class_dir = root / "stationary"
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        name = f"stationary_t1_f{i:06d}.jpg"
        Image.fromarray(np.zeros((256, 128, 3), dtype=np.uint8)).save(class_dir / name)
        rows.append(
            {
                "file": f"stationary/{name}",
                "label": "STATIONARY",
                "track_uid": "track_1",
                "frame_id": i,
                "review_status": "auto_accepted",
            }
        )
    with open(root / "metadata.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return root


class TestROIDatasetTemporalAPI:
    def test_temporal_window_one_preserves_time_dimension(self):
        root = _build_dataset_root()
        try:
            dataset = ROIDataset(root, temporal_window=1)
            img, label, direction = dataset[0]
            assert tuple(img.shape) == (1, 3, 256, 128)
            assert label.ndim == 0
            assert tuple(direction.shape) == (2,)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_temporal_window_three_returns_sequence_tensor(self):
        root = _build_dataset_root(num_frames=4)
        try:
            dataset = ROIDataset(root, temporal_window=3)
            img, _label, _direction = dataset[2]
            assert tuple(img.shape) == (3, 3, 256, 128)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_load_metadata_ignores_reports_and_review_dirs(self):
        root = _build_dataset_root(num_frames=2)
        try:
            (root / "reports").mkdir(exist_ok=True)
            (root / "_extracted").mkdir(exist_ok=True)
            (root / "review_queue").mkdir(exist_ok=True)
            for dirname in ("reports", "_extracted", "review_queue"):
                with open(root / dirname / "metadata.jsonl", "w", encoding="utf-8") as handle:
                    handle.write('{"file":"bogus.jpg","label":"STATIONARY"}\n')
            index = ROIDataset._load_metadata(root)
            assert all("bogus.jpg" not in key for key in index)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_load_metadata_reads_imported_metadata_jsonl(self):
        root = _build_dataset_root(num_frames=1)
        try:
            imported_path = root / "imported_metadata.jsonl"
            with open(imported_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "file": "stationary/stationary_t1_f000000.jpg",
                            "label": "STATIONARY",
                            "track_uid": "track_1",
                            "frame_id": 0,
                            "review_status": "human_verified",
                        }
                    )
                    + "\n"
                )
            index = ROIDataset._load_metadata(root)
            assert any("stationary_t1_f000000.jpg" in key for key in index)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_split_by_track_falls_back_when_val_empty(self):
        samples = [
            ([Path("a.jpg")], 0, 0.0, 0.0, "track_a"),
            ([Path("b.jpg")], 0, 0.0, 0.0, "track_b"),
        ]
        train_idx, val_idx = _split_by_track(samples, 0.5)
        assert train_idx
        assert val_idx

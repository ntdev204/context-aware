from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.data.autolabel import run_autolabel
from scripts.train.train_intent_cnn import ROIDataset


def _write_track_sequence(root: Path, distances: list[int]) -> Path:
    track_dir = root / "raw_session" / "session_001" / "track_0001"
    frames_dir = track_dir / "frames"
    frames_dir.mkdir(parents=True)

    frame_files = []
    for index, _distance in enumerate(distances, start=1):
        name = f"{index:04d}.jpg"
        Image.fromarray(np.full((256, 128, 3), 80 + index, dtype=np.uint8)).save(frames_dir / name)
        frame_files.append(f"frames/{name}")

    meta = {
        "track_id": "track_0001",
        "source_track_id": 7,
        "session_id": "session_001",
        "source_session_id": "raw_session",
        "timestamps": [1000 + i * 100 for i in range(len(distances))],
        "frame_ids": list(range(len(distances))),
        "frames": frame_files,
        "dist_mm": distances,
        "cx": [320 for _ in distances],
        "cy": [240 for _ in distances],
        "bw": [60 for _ in distances],
        "bh": [160 for _ in distances],
        "vx": [0.0 for _ in distances],
        "vy": [0.0 for _ in distances],
        "vtheta": [0.0 for _ in distances],
        "frame_count": len(distances),
        "frame_w": 640,
        "frame_h": 480,
    }
    (track_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return track_dir


def test_autolabel_reads_raw_track_sequence_v1(tmp_path):
    batch_dir = tmp_path / "batch"
    output_dir = tmp_path / "intent_dataset"
    _write_track_sequence(batch_dir, [2200 - index * 50 for index in range(20)])

    stats = run_autolabel(batch_dir, output_dir)

    assert stats["APPROACHING"] == 1
    labeled = sorted((output_dir / "approaching").glob("*.jpg"))
    assert len(labeled) == 1

    rows = [
        json.loads(line)
        for line in (output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["label"] for row in rows} == {"APPROACHING"}
    assert rows[0]["schema"] == "track_sequence_v1"
    assert rows[0]["track_uid"] == "raw_session:session_001:track_0001"
    assert rows[0]["review_status"] == "auto_accepted"

    sequences = [
        json.loads(line)
        for line in (output_dir / "sequence_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(sequences) == 1
    assert sequences[0]["schema"] == "intent_sequence_v1"
    assert sequences[0]["label"] == "APPROACHING"
    assert sequences[0]["sample_policy"] == "whole_track_k"
    assert sequences[0]["frame_count"] == 20
    assert len(sequences[0]["files"]) == 20
    assert all((output_dir / file_value).exists() for file_value in sequences[0]["files"])


def test_training_dataset_prefers_sequence_manifest(tmp_path):
    batch_dir = tmp_path / "batch"
    output_dir = tmp_path / "intent_dataset"
    _write_track_sequence(batch_dir, [2200 - index * 50 for index in range(20)])
    run_autolabel(batch_dir, output_dir)

    dataset = ROIDataset(output_dir, temporal_window=8)

    img, label, direction = dataset[0]
    assert tuple(img.shape) == (8, 3, 256, 128)
    assert label.item() == 1
    assert tuple(direction.tolist()) == (0.0, -0.6000000238418579)


def test_training_dataset_pads_short_whole_track(tmp_path):
    batch_dir = tmp_path / "batch"
    output_dir = tmp_path / "intent_dataset"
    _write_track_sequence(batch_dir, [2200 - index * 80 for index in range(4)])
    run_autolabel(batch_dir, output_dir)

    sequences = [
        json.loads(line)
        for line in (output_dir / "sequence_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sequences[0]["frame_count"] == 4

    dataset = ROIDataset(output_dir, temporal_window=6)
    img, label, _direction = dataset[0]
    assert tuple(img.shape) == (6, 3, 256, 128)
    assert label.item() == 1

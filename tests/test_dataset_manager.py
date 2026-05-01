from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from src.experience.dataset_manager import DATASET_INTENT, DatasetManager
from src.experience.roi_saver import ROISaver
from src.perception.roi_extractor import PersonROI


class _DummyCollector:
    stats = {"enabled": False, "collected": 0}

    def stop_session(self):
        return None

    def start_session(self, session_id):
        return None


class _DummyBuffer:
    def __init__(self, write_dir: Path):
        self.write_dir = write_dir

    def flush(self):
        return None


def _build_manager(tmp_path: Path) -> tuple[DatasetManager, ROISaver]:
    saver = ROISaver(str(tmp_path / "roi_dataset"))
    saver.start()
    manager = DatasetManager(
        roi_saver=saver,
        exp_collector=_DummyCollector(),
        exp_buffer=_DummyBuffer(tmp_path / "hdf5"),
    )
    return manager, saver


def _push_track_frames(saver: ROISaver, track_id: int = 7, count: int = 15) -> None:
    for index in range(count):
        roi = PersonROI(
            image=np.full((64, 32, 3), 40 + index, dtype=np.uint8),
            bbox=(300 + index, 100, 360 + index, 260),
            track_id=track_id,
            relative_position=(0.5, 0.5),
            distance_estimate=0.5,
            dist_mm=1800 - index * 20,
            frame_w=640,
            frame_h=480,
        )
        saver.push([roi], index, timestamp_ms=1000 + index * 100)


def test_intent_save_writes_raw_track_sequence(tmp_path):
    manager, saver = _build_manager(tmp_path)
    try:
        manager.start(DATASET_INTENT)
        _push_track_frames(saver)
        manager.stop()

        status = manager.save()
        session_dir = saver._session_dir

        assert status["saved"] is True
        assert status["dataset_stage"] == "raw_sequences"
        assert status["sequence_count"] == 1
        assert session_dir is not None
        assert (session_dir / "session_001" / "track_0001" / "frames" / "0001.jpg").exists()
        meta = json.loads(
            (session_dir / "session_001" / "track_0001" / "meta.json").read_text(encoding="utf-8")
        )
        assert meta["frame_count"] == 15
        assert meta["depth_valid_ratio"] == 1.0
        assert not (session_dir / "session_001" / "track_0001" / "label.json").exists()
    finally:
        saver.stop()


def test_intent_download_zips_raw_sequence_without_labels(tmp_path):
    manager, saver = _build_manager(tmp_path)
    try:
        manager.start(DATASET_INTENT)
        _push_track_frames(saver)
        manager.stop()

        zip_path, session_id = manager.build_zip()

        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())

        assert f"{session_id}/session_001/track_0001/frames/0001.jpg" in names
        assert f"{session_id}/session_001/track_0001/meta.json" in names
        assert f"{session_id}/manifest.json" in names
        assert all("label.json" not in name for name in names)
        assert all("intent_dataset" not in name for name in names)
    finally:
        saver.stop()


def test_intent_autolabel_is_disabled_on_jetson(tmp_path):
    manager, saver = _build_manager(tmp_path)
    try:
        manager.start(DATASET_INTENT)
        _push_track_frames(saver)
        manager.stop()

        try:
            manager.autolabel()
        except ValueError as exc:
            assert "Jetson auto-label is disabled" in str(exc)
        else:
            raise AssertionError("Jetson auto-label must be disabled")
    finally:
        saver.stop()

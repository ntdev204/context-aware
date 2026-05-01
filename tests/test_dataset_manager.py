from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from src.experience.dataset_manager import DATASET_INTENT, DatasetManager


class _DummyROISaver:
    def __init__(self, session_dir: Path):
        import threading

        self._state_lock = threading.Lock()
        self._session_dir = session_dir

    def status(self):
        return {
            "status": "stopped",
            "dataset_type": "roi",
            "session_id": self._session_dir.name,
            "frame_count": len(list(self._session_dir.glob("*.jpg"))),
        }

    def start_collection(self, session_id=None, clear_existing=True):
        return self.status()

    def stop_collection(self):
        return self.status()

    def discard_collection(self):
        return {"status": "discarded"}


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


def _install_data_script_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    autolabel = ModuleType("scripts.data.autolabel")

    def run_autolabel(raw_dir: Path, labeled_dir: Path):
        cls_dir = labeled_dir / "stationary"
        cls_dir.mkdir(parents=True, exist_ok=True)
        source = next(raw_dir.glob("*.jpg"))
        target = cls_dir / source.name
        target.write_bytes(source.read_bytes())
        row = {
            "file": f"stationary/{source.name}",
            "label": "STATIONARY",
            "track_uid": "test:t1",
            "frame_id": 1,
            "review_required": False,
            "review_status": "auto_accepted",
        }
        (labeled_dir / "metadata.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return {"STATIONARY": 1}

    autolabel.run_autolabel = run_autolabel
    monkeypatch.setitem(sys.modules, "scripts.data.autolabel", autolabel)

    explore = ModuleType("scripts.data.explore_roi")

    class DatasetExplorer:
        def __init__(self, dataset_dir: Path, output_dir: Path):
            self.output_dir = output_dir

        def run(self):
            reports = self.output_dir / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            report = reports / "exploration_1.json"
            report.write_text(
                json.dumps(
                    {
                        "total_images": 1,
                        "classes": {"STATIONARY": 1},
                        "duplicates": [],
                        "corrupt_images": [],
                        "review_pending_by_class": {},
                    }
                ),
                encoding="utf-8",
            )
            return str(report)

    explore.DatasetExplorer = DatasetExplorer
    monkeypatch.setitem(sys.modules, "scripts.data.explore_roi", explore)

    validate_mod = ModuleType("scripts.data.validate_dataset")
    validate_mod.validate = lambda report_path: 1
    monkeypatch.setitem(sys.modules, "scripts.data.validate_dataset", validate_mod)

    manifest_mod = ModuleType("scripts.data.build_intent_manifest")
    manifest_mod.build_manifest = lambda dataset, temporal_window=15: {
        "temporal_window": temporal_window,
        "ready_for_phase2_training": False,
    }
    monkeypatch.setitem(sys.modules, "scripts.data.build_intent_manifest", manifest_mod)


def _build_manager(tmp_path):
    session_dir = tmp_path / "20260501_010203_abcd1234"
    session_dir.mkdir()
    (session_dir / "roi_t1_f000001.jpg").write_bytes(b"fake-jpeg")
    (session_dir / "roi_t1_f000002.jpg").write_bytes(b"fake-jpeg-2")
    (session_dir / "metadata.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "file": "roi_t1_f000001.jpg",
                        "frame_id": 1,
                        "session_id": session_dir.name,
                        "tid": 1,
                        "ts": 1,
                        "cx": 10,
                        "cy": 10,
                        "bw": 10,
                        "bh": 20,
                        "frame_w": 640,
                        "frame_h": 480,
                        "dist_mm": 1000,
                    }
                ),
                json.dumps(
                    {
                        "file": "roi_t1_f000002.jpg",
                        "frame_id": 2,
                        "session_id": session_dir.name,
                        "tid": 1,
                        "ts": 2,
                        "cx": 12,
                        "cy": 10,
                        "bw": 10,
                        "bh": 20,
                        "frame_w": 640,
                        "frame_h": 480,
                        "dist_mm": 950,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manager = DatasetManager(
        roi_saver=_DummyROISaver(session_dir),
        exp_collector=_DummyCollector(),
        exp_buffer=_DummyBuffer(tmp_path / "hdf5"),
    )
    manager._active_mode = DATASET_INTENT
    manager._session_id = session_dir.name
    return manager, session_dir


def test_intent_save_keeps_raw_session_for_manual_review(tmp_path):
    manager, session_dir = _build_manager(tmp_path)

    status = manager.save()

    assert status["saved"] is True
    assert status["dataset_stage"] == "raw_review"
    assert (session_dir / "manifest.json").exists()
    assert not (session_dir / "intent_dataset").exists()


def test_delete_image_removes_file_and_matching_metadata(tmp_path):
    manager, session_dir = _build_manager(tmp_path)

    result = manager.delete_image(0)

    assert result["count"] == 1
    assert not (session_dir / "roi_t1_f000001.jpg").exists()
    assert (session_dir / "roi_t1_f000002.jpg").exists()
    metadata = (session_dir / "metadata.jsonl").read_text(encoding="utf-8")
    assert "roi_t1_f000001.jpg" not in metadata
    assert "roi_t1_f000002.jpg" in metadata


def test_intent_download_zips_raw_session_without_autolabel(tmp_path):
    manager, session_dir = _build_manager(tmp_path)

    zip_path, session_id = manager.build_zip()

    assert session_id == session_dir.name
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())

    assert f"{session_id}/raw/roi_t1_f000001.jpg" in names
    assert f"{session_id}/raw/metadata.jsonl" in names
    assert f"{session_id}/manifest.json" in names
    assert all("intent_dataset" not in name for name in names)


def test_intent_autolabel_is_explicit_manual_step(tmp_path, monkeypatch):
    _install_data_script_stubs(monkeypatch)
    manager, session_dir = _build_manager(tmp_path)

    result = manager.autolabel()

    assert result["status"] == "ok"
    assert (session_dir / "intent_dataset" / "metadata.jsonl").exists()
    assert (session_dir / "intent_dataset" / "manifest.json").exists()


def test_intent_download_after_autolabel_uses_train_ready_layout(tmp_path, monkeypatch):
    _install_data_script_stubs(monkeypatch)
    manager, _session_dir = _build_manager(tmp_path)
    manager.autolabel()

    zip_path, session_id = manager.build_zip()

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())

    assert f"{session_id}/raw/roi_t1_f000001.jpg" in names
    assert f"{session_id}/raw/metadata.jsonl" in names
    assert f"{session_id}/intent_dataset/stationary/" in names
    assert f"{session_id}/intent_dataset/approaching/" in names
    assert f"{session_id}/intent_dataset/departing/" in names
    assert f"{session_id}/intent_dataset/crossing/" in names
    assert f"{session_id}/intent_dataset/erratic/" in names
    assert f"{session_id}/intent_dataset/uncertain/" in names
    assert f"{session_id}/intent_dataset/metadata.jsonl" in names
    assert f"{session_id}/intent_dataset/reports/exploration_1.json" in names
    assert f"{session_id}/intent_dataset/manifest.json" in names
    assert f"{session_id}/manifest.json" in names

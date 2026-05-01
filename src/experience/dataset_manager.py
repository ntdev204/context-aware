from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from .buffer import ExperienceBuffer
from .collector import ExperienceCollector
from .roi_saver import ROISaver

DATASET_INTENT = "intent_cnn"
DATASET_RL = "rl"
DATASET_MODES = frozenset({DATASET_INTENT, DATASET_RL})


class DatasetManager:
    def __init__(
        self,
        roi_saver: ROISaver,
        exp_collector: ExperienceCollector,
        exp_buffer: ExperienceBuffer,
    ) -> None:
        self.roi_saver = roi_saver
        self.exp_collector = exp_collector
        self.exp_buffer = exp_buffer
        self._lock = threading.Lock()
        self._active_mode = ""
        self._session_id = ""
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._zip_path: Path | None = None
        self._bundle_dir: Path | None = None

    def start(self, mode: str) -> dict[str, Any]:
        mode = self._normalize_mode(mode)
        with self._lock:
            current = self._status_locked()
            if current["status"] == "recording":
                raise ValueError("Stop current dataset collection before starting another mode")
            self._cleanup_zip_locked()
            if self._session_id:
                self._discard_current_locked()

            session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            self._active_mode = mode
            self._session_id = session_id
            self._started_at = time.time()
            self._stopped_at = None

        if mode == DATASET_INTENT:
            self.exp_collector.stop_session()
            self.roi_saver.start_collection(session_id=session_id, clear_existing=True)
            return self.status()

        self.roi_saver.stop_collection()
        self._discard_hdf5_session_files(session_id)
        self.exp_collector.start_session(session_id)
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            mode = self._active_mode

        if mode == DATASET_INTENT:
            status = self.roi_saver.stop_collection()
            with self._lock:
                self._stopped_at = status.get("stopped_at") or time.time()
            return self.status()

        if mode == DATASET_RL:
            self.exp_collector.stop_session()
            with self._lock:
                self._stopped_at = time.time()
            self.exp_buffer.flush()
            return self.status()

        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def discard(self) -> dict[str, Any]:
        with self._lock:
            session_id = self._session_id
            mode = self._active_mode
            zip_path = self._zip_path
            bundle_dir = self._bundle_dir
            self._zip_path = None
            self._bundle_dir = None
            self._discard_current_locked()

        if zip_path and zip_path.exists():
            zip_path.unlink()
        if bundle_dir and bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        return {"status": "discarded", "session_id": session_id, "dataset_mode": mode}

    def preview_frame_path(self, index: int) -> Path:
        status = self.status()
        if status.get("status") == "idle":
            raise FileNotFoundError("No dataset session")

        if status.get("dataset_mode") == DATASET_INTENT:
            with self.roi_saver._state_lock:
                session_dir = self.roi_saver._session_dir
                if session_dir is None:
                    raise FileNotFoundError("No intent dataset session")
                frames = sorted(session_dir.glob("*.jpg"))
                if index < 0 or index >= len(frames):
                    raise FileNotFoundError("Intent preview frame not found")
                return frames[index]

        frame = self._extract_hdf5_preview(index)
        if frame is None:
            raise FileNotFoundError("RL preview frame not found")
        return frame

    def build_zip(self) -> tuple[Path, str]:
        status = self.status()
        if status.get("status") == "recording":
            raise ValueError("Stop collection before saving")
        if status.get("status") in {"idle", "unavailable"}:
            raise ValueError("No dataset session to save")
        if int(status.get("frame_count") or 0) <= 0:
            raise ValueError("Dataset session has no frames")

        session_id = str(status["session_id"])
        root = self._session_root(status)
        zip_path = root.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()

        if status.get("dataset_mode") == DATASET_INTENT:
            root = self._prepare_intent_training_bundle(root, status)
            zip_path = root.with_suffix(".zip")
            if zip_path.exists():
                zip_path.unlink()
        else:
            self._write_manifest(root, status)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if status.get("dataset_mode") == DATASET_INTENT:
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        archive.write(path, Path(session_id) / path.relative_to(root))
            else:
                for path in self._hdf5_session_files(session_id):
                    archive.write(path, path.name)
                manifest = root / "manifest.json"
                if manifest.exists():
                    archive.write(manifest, manifest.name)

        with self._lock:
            self._zip_path = zip_path
            if status.get("dataset_mode") == DATASET_INTENT:
                self._bundle_dir = root
        return zip_path, session_id

    def cleanup_after_download(self) -> None:
        self.discard()

    def _normalize_mode(self, mode: str) -> str:
        mode = (mode or "").strip().lower()
        aliases = {
            "intent": DATASET_INTENT,
            "intent-cnn": DATASET_INTENT,
            "intent_cnn": DATASET_INTENT,
            "roi": DATASET_INTENT,
            "rl": DATASET_RL,
            "hdf5": DATASET_RL,
            "experience": DATASET_RL,
        }
        normalized = aliases.get(mode, mode)
        if normalized not in DATASET_MODES:
            raise ValueError(f"Invalid dataset mode '{mode}'. Valid: {sorted(DATASET_MODES)}")
        return normalized

    def _status_locked(self) -> dict[str, Any]:
        if not self._session_id:
            return {
                "status": "idle",
                "dataset_mode": None,
                "available_modes": sorted(DATASET_MODES),
            }

        if self._active_mode == DATASET_INTENT:
            status = self.roi_saver.status()
            status["dataset_mode"] = DATASET_INTENT
            status["available_modes"] = sorted(DATASET_MODES)
            return status

        stats = self.exp_collector.stats
        files = self._hdf5_session_files(self._session_id)
        bytes_total = sum(p.stat().st_size for p in files if p.is_file())
        frame_count = self._count_hdf5_frames(files) or int(stats.get("collected", 0))
        return {
            "status": "recording" if stats.get("enabled") else "stopped",
            "dataset_mode": DATASET_RL,
            "dataset_type": "hdf5",
            "available_modes": sorted(DATASET_MODES),
            "session_id": self._session_id,
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "frame_count": frame_count,
            "bytes_total": bytes_total,
            "preview_count": min(frame_count, 12),
            "preview_indexes": list(range(min(frame_count, 12))),
        }

    def _discard_current_locked(self) -> None:
        session_id = self._session_id
        self.roi_saver.discard_collection()
        self.exp_collector.stop_session()
        if session_id:
            self._discard_hdf5_session_files(session_id)
        self._active_mode = ""
        self._session_id = ""
        self._started_at = None
        self._stopped_at = None

    def _session_root(self, status: dict[str, Any]) -> Path:
        if status.get("dataset_mode") == DATASET_INTENT:
            if self.roi_saver._session_dir is None:
                raise ValueError("No intent dataset session directory")
            return self.roi_saver._session_dir
        self.exp_buffer.write_dir.mkdir(parents=True, exist_ok=True)
        return self.exp_buffer.write_dir

    def _hdf5_session_files(self, session_id: str) -> list[Path]:
        return sorted(self.exp_buffer.write_dir.glob(f"session_{session_id}.h5"))

    def _discard_hdf5_session_files(self, session_id: str) -> None:
        for path in self._hdf5_session_files(session_id):
            if path.exists():
                path.unlink()
        manifest = self.exp_buffer.write_dir / "manifest.json"
        if manifest.exists():
            manifest.unlink()
        for preview in self.exp_buffer.write_dir.glob(f"preview_{session_id}_*.jpg"):
            preview.unlink()

    def _count_hdf5_frames(self, files: list[Path]) -> int:
        if not files:
            return 0
        try:
            import h5py

            count = 0
            for path in files:
                with h5py.File(path, "r", locking=False) as handle:
                    count += len(handle.keys())
            return count
        except Exception:
            return 0

    def _extract_hdf5_preview(self, index: int) -> Path | None:
        status = self.status()
        session_id = status.get("session_id")
        if not session_id:
            return None
        try:
            import h5py

            files = self._hdf5_session_files(str(session_id))
            offset = 0
            for path in files:
                with h5py.File(path, "r", locking=False) as handle:
                    keys = sorted(handle.keys())
                    if index < offset + len(keys):
                        jpeg = bytes(handle[keys[index - offset]]["image_jpeg"][:])
                        preview_path = self.exp_buffer.write_dir / f"preview_{session_id}_{index}.jpg"
                        preview_path.write_bytes(jpeg)
                        return preview_path
                    offset += len(keys)
        except Exception:
            return None
        return None

    def _write_manifest(self, root: Path, status: dict[str, Any]) -> None:
        manifest = dict(status)
        manifest["created_at"] = time.time()
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _prepare_intent_training_bundle(self, raw_session_dir: Path, status: dict[str, Any]) -> Path:
        """Build a session-scoped train-ready bundle on Jetson before download."""
        session_id = str(status["session_id"])
        bundle_dir = raw_session_dir.parent / f"{session_id}_train_ready"
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)

        raw_dir = bundle_dir / "raw"
        labeled_dir = bundle_dir / "intent_dataset"
        raw_dir.mkdir(parents=True, exist_ok=True)
        labeled_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(raw_session_dir.iterdir()):
            if path.is_file():
                shutil.copy2(path, raw_dir / path.name)

        report_path = ""
        validation_status: int | None = None
        manifest: dict[str, Any] | None = None
        error: str | None = None

        try:
            from scripts.data.autolabel import run_autolabel
            from scripts.data.build_intent_manifest import build_manifest
            from scripts.data.explore_roi import DatasetExplorer
            from scripts.data.validate_dataset import validate

            run_autolabel(raw_dir, labeled_dir)
            report_path = DatasetExplorer(labeled_dir, labeled_dir).run()
            if report_path:
                validation_status = validate(Path(report_path))
            manifest = build_manifest(labeled_dir, temporal_window=15)
            manifest["validation_status"] = validation_status
            with open(labeled_dir / "manifest.json", "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
        except Exception as exc:
            error = str(exc)

        bundle_manifest = {
            "bundle_type": "intent_train_ready_session",
            "session_id": session_id,
            "created_at": time.time(),
            "raw_dir": "raw",
            "train_dataset_dir": "intent_dataset",
            "frame_count": status.get("frame_count", 0),
            "autolabel_completed": error is None,
            "report_path": str(Path(report_path).relative_to(bundle_dir)) if report_path else None,
            "validation_status": validation_status,
            "ready_for_phase2_training": bool(
                manifest.get("ready_for_phase2_training") if manifest else False
            ),
            "error": error,
        }
        (bundle_dir / "manifest.json").write_text(
            json.dumps(bundle_manifest, indent=2),
            encoding="utf-8",
        )
        return bundle_dir

    def _cleanup_zip_locked(self) -> None:
        if self._zip_path and self._zip_path.exists():
            self._zip_path.unlink()
        self._zip_path = None
        if self._bundle_dir and self._bundle_dir.exists():
            shutil.rmtree(self._bundle_dir)
        self._bundle_dir = None
        for preview in self.exp_buffer.write_dir.glob("preview_*.jpg"):
            preview.unlink()
        manifest = self.exp_buffer.write_dir / "manifest.json"
        if manifest.exists():
            manifest.unlink()

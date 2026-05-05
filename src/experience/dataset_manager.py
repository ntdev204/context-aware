from __future__ import annotations

import json
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
INTENT_LABEL_DIRS = (
    "stationary",
    "approaching",
    "departing",
    "crossing",
    "erratic",
    "uncertain",
)


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
            self._zip_path = None
            self._discard_current_locked()

        if zip_path and zip_path.exists():
            zip_path.unlink()
        return {"status": "discarded", "session_id": session_id, "dataset_mode": mode}

    def save(self) -> dict[str, Any]:
        status = self.status()
        if status.get("status") == "recording":
            raise ValueError("Stop collection before saving")
        if status.get("status") in {"idle", "unavailable"}:
            raise ValueError("No dataset session to save")
        if int(status.get("frame_count") or 0) <= 0:
            raise ValueError("Dataset session has no frames")

        root = self._session_root(status)
        if status.get("dataset_mode") == DATASET_INTENT and hasattr(self.roi_saver, "_finalize_session"):
            self.roi_saver._finalize_session()
            status = self.status()
        saved_status = dict(status)
        saved_status["saved"] = True
        saved_status["dataset_stage"] = "raw_sequences"
        self._write_manifest(root, saved_status)
        return self.status()

    def preview_frame_path(self, index: int) -> Path:
        status = self.status()
        if status.get("status") == "idle":
            raise FileNotFoundError("No dataset session")

        if status.get("dataset_mode") == DATASET_INTENT:
            with self.roi_saver._state_lock:
                session_dir = self.roi_saver._session_dir
                if session_dir is None:
                    raise FileNotFoundError("No intent dataset session")
                rows = self._load_metadata_rows(session_dir)
                if rows:
                    if index < 0 or index >= len(rows):
                        raise FileNotFoundError("Intent preview frame not found")
                    file_name = str(rows[index].get("file") or "")
                    frame_path = session_dir / file_name
                    if not file_name or not frame_path.is_file():
                        raise FileNotFoundError("Intent preview frame not found")
                    return frame_path
                frames = sorted(session_dir.glob("session_*/track_*/frames/*.jpg"))
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
            if hasattr(self.roi_saver, "_finalize_session"):
                self.roi_saver._finalize_session()
                status = self.status()
            self._write_manifest(root, {**status, "saved": True, "dataset_stage": "raw_sequences"})
        else:
            self._write_manifest(root, status)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if status.get("dataset_mode") == DATASET_INTENT:
                self._write_intent_session_archive(archive, root, session_id)
            else:
                for path in self._hdf5_session_files(session_id):
                    archive.write(path, path.name)
                manifest = root / "manifest.json"
                if manifest.exists():
                    archive.write(manifest, manifest.name)

        with self._lock:
            self._zip_path = zip_path
        return zip_path, session_id

    def cleanup_after_download(self) -> None:
        with self._lock:
            zip_path = self._zip_path
            self._zip_path = None
        if zip_path and zip_path.exists():
            zip_path.unlink()

    def list_images(self) -> dict[str, Any]:
        status = self.status()
        if status.get("status") in {"idle", "unavailable"}:
            raise ValueError("No dataset session")
        if status.get("dataset_mode") != DATASET_INTENT:
            raise ValueError("Image review is only available for intent ROI sessions")

        root = self._session_root(status)
        rows = self._load_metadata_rows(root)
        images = []
        if rows:
            for original_index, row in enumerate(rows):
                file_name = str(row.get("file") or "")
                path = root / file_name
                if not file_name or not path.is_file():
                    continue
                images.append(
                    {
                        "index": original_index,
                        "file": file_name,
                        "session_id": status.get("session_id"),
                        "frame_id": row.get("frame_id"),
                        "track_id": row.get("track_uid") or row.get("tid") or row.get("track_id"),
                        "timestamp": row.get("ts"),
                        "metadata": row,
                    }
                )
        else:
            for original_index, frame in enumerate(sorted(root.glob("session_*/track_*/frames/*.jpg"))):
                meta = self._read_manifest(frame.parent.parent)
                images.append(
                    {
                        "index": original_index,
                        "file": str(frame.relative_to(root)),
                        "session_id": meta.get("session_id") or status.get("session_id"),
                        "frame_id": None,
                        "track_id": meta.get("track_id") or frame.parent.parent.name,
                        "timestamp": None,
                        "metadata": meta,
                    }
                )

        return {
            "status": status.get("status"),
            "dataset_mode": DATASET_INTENT,
            "session_id": status.get("session_id"),
            "count": len(images),
            "images": images,
        }

    def delete_image(self, index: int) -> dict[str, Any]:
        status = self.status()
        if status.get("status") == "recording":
            raise ValueError("Stop collection before editing dataset images")
        if status.get("dataset_mode") != DATASET_INTENT:
            raise ValueError("Image editing is only available for intent ROI sessions")

        root = self._session_root(status)
        rows = self._load_metadata_rows(root)
        if index < 0 or index >= len(rows):
            raise FileNotFoundError("Dataset image not found")

        row = rows[index]
        file_name = str(row.get("file") or "")
        if file_name:
            image_path = root / file_name
            if image_path.exists():
                image_path.unlink()

        remaining = [item for i, item in enumerate(rows) if i != index]
        self._write_metadata_rows(root, remaining)
        self._write_manifest(
            root,
            {**self.status(), "saved": True, "dataset_stage": "raw_sequences"},
        )
        return self.list_images()

    def autolabel(self) -> dict[str, Any]:
        raise ValueError(
            "Jetson auto-label is disabled. Download the raw sequence dataset and upload it to the server Dataset page."
        )

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
            if self.roi_saver._session_dir is not None:
                manifest = self._read_manifest(self.roi_saver._session_dir)
                if manifest:
                    status["saved"] = bool(manifest.get("saved"))
                    status["dataset_stage"] = manifest.get("dataset_stage")
                    status["autolabel_result"] = manifest.get("autolabel_result")
                labeled_dir = self.roi_saver._session_dir / "intent_dataset"
                status["autolabeled"] = labeled_dir.exists()
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
        manifest = {**self._read_manifest(root), **status}
        manifest["created_at"] = time.time()
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _write_intent_session_archive(
        self,
        archive: zipfile.ZipFile,
        root: Path,
        session_id: str,
    ) -> None:
        session_root = Path(session_id)
        for path in sorted(root.rglob("*")):
            if path.is_file() and "intent_dataset" not in path.parts:
                archive.write(path, session_root / path.relative_to(root))

        labeled_dir = root / "intent_dataset"
        if labeled_dir.exists():
            labeled_root = session_root / "intent_dataset"
            for label in INTENT_LABEL_DIRS:
                archive.writestr(str(labeled_root / label) + "/", "")
            for path in sorted(labeled_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, labeled_root / path.relative_to(labeled_dir))

    def _read_manifest(self, root: Path) -> dict[str, Any]:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_metadata_rows(self, root: Path) -> list[dict[str, Any]]:
        metadata_path = root / "metadata.jsonl"
        if not metadata_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _write_metadata_rows(self, root: Path, rows: list[dict[str, Any]]) -> None:
        metadata_path = root / "metadata.jsonl"
        tmp_path = metadata_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_path.replace(metadata_path)

    def _cleanup_zip_locked(self) -> None:
        if self._zip_path and self._zip_path.exists():
            self._zip_path.unlink()
        self._zip_path = None
        for preview in self.exp_buffer.write_dir.glob("preview_*.jpg"):
            preview.unlink()
        manifest = self.exp_buffer.write_dir / "manifest.json"
        if manifest.exists():
            manifest.unlink()

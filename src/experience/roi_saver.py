"""Raw track sequence saver for dataset collection.

The class keeps the historical ROISaver name because the pipeline already
depends on it, but the artifact is no longer image-level ROI metadata. During
collection it buffers frames per track, and when collection stops it flushes
valid segments into:

dataset/
  session_001/
    track_0001/
      frames/0001.jpg ...
      meta.json

No label.json is written on Jetson. Labeling happens on the server after the
raw dataset zip is uploaded.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..perception.roi_extractor import PersonROI

logger = logging.getLogger(__name__)

MIN_SEQUENCE_FRAMES = 15
MAX_SEQUENCE_FRAMES = 30
MAX_TRACK_GAP_FRAMES = 45


@dataclass
class ROISaveRecord:
    track_id: int
    frame_id: int
    session_dir: Path
    image: np.ndarray
    cx: float
    cy: float
    bw: int
    bh: int
    frame_w: int
    frame_h: int
    distance_estimate: float
    dist_mm: float
    timestamp_ms: float
    vx: float = 0.0
    vy: float = 0.0
    vtheta: float = 0.0


class ROISaver:
    def __init__(self, save_dir: str, jpeg_quality: int = 90):
        self.root_dir = Path(save_dir)
        self.save_dir = self.root_dir
        self.jpeg_quality = jpeg_quality
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[list[ROISaveRecord]] = queue.Queue(maxsize=400)
        self._running = False
        self._thread: threading.Thread | None = None
        self._collecting = False
        self._session_id = ""
        self._session_dir: Path | None = None
        self._saved_count = 0
        self._sequence_count = 0
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._state_lock = threading.Lock()
        self._tracks: dict[int, list[ROISaveRecord]] = {}
        self._track_names: dict[int, str] = {}
        self._finalized = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="track-sequence-saver")
        self._thread.start()
        logger.info("Track sequence saver started -> %s", self.save_dir)

    def stop(self) -> None:
        self._wait_for_queue_idle(timeout=3.0)
        with self._state_lock:
            if self._collecting:
                self._collecting = False
                self._stopped_at = time.time()
        self._finalize_session()
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def start_collection(self, session_id: str | None = None, clear_existing: bool = True) -> dict[str, Any]:
        with self._state_lock:
            if self._collecting:
                return self._status_locked()
            if clear_existing:
                self._delete_session_dir_locked()

            session_id = session_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            self._session_id = session_id
            self._session_dir = self.root_dir / session_id
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self.save_dir = self._session_dir
            self._saved_count = 0
            self._sequence_count = 0
            self._started_at = time.time()
            self._stopped_at = None
            self._collecting = True
            self._tracks = {}
            self._track_names = {}
            self._finalized = False
            self._drain_queue()
            logger.info("Track sequence collection started: %s", self._session_dir)
            return self._status_locked()

    def stop_collection(self) -> dict[str, Any]:
        self._wait_for_queue_idle(timeout=3.0)
        with self._state_lock:
            if self._collecting:
                self._collecting = False
                self._stopped_at = time.time()
        self._finalize_session()
        with self._state_lock:
            logger.info(
                "Track sequence collection stopped: %s frames, %s sequences",
                self._saved_count,
                self._sequence_count,
            )
            return self._status_locked()

    def discard_collection(self) -> dict[str, Any]:
        with self._state_lock:
            session_id = self._session_id
            self._collecting = False
            self._drain_queue()
            self._delete_session_dir_locked()
            self._session_id = ""
            self._session_dir = None
            self.save_dir = self.root_dir
            self._saved_count = 0
            self._sequence_count = 0
            self._started_at = None
            self._stopped_at = None
            self._tracks = {}
            self._track_names = {}
            self._finalized = False
            return {"status": "discarded", "session_id": session_id}

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return self._status_locked()

    def push(
        self,
        rois: list[PersonROI],
        frame_id: int,
        robot_state=None,
        timestamp_ms: float | None = None,
    ) -> None:
        ts_ms = timestamp_ms if timestamp_ms is not None else time.time() * 1000.0
        vx = vy = vtheta = 0.0
        if robot_state is not None:
            vx = float(getattr(robot_state, "vx", 0.0))
            vy = float(getattr(robot_state, "vy", 0.0))
            vtheta = float(getattr(robot_state, "vtheta", 0.0))

        records: list[ROISaveRecord] = []
        with self._state_lock:
            collecting = self._collecting
            session_dir = self._session_dir
            if not self._running or not collecting or session_dir is None or not rois:
                return

            for roi in rois:
                x1, y1, x2, y2 = roi.bbox
                records.append(
                    ROISaveRecord(
                        track_id=roi.track_id,
                        frame_id=frame_id,
                        session_dir=session_dir,
                        image=roi.image.copy(),
                        cx=(x1 + x2) / 2.0,
                        cy=(y1 + y2) / 2.0,
                        bw=x2 - x1,
                        bh=y2 - y1,
                        frame_w=roi.frame_w,
                        frame_h=roi.frame_h,
                        distance_estimate=roi.distance_estimate,
                        dist_mm=roi.dist_mm,
                        timestamp_ms=ts_ms,
                        vx=vx,
                        vy=vy,
                        vtheta=vtheta,
                    )
                )

        if not records:
            return
        try:
            self._queue.put_nowait(records)
        except queue.Full:
            logger.debug("Track sequence saver queue full; dropping frame %s", frame_id)

    def _worker(self) -> None:
        while self._running:
            try:
                records = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._state_lock:
                    if self._session_dir is None or self._finalized:
                        continue
                    for rec in records:
                        self._tracks.setdefault(rec.track_id, []).append(rec)
                        self._saved_count += 1
            except Exception as exc:
                logger.error("Track sequence saver worker error: %s", exc)
            finally:
                self._queue.task_done()

    def _finalize_session(self) -> None:
        with self._state_lock:
            if self._finalized or self._session_dir is None or not self._session_id:
                return
            session_dir = self._session_dir
            session_id = self._session_id
            tracks = {track_id: list(records) for track_id, records in self._tracks.items()}
            started_at = self._started_at
            stopped_at = self._stopped_at
            self._finalized = True

        sequence_count = 0
        frame_count = 0
        rejected: list[dict[str, Any]] = []
        session_name = "session_001"
        session_root = session_dir / session_name
        session_root.mkdir(parents=True, exist_ok=True)

        for track_id, records in sorted(tracks.items()):
            for segment in _split_track(records):
                if len(segment) < MIN_SEQUENCE_FRAMES:
                    rejected.append(
                        {
                            "track_id": track_id,
                            "frame_count": len(segment),
                            "reason": "too_short",
                        }
                    )
                    continue
                sequence_count += 1
                track_name = f"track_{sequence_count:04d}"
                track_dir = session_root / track_name
                frames_dir = track_dir / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                frame_files: list[str] = []
                for index, rec in enumerate(segment, start=1):
                    filename = f"{index:04d}.jpg"
                    frame_path = frames_dir / filename
                    ok = cv2.imwrite(
                        str(frame_path),
                        rec.image,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                    if ok:
                        frame_files.append(f"frames/{filename}")
                meta = _build_meta(
                    session_id=session_name,
                    track_id=track_name,
                    source_session_id=session_id,
                    source_track_id=track_id,
                    frame_files=frame_files,
                    records=segment[: len(frame_files)],
                )
                (track_dir / "meta.json").write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                frame_count += len(frame_files)

        manifest = {
            "session_id": session_id,
            "dataset_type": "track_sequence",
            "schema": "track_sequence_v1",
            "dataset_stage": "raw_sequences",
            "started_at": started_at,
            "stopped_at": stopped_at,
            "saved": True,
            "sequence_count": sequence_count,
            "frame_count": frame_count,
            "rejected_count": len(rejected),
            "rejected": rejected[:200],
        }
        (session_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        with self._state_lock:
            self._sequence_count = sequence_count
            self._saved_count = frame_count

    def _status_locked(self) -> dict[str, Any]:
        if not self._session_id:
            return {"status": "idle", "dataset_type": "track_sequence"}
        bytes_total = 0
        if self._session_dir and self._session_dir.exists():
            bytes_total = sum(p.stat().st_size for p in self._session_dir.rglob("*") if p.is_file())
        buffered_count = self._saved_count
        if self._collecting:
            buffered_count = sum(len(records) for records in self._tracks.values())
        return {
            "status": "recording" if self._collecting else "stopped",
            "dataset_type": "track_sequence",
            "session_id": self._session_id,
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "frame_count": buffered_count,
            "sequence_count": self._sequence_count,
            "bytes_total": bytes_total,
            "preview_count": min(buffered_count, 12),
            "preview_indexes": list(range(min(buffered_count, 12))),
        }

    def _delete_session_dir_locked(self) -> None:
        if self._session_dir and self._session_dir.exists():
            shutil.rmtree(self._session_dir)

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                return

    def _wait_for_queue_idle(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while getattr(self._queue, "unfinished_tasks", 0) > 0 and time.monotonic() < deadline:
            time.sleep(0.01)


def _split_track(records: list[ROISaveRecord]) -> list[list[ROISaveRecord]]:
    if not records:
        return []
    ordered = sorted(records, key=lambda item: (item.frame_id, item.timestamp_ms))
    chunks: list[list[ROISaveRecord]] = []
    current: list[ROISaveRecord] = [ordered[0]]
    for rec in ordered[1:]:
        prev = current[-1]
        if rec.frame_id - prev.frame_id > MAX_TRACK_GAP_FRAMES or len(current) >= MAX_SEQUENCE_FRAMES:
            chunks.append(current)
            current = [rec]
        else:
            current.append(rec)
    chunks.append(current)
    return chunks


def _build_meta(
    session_id: str,
    track_id: str,
    source_session_id: str,
    source_track_id: int,
    frame_files: list[str],
    records: list[ROISaveRecord],
) -> dict[str, Any]:
    depth_valid = [rec.dist_mm > 0 for rec in records]
    return {
        "track_id": track_id,
        "source_track_id": source_track_id,
        "session_id": session_id,
        "source_session_id": source_session_id,
        "timestamps": [int(rec.timestamp_ms) for rec in records],
        "frame_ids": [rec.frame_id for rec in records],
        "frames": frame_files,
        "dist_mm": [int(rec.dist_mm) for rec in records],
        "cx": [round(rec.cx, 1) for rec in records],
        "cy": [round(rec.cy, 1) for rec in records],
        "bw": [int(rec.bw) for rec in records],
        "bh": [int(rec.bh) for rec in records],
        "vx": [round(rec.vx, 4) for rec in records],
        "vy": [round(rec.vy, 4) for rec in records],
        "vtheta": [round(rec.vtheta, 4) for rec in records],
        "depth_valid_ratio": round(sum(depth_valid) / len(depth_valid), 4) if depth_valid else 0.0,
        "frame_count": len(records),
        "frame_w": records[0].frame_w if records else 0,
        "frame_h": records[0].frame_h if records else 0,
        "bbox_center_trajectory": [
            {"cx": round(rec.cx, 1), "cy": round(rec.cy, 1)}
            for rec in records
        ],
    }

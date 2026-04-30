"""ROISaver — saves person-crop JPGs + sidecar metadata.jsonl for auto-labelling.

Each saved ROI produces:
  roi_t{track_id}_f{frame_id:06d}.jpg   — 128×256 BGR crop (clean, no bbox overlay)
  metadata.jsonl                         — one JSON line per saved ROI

metadata.jsonl schema (one object per line):
  {
    "file":    "roi_t1_f000042.jpg",
    "frame_id": 42,
    "session_id": "20260430_120000_abcd1234",
    "tid":     1,
    "ts":      1714500000123,   # wall-clock ms since epoch
    "cx":      320.5,           # bbox centre x in original frame pixels
    "cy":      240.1,           # bbox centre y in original frame pixels
    "bw":      80,              # bbox width in original frame pixels
    "bh":      180,             # bbox height in original frame pixels
    "frame_w": 640,
    "frame_h": 480,
    "distance_estimate": 0.625, # bbox-height proxy [0, 1]
    "dist_mm": 1850,            # sensor depth in mm  (0 = unknown)
    "vx":      0.30,            # robot linear-x m/s
    "vy":      0.00,            # robot linear-y m/s
    "vtheta":  -0.05            # robot angular-z rad/s
  }
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

import cv2
import numpy as np

from ..perception.roi_extractor import PersonROI

logger = logging.getLogger(__name__)


@dataclass
class ROISaveRecord:
    """Bundles everything needed to write one ROI entry to disk."""

    track_id: int
    frame_id: int
    session_dir: Path
    image: np.ndarray        # 128×256 BGR, already cropped & resized
    cx: float                # bbox centre-x in the original camera frame (pixels)
    cy: float                # bbox centre-y in the original camera frame (pixels)
    bw: int
    bh: int
    frame_w: int
    frame_h: int
    distance_estimate: float
    dist_mm: float           # sensor depth in mm  (0 = unknown / depth not available)
    timestamp_ms: float      # wall-clock timestamp in milliseconds
    vx: float = 0.0          # robot forward velocity  (m/s)
    vy: float = 0.0          # robot lateral velocity  (m/s)
    vtheta: float = 0.0      # robot angular velocity  (rad/s)


class ROISaver:
    def __init__(self, save_dir: str, jpeg_quality: int = 90):
        self.root_dir = Path(save_dir)
        self.save_dir = self.root_dir
        self.jpeg_quality = jpeg_quality
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._running = False
        self._thread: threading.Thread | None = None
        self._collecting = False
        self._session_id = ""
        self._session_dir: Path | None = None
        self._saved_count = 0
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._state_lock = threading.Lock()
        self._last_saved: dict[int, int] = {}  # track_id -> last saved frame_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="roi-saver")
        self._thread.start()
        logger.info("ROISaver started → %s", self.save_dir)

    def stop(self) -> None:
        with self._state_lock:
            if self._collecting:
                self._collecting = False
                self._stopped_at = time.time()
        self._wait_for_queue_idle(timeout=2.0)
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_collection(self, session_id: str | None = None, clear_existing: bool = True) -> dict:
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
            self._started_at = time.time()
            self._stopped_at = None
            self._collecting = True
            self._last_saved = {}
            self._drain_queue()
            logger.info("ROI collection started: %s", self._session_dir)
            return self._status_locked()

    def stop_collection(self) -> dict:
        with self._state_lock:
            if self._collecting:
                self._collecting = False
                self._stopped_at = time.time()
                logger.info("ROI collection stopped: %s images", self._saved_count)
        self._wait_for_queue_idle(timeout=2.0)
        with self._state_lock:
            return self._status_locked()

    def discard_collection(self) -> dict:
        with self._state_lock:
            session_id = self._session_id
            self._collecting = False
            self._drain_queue()
            self._delete_session_dir_locked()
            self._session_id = ""
            self._session_dir = None
            self.save_dir = self.root_dir
            self._saved_count = 0
            self._started_at = None
            self._stopped_at = None
            self._last_saved = {}
            return {"status": "discarded", "session_id": session_id}

    def status(self) -> dict:
        with self._state_lock:
            return self._status_locked()

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push(
        self,
        rois: list[PersonROI],
        frame_id: int,
        robot_state=None,       # navigation.RobotState | None
        timestamp_ms: float | None = None,
    ) -> None:
        """Queue ROIs for background saving.

        Args:
            rois:          Cropped person regions from this frame.
            frame_id:      Monotonic frame counter from the inference loop.
            robot_state:   Current RobotState (provides vx, vy, vtheta).
                           Pass None when unavailable — velocities default to 0.
            timestamp_ms:  Wall-clock time in milliseconds. Defaults to now.
        """
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

            for r in rois:
                last_f = self._last_saved.get(r.track_id, -999)
                if frame_id - last_f < 15:           # throttle: max 2 fps @ 30fps camera
                    continue
                self._last_saved[r.track_id] = frame_id

                x1, y1, x2, y2 = r.bbox
                records.append(
                    ROISaveRecord(
                        track_id=r.track_id,
                        frame_id=frame_id,
                        session_dir=session_dir,
                        image=r.image.copy(),
                        cx=(x1 + x2) / 2.0,
                        cy=(y1 + y2) / 2.0,
                        bw=x2 - x1,
                        bh=y2 - y1,
                        frame_w=r.frame_w,
                        frame_h=r.frame_h,
                        distance_estimate=r.distance_estimate,
                        dist_mm=r.dist_mm,
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
            pass  # drop silently — dataset collection is best-effort

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while self._running:
            try:
                records: list[ROISaveRecord] = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            session_dir = records[0].session_dir

            jsonl_path = session_dir / "metadata.jsonl"

            try:
                with open(jsonl_path, "a", encoding="utf-8") as jf:
                    for rec in records:
                        filename = f"roi_t{rec.track_id}_f{rec.frame_id:06d}.jpg"
                        filepath = session_dir / filename

                        ok = cv2.imwrite(
                            str(filepath),
                            rec.image,
                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                        )
                        if not ok:
                            logger.warning("ROISaver: imwrite failed for %s", filepath)
                            continue

                        meta = {
                            "file": filename,
                            "frame_id": rec.frame_id,
                            "session_id": rec.session_dir.name,
                            "tid": rec.track_id,
                            "ts": int(rec.timestamp_ms),
                            "cx": round(rec.cx, 1),
                            "cy": round(rec.cy, 1),
                            "bw": rec.bw,
                            "bh": rec.bh,
                            "frame_w": rec.frame_w,
                            "frame_h": rec.frame_h,
                            "distance_estimate": round(rec.distance_estimate, 4),
                            "dist_mm": int(rec.dist_mm),
                            "vx": round(rec.vx, 4),
                            "vy": round(rec.vy, 4),
                            "vtheta": round(rec.vtheta, 4),
                        }
                        jf.write(json.dumps(meta, ensure_ascii=False) + "\n")

                        with self._state_lock:
                            self._saved_count += 1

            except Exception as exc:
                logger.error("ROISaver worker error: %s", exc)
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _status_locked(self) -> dict:
        if not self._session_id:
            return {"status": "idle", "dataset_type": "roi"}
        file_count = self._saved_count
        bytes_total = 0
        if self._session_dir and self._session_dir.exists():
            files = list(self._session_dir.glob("*.jpg"))
            file_count = len(files)
            bytes_total = sum(p.stat().st_size for p in files if p.is_file())
        return {
            "status": "recording" if self._collecting else "stopped",
            "dataset_type": "roi",
            "session_id": self._session_id,
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "frame_count": file_count,
            "bytes_total": bytes_total,
            "preview_count": min(file_count, 12),
            "preview_indexes": list(range(min(file_count, 12))),
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

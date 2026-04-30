import logging
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2

from ..perception.roi_extractor import PersonROI

logger = logging.getLogger(__name__)


class ROISaver:
    def __init__(self, save_dir: str, jpeg_quality: int = 90):
        self.root_dir = Path(save_dir)
        self.save_dir = self.root_dir
        self.jpeg_quality = jpeg_quality
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue(maxsize=100)
        self._running = False
        self._thread = None
        self._collecting = False
        self._session_id = ""
        self._session_dir: Path | None = None
        self._saved_count = 0
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._state_lock = threading.Lock()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("ROISaver background thread strictly assigned to %s", self.save_dir)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

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
            if hasattr(self, "_last_saved"):
                self._last_saved = {}
            self._drain_queue()
            logger.info("ROI dataset collection started: %s", self._session_dir)
            return self._status_locked()

    def stop_collection(self) -> dict:
        with self._state_lock:
            if self._collecting:
                self._collecting = False
                self._stopped_at = time.time()
                logger.info("ROI dataset collection stopped: %s", self._session_dir)
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
            return {"status": "discarded", "session_id": session_id}

    def status(self) -> dict:
        with self._state_lock:
            return self._status_locked()

    def push(self, rois: list[PersonROI], frame_id: int):
        with self._state_lock:
            collecting = self._collecting
        if not self._running or not collecting or not rois:
            return

        roi_data = []
        for r in rois:
            if not hasattr(self, "_last_saved"):
                self._last_saved = {}

            last_f = self._last_saved.get(r.track_id, -999)
            if frame_id - last_f >= 15:
                roi_data.append((r.track_id, r.image.copy()))
                self._last_saved[r.track_id] = frame_id

        if not roi_data:
            return

        try:
            self._queue.put_nowait((frame_id, roi_data))
        except queue.Full:
            pass

    def _worker(self):
        while self._running:
            try:
                frame_id, roi_data = self._queue.get(timeout=0.1)
                for track_id, img_array in roi_data:
                    with self._state_lock:
                        if not self._collecting or self._session_dir is None:
                            continue
                        filepath = self._session_dir / f"roi_t{track_id}_f{frame_id:06d}.jpg"
                        ok = cv2.imwrite(
                            str(filepath),
                            img_array,
                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                        )
                        if ok:
                            self._saved_count += 1
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("ROISaver writing failed: %s", e)

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
            except queue.Empty:
                return

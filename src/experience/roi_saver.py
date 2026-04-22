"""Daemon thread to save ROI images for pure data collection mode."""

import logging
import queue
import threading
from pathlib import Path

import cv2

from ..perception.roi_extractor import PersonROI

logger = logging.getLogger(__name__)

class ROISaver:
    """Saves ROI images asynchronously to avoid blocking main thread."""
    def __init__(self, save_dir: str, jpeg_quality: int = 90):
        self.save_dir = Path(save_dir)
        self.jpeg_quality = jpeg_quality
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue(maxsize=100)
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("ROISaver background thread strictly assigned to %s", self.save_dir)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def push(self, rois: list[PersonROI], frame_id: int):
        if not self._running or not rois:
            return

        # Chỉ lưu 1 ảnh sau mỗi 15 frame (tương đương ~2 ảnh/giây)
        # Việc này giúp tránh rác dữ liệu do lưu hàng chục ảnh giống hệt nhau trong 1 giây.
        if frame_id % 15 != 0:
            return

        # Snapshot necessary info
        roi_data = []
        for r in rois:
            roi_data.append((r.track_id, r.image.copy()))

        try:
            self._queue.put_nowait((frame_id, roi_data))
        except queue.Full:
            pass  # Drop if saving is too slow

    def _worker(self):
        while self._running:
            try:
                frame_id, roi_data = self._queue.get(timeout=0.1)
                for track_id, img_array in roi_data:
                    # Tên file gom track_id & frame_id. Đuôi .jpg
                    filepath = self.save_dir / f"roi_t{track_id}_f{frame_id:06d}.jpg"
                    cv2.imwrite(str(filepath), img_array, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("ROISaver writing failed: %s", e)

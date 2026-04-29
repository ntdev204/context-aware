from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .yolo_detector import DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class FaceAuthResult:
    matched: bool
    track_id: int
    user_id: int | None = None
    username: str | None = None
    face_id: str | None = None
    score: float = 0.0
    created_at: float = 0.0


class FaceAuthClient:
    def __init__(
        self,
        verify_url: str = "",
        shared_secret: str = "",
        min_interval_s: float = 2.0,
        jpeg_quality: int = 82,
    ) -> None:
        self.verify_url = verify_url
        self.shared_secret = shared_secret
        self.min_interval_s = min_interval_s
        self.jpeg_quality = jpeg_quality
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_submit_at = 0.0
        self._latest_result: FaceAuthResult | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self.verify_url or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="face-auth-client")
        self._thread.start()
        logger.info("Face auth client enabled: %s", self.verify_url)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def latest_result(self) -> FaceAuthResult | None:
        with self._lock:
            return self._latest_result

    def submit_open_palm(self, frame: np.ndarray, person: DetectionResult) -> bool:
        if not self.verify_url or person.track_id < 0:
            return False
        now = time.monotonic()
        if now - self._last_submit_at < self.min_interval_s:
            return False

        crop = self._crop_face(frame, person.bbox)
        if crop is None:
            return False
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return False

        payload = {
            "track_id": person.track_id,
            "gesture": "open_palm",
            "image_base64": base64.b64encode(buf.tobytes()).decode("ascii"),
        }
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(payload)
        self._last_submit_at = now
        return True

    @staticmethod
    def _crop_face(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        fx1 = max(0, x1 - int(bw * 0.12))
        fx2 = min(w, x2 + int(bw * 0.12))
        fy1 = max(0, y1 - int(bh * 0.06))
        fy2 = min(h, y1 + int(bh * 0.42))
        if fx2 <= fx1 or fy2 <= fy1:
            return None
        crop = frame[fy1:fy2, fx1:fx2]
        if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
            return None
        return crop

    def _run(self) -> None:
        while self._running:
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            result = self._post_verify(payload)
            if result is None:
                continue
            with self._lock:
                self._latest_result = result

    def _post_verify(self, payload: dict[str, Any]) -> FaceAuthResult | None:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.shared_secret:
            headers["X-Face-Auth-Token"] = self.shared_secret
        req = urllib.request.Request(self.verify_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=2.0) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning("Face auth verify failed: %s", exc)
            return None

        matched = bool(data.get("matched", False))
        track_id = int(data.get("track_id") if data.get("track_id") is not None else payload["track_id"])
        return FaceAuthResult(
            matched=matched,
            track_id=track_id,
            user_id=data.get("user_id"),
            username=data.get("username"),
            face_id=data.get("face_id"),
            score=float(data.get("score", 0.0) or 0.0),
            created_at=time.monotonic(),
        )

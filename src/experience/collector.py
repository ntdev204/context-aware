from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from ..navigation.context_builder import RobotState
from ..navigation.nav_command import NavigationCommand
from ..perception.intent_cnn import IntentPrediction
from ..perception.yolo_detector import FrameDetections
from .buffer import ExperienceBuffer

logger = logging.getLogger(__name__)

JPEG_QUALITY = 85


@dataclass
class ExperienceFrame:
    frame_id: int
    timestamp: float
    wall_time: float
    raw_image_jpeg: bytes
    detections: FrameDetections
    intent_predictions: list[IntentPrediction]
    observation: np.ndarray
    action: np.ndarray
    robot_state: RobotState
    session_id: str = ""


def _encode_action(cmd: NavigationCommand) -> np.ndarray:
    mode_oh = np.zeros(5, dtype=np.float32)
    mode_oh[int(cmd.mode)] = 1.0
    return np.array(
        [cmd.velocity_scale, cmd.heading_offset, *mode_oh],
        dtype=np.float32,
    )


class ExperienceCollector:
    def __init__(
        self,
        buffer: ExperienceBuffer,
        jpeg_quality: int = JPEG_QUALITY,
        enabled: bool = True,
        session_id: str = "",
    ) -> None:
        self._buffer = buffer
        self.jpeg_quality = jpeg_quality
        self.enabled = enabled
        self.session_id = session_id
        self._frame_id = 0
        self._dropped = 0
        self._lock = threading.Lock()

    def start_session(self, session_id: str) -> None:
        with self._lock:
            self.session_id = session_id
            self.enabled = True
            self._frame_id = 0
            self._dropped = 0

    def stop_session(self) -> None:
        with self._lock:
            self.enabled = False

    def collect(
        self,
        raw_frame: np.ndarray,
        frame_det: FrameDetections,
        intent_preds: list[IntentPrediction],
        observation: np.ndarray,
        cmd: NavigationCommand,
        robot_state: RobotState,
    ) -> ExperienceFrame | None:
        with self._lock:
            if not self.enabled:
                return None
            session_id = self.session_id

        ts = time.monotonic()
        wall = time.time()

        ok, buf = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            logger.warning("JPEG encode failed for dataset session %s", session_id)
            return None

        action = _encode_action(cmd)

        with self._lock:
            if not self.enabled or self.session_id != session_id:
                return None
            frame_id = self._frame_id
            self._frame_id += 1

            exp = ExperienceFrame(
                frame_id=frame_id,
                timestamp=ts,
                wall_time=wall,
                raw_image_jpeg=buf.tobytes(),
                detections=frame_det,
                intent_predictions=intent_preds,
                observation=observation.copy(),
                action=action,
                robot_state=robot_state,
                session_id=session_id,
            )

            pushed = self._buffer.push(exp)
            if not pushed:
                self._dropped += 1
            dropped = self._dropped

        if not pushed:
            if dropped % 100 == 0:
                logger.warning("Experience buffer full — %d frames dropped", dropped)

        return exp

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "session_id": self.session_id,
                "collected": self._frame_id,
                "dropped": self._dropped,
                "buffer_size": len(self._buffer),
            }

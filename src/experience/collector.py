"""CRITICAL: Data Logging Pipeline (Bottleneck C1).

ExperienceCollector gathers EVERY field needed for RL training per frame:
    raw_image       JPEG bytes  quality ≥ 85
    detections      list        from YOLO + tracker
    intent_preds    list        from CNN
    observation     np.ndarray  104-float vector (OBS_DIM=104)
    action          np.ndarray  7-float vector
    robot_state     RobotState  from ZMQ
    timestamp       float64     time.monotonic() for ordering
    frame_id        int         global sequential counter

All fields share the SAME timestamp to guarantee alignment.
Writing is async (pushed to ExperienceBuffer, written by a daemon thread).
"""

from __future__ import annotations

import logging
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
    """One fully-aligned data record per inference frame."""

    frame_id: int
    timestamp: float  # time.monotonic()
    wall_time: float  # time.time() for human readability
    raw_image_jpeg: bytes  # JPEG compressed
    detections: FrameDetections
    intent_predictions: list[IntentPrediction]
    observation: np.ndarray  # shape (104*k,)  — OBS_DIM=104
    action: np.ndarray  # shape (7,) = [v_scale, h_offset, mode_oh×5]
    robot_state: RobotState
    session_id: str = ""


def _encode_action(cmd: NavigationCommand) -> np.ndarray:
    """Flatten NavigationCommand into a 7-float vector."""
    mode_oh = np.zeros(5, dtype=np.float32)
    mode_oh[int(cmd.mode)] = 1.0
    return np.array(
        [cmd.velocity_scale, cmd.heading_offset, *mode_oh],
        dtype=np.float32,
    )


class ExperienceCollector:
    """Collects and enqueues experience frames for async HDF5 storage."""

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

    # ------------------------------------------------------------------
    # Main collection entry-point (called every inference frame)
    # ------------------------------------------------------------------
    def collect(
        self,
        raw_frame: np.ndarray,
        frame_det: FrameDetections,
        intent_preds: list[IntentPrediction],
        observation: np.ndarray,
        cmd: NavigationCommand,
        robot_state: RobotState,
    ) -> ExperienceFrame | None:
        if not self.enabled:
            return None

        ts = time.monotonic()
        wall = time.time()

        # JPEG compress
        ok, buf = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            logger.warning("JPEG encode failed, frame %d skipped", self._frame_id)
            return None

        action = _encode_action(cmd)

        exp = ExperienceFrame(
            frame_id=self._frame_id,
            timestamp=ts,
            wall_time=wall,
            raw_image_jpeg=buf.tobytes(),
            detections=frame_det,
            intent_predictions=intent_preds,
            observation=observation.copy(),
            action=action,
            robot_state=robot_state,
            session_id=self.session_id,
        )

        self._frame_id += 1

        # Non-blocking push to ring buffer
        if not self._buffer.push(exp):
            self._dropped += 1
            if self._dropped % 100 == 0:
                logger.warning("Experience buffer full — %d frames dropped", self._dropped)

        return exp

    @property
    def stats(self) -> dict:
        return {
            "collected": self._frame_id,
            "dropped": self._dropped,
            "buffer_size": len(self._buffer),
        }

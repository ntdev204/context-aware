"""Rule-based navigation policy with camera-driven motion disabled.

The Jetson perception service still detects people and obstacles for monitoring,
but it no longer generates movement from camera detections. Robot motion must
come from explicit teleop, Nav2, or other external control paths.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..perception.intent_cnn import IntentPrediction
from ..perception.yolo_detector import FrameDetections
from .nav_command import NavigationCommand, NavigationMode

if TYPE_CHECKING:
    from .context_builder import RobotState

logger = logging.getLogger(__name__)


class HeuristicPolicy:
    """STOP-only policy.

    Person tracking, hand activation, and camera-driven autonomous motion have
    been removed because the 640x480 camera stream is not reliable enough for
    motion control.
    """

    def __init__(self, *_, **__) -> None:
        self._observation: np.ndarray | None = None

    def decide(
        self,
        observation: np.ndarray,
        frame_det: FrameDetections,
        intent_preds: list[IntentPrediction],
        robot_state: RobotState | None = None,
    ) -> NavigationCommand:
        self._observation = observation
        return self._make(
            NavigationMode.STOP,
            velocity=0.0,
            heading=0.0,
            confidence=0.99,
            safety_override=True,
        )

    @staticmethod
    def _make(
        mode: NavigationMode,
        velocity: float,
        heading: float,
        confidence: float = 1.0,
        safety_override: bool = False,
    ) -> NavigationCommand:
        return NavigationCommand(
            mode=mode,
            velocity_scale=velocity,
            heading_offset=heading,
            confidence=confidence,
            safety_override=safety_override,
        ).clip()

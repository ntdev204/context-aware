"""Safety Monitor -- runs after every policy decision.

Safety cannot be learned; it must be engineered.

Layer 1 -- Output clipping (NavigationCommand.clip())
Layer 2 -- Proximity hard-stop + ERRATIC override (this file)
Layer 3 -- Watchdog timeout (zmq_subscriber.py)

This module never blocks the inference thread.
"""

from __future__ import annotations

import logging
import time

from ..perception.intent_cnn import ERRATIC, IntentPrediction
from ..perception.yolo_detector import FrameDetections
from .context_builder import RobotState
from .nav_command import NavigationCommand, NavigationMode

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """Enforces hard safety constraints on every NavigationCommand."""

    def __init__(
        self,
        hard_stop_person: float = 2.0,
        hard_stop_obstacle: float = 0.3,
        slow_down_distance: float = 3.0,
        slow_down_factor: float = 0.5,
        watchdog_timeout_ms: float = 500.0,
        battery_threshold: float = 10.0,
        watchdog_log_interval_s: float = 5.0,
        follow_min_distance: float = 0.5,
    ) -> None:
        self.hard_stop_person = hard_stop_person
        self.hard_stop_obstacle = hard_stop_obstacle
        self.slow_down_distance = slow_down_distance
        self.slow_down_factor = slow_down_factor
        self.watchdog_timeout_ms = watchdog_timeout_ms
        self.battery_threshold = battery_threshold
        self.watchdog_log_interval_s = watchdog_log_interval_s
        # Ngưỡng emergency stop thực sự trong follow mode (cho phép robot lùi khi người ở giữa)
        self.follow_min_distance = follow_min_distance

        self._last_robot_state_ts: float = time.monotonic()
        self._last_watchdog_log_ts: float = 0.0
        self._robot_state: RobotState | None = None

    def update_robot_state(self, state: RobotState) -> None:
        self._robot_state = state
        self._last_robot_state_ts = time.monotonic()

    def check(
        self,
        cmd: NavigationCommand,
        frame_det: FrameDetections,
        intent_preds: list[IntentPrediction],
    ) -> NavigationCommand:
        """Apply Layer 2 safety rules.  Returns (possibly modified) cmd."""

        elapsed_ms = (time.monotonic() - self._last_robot_state_ts) * 1000
        if elapsed_ms > self.watchdog_timeout_ms and cmd.mode != NavigationMode.STOP:
            self._log_watchdog(elapsed_ms)
            return self._emergency_stop(cmd, "watchdog_timeout")

        # Layer 3-b: Battery critical
        if (
            self._robot_state
            and self._robot_state.battery_percent < self.battery_threshold
            and cmd.mode != NavigationMode.STOP
        ):
            logger.warning("Battery critical (%.1f%%) — STOP", self._robot_state.battery_percent)
            return self._emergency_stop(cmd, "battery_critical")

        # Proximity distances from observation (de-normalised)
        nearest_person_dist = self._nearest_person_dist(frame_det)
        nearest_obstacle_dist = self._nearest_obstacle_dist(frame_det)

        # Hard stop: person too close.
        # Exception: nếu robot đang lùi (vel âm) để thoát ra xa — cho phép, chỉ dừng khi quá gần.
        is_reversing = cmd.velocity_scale < 0
        effective_stop_dist = self.follow_min_distance if is_reversing else self.hard_stop_person
        if frame_det.persons and nearest_person_dist < effective_stop_dist:
            logger.info(
                "Safety STOP: person at %.2f m (threshold=%.2f, reversing=%s)",
                nearest_person_dist,
                effective_stop_dist,
                is_reversing,
            )
            return self._emergency_stop(cmd, "person_proximity")

        # Hard stop: obstacle too close (Camera)
        if frame_det.obstacles and nearest_obstacle_dist < self.hard_stop_obstacle:
            logger.info("Safety STOP: camera obstacle at %.2f m", nearest_obstacle_dist)
            return self._emergency_stop(cmd, "obstacle_proximity")

        # Note: Lidar-based collision avoidance is fully handled by the Pi Safety Shield.
        # No duplicate check here to avoid false positives from noisy Lidar data.

        # ERRATIC intent override
        for pred in intent_preds:
            if pred.intent_class == ERRATIC and pred.confidence > 0.6:
                logger.warning(
                    "ERRATIC person (track=%d conf=%.2f) — STOP", pred.track_id, pred.confidence
                )
                return self._emergency_stop(cmd, "erratic_intent")

        # Slow-down zone (proportional reduction) — chỉ ảnh hưởng forward velocity
        if frame_det.persons and nearest_person_dist < self.slow_down_distance and not is_reversing:
            factor = nearest_person_dist / self.slow_down_distance * self.slow_down_factor
            if cmd.velocity_scale > factor:
                cmd.velocity_scale = factor
                cmd.safety_override = True
                logger.debug(
                    "Safety slow-down: v→%.2f (person at %.2f m)", factor, nearest_person_dist
                )

        return cmd.clip()

    def _log_watchdog(self, elapsed_ms: float) -> None:
        now = time.monotonic()
        if now - self._last_watchdog_log_ts >= self.watchdog_log_interval_s:
            logger.warning("Watchdog timeout (%.0f ms) -- STOP", elapsed_ms)
            self._last_watchdog_log_ts = now

    @staticmethod
    def _emergency_stop(cmd: NavigationCommand, reason: str) -> NavigationCommand:
        # User requested: Do not change mode to STOP so it can seamlessly resume FOLLOW
        # cmd.mode = NavigationMode.STOP 
        cmd.velocity_scale = 0.0
        cmd.velocity_y = 0.0
        cmd.heading_offset = 0.0
        cmd.safety_override = True
        cmd.confidence = 1.0
        return cmd

    @staticmethod
    def _nearest_person_dist(frame_det: FrameDetections) -> float:
        if not frame_det.persons:
            return 999.0
        
        dists = []
        for p in frame_det.persons:
            dist = p.distance
            # Fallback nếu depth camera bị nhiễu (depth = 0.0) giống heuristic_policy
            if dist < 0.15:
                import numpy as np
                x1, y1, x2, y2 = p.bbox
                bbox_h = max(1, y2 - y1)
                dist = float(np.clip(525.0 * 1.7 / bbox_h, 0.3, 5.0))
            dists.append(dist)
            
        return min(dists)

    @staticmethod
    def _nearest_obstacle_dist(frame_det: FrameDetections) -> float:
        if not frame_det.obstacles:
            return 999.0
        return min(d.distance for d in frame_det.obstacles)

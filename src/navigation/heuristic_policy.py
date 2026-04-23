"""Rule-based heuristic navigation policy (Phase 1 — no RL).

Replaces the RL policy while data is being collected.
All decision logic is deterministic and human-interpretable.
The output NavigationCommand interface is identical to what the RL policy
will produce in Phase 3, so the rest of the pipeline needs zero changes.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from ..perception.intent_cnn import APPROACHING, CROSSING, ERRATIC, IntentPrediction
from ..perception.yolo_detector import DetectionResult, FrameDetections
from .nav_command import NavigationCommand, NavigationMode

logger = logging.getLogger(__name__)


class HeuristicPolicy:
    """State-machine navigation policy driven by perception outputs.

    Decision priority (highest → lowest):
        1. Safety: person/obstacle too close → STOP
        2. ERRATIC intent detected → STOP
        3. CROSSING / APPROACHING → AVOID or slow down
        4. Free-space available → CRUISE
        5. Persons but not blocking → CAUTIOUS
        6. Follow mode active → FOLLOW
    """

    def __init__(
        self,
        cruise_free_space_threshold: float = 0.8,
        cruise_velocity: float = 1.0,
        cautious_velocity: float = 0.6,
        avoid_velocity: float = 0.3,
        follow_velocity: float = 0.5,
        hard_stop_distance: float = 0.3,
        slow_down_distance: float = 1.0,
        auto_follow: bool = False,
        follow_target_distance: float = 0.5,
        follow_deadband: float = 0.08,
        follow_kp: float = 1.0,
        target_lost_timeout_s: float = 2.0,
    ) -> None:
        self.cruise_threshold = cruise_free_space_threshold
        self.cruise_vel = cruise_velocity
        self.cautious_vel = cautious_velocity
        self.avoid_vel = avoid_velocity
        self.follow_vel = follow_velocity
        self.hard_stop_dist = hard_stop_distance
        self.slow_down_dist = slow_down_distance
        self.auto_follow = auto_follow
        self.follow_target_distance = follow_target_distance
        self.follow_deadband = follow_deadband
        self.follow_kp = follow_kp
        self.target_lost_timeout_s = target_lost_timeout_s

        self._follow_target_id: int = -1
        self._observation: np.ndarray | None = None
        self._target_last_seen: float = 0.0

    def set_follow_target(self, track_id: int) -> None:
        """Enable FOLLOW mode for the given track_id (-1 to disable)."""
        self._follow_target_id = track_id

    def decide(
        self,
        observation: np.ndarray,
        frame_det: FrameDetections,
        intent_preds: list[IntentPrediction],
    ) -> NavigationCommand:
        """Return a NavigationCommand based on pure rule logic."""
        self._observation = observation

        persons = frame_det.persons
        obstacles = frame_det.obstacles
        free_ratio = frame_det.free_space_ratio

        intent_map = {p.track_id: p for p in intent_preds}

        # Extract distances from observation vector (de-normalised)
        nearest_person_dist = float(observation[1]) * 5.0  # de-normalised
        nearest_obstacle_dist = float(observation[3]) * 5.0

        # RULE 1: Hard stop for proximity
        min(nearest_person_dist, nearest_obstacle_dist)
        if persons and nearest_person_dist < self.hard_stop_dist:
            return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.99)
        if obstacles and nearest_obstacle_dist < 0.3:
            return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.99)

        # RULE 2: ERRATIC intent → STOP
        for pred in intent_preds:
            if pred.intent_class == ERRATIC and pred.confidence > 0.6:
                logger.warning("ERRATIC intent detected (track=%d)", pred.track_id)
                return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.95)

        # RULE 3: CROSSING / APPROACHING → AVOID
        avoid_heading = 0.0
        blocking = False
        for person in persons:
            pred = intent_map.get(person.track_id)
            if pred and pred.intent_class in (CROSSING, APPROACHING) and pred.confidence > 0.5:
                blocking = True
                avoid_heading = self._compute_avoid_heading(person, frame_det)
                break

        if blocking:
            slow = max(self.avoid_vel, self.avoid_vel * (nearest_person_dist / self.slow_down_dist))
            return self._make(NavigationMode.AVOID, slow, avoid_heading, confidence=0.85)

        # RULE 4: AUTO-FOLLOW (bám người tự động)
        if self.auto_follow:
            return self._decide_follow(persons)

        # RULE 4b: FOLLOW mode (set thủ công từ ngoài)
        if self._follow_target_id >= 0:
            heading = self._heading_toward(self._follow_target_id, persons)
            return self._make(
                NavigationMode.FOLLOW,
                self.follow_vel,
                heading,
                confidence=0.80,
                follow_target_id=self._follow_target_id,
            )

        # RULE 5: Open space → CRUISE
        if free_ratio >= self.cruise_threshold and not persons:
            return self._make(NavigationMode.CRUISE, self.cruise_vel, 0.0, confidence=0.90)

        # RULE 6: Persons present but not blocking → CAUTIOUS
        if persons:
            vel = self.cautious_vel
            if nearest_person_dist < self.slow_down_dist:
                vel *= nearest_person_dist / self.slow_down_dist
            return self._make(NavigationMode.CAUTIOUS, vel, 0.0, confidence=0.75)

        # Default: CRUISE (empty scene)
        return self._make(NavigationMode.CRUISE, self.cruise_vel, 0.0, confidence=0.70)

    def _decide_follow(self, persons: list[DetectionResult]) -> NavigationCommand:
        """Logic bám người tự động (single-target tracking).

        - Nếu chưa có target: lock vào người gần nhất.
        - Nếu đã có target: tính hướng và tốc độ hướng tới.
        - Nếu target bị mất > timeout: giải phóng target, chờ người mới.
        """
        now = time.monotonic()
        active_ids = {p.track_id for p in persons}

        # --- Target đang bị mất ---
        if self._follow_target_id >= 0 and self._follow_target_id not in active_ids:
            lost_for = now - self._target_last_seen
            if lost_for > self.target_lost_timeout_s:
                logger.info(
                    "Follow target %d lost for %.1fs — releasing.",
                    self._follow_target_id,
                    lost_for,
                )
                self._follow_target_id = -1
            else:
                # Chờ target quay lại — đứng yên (STOP)
                return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.90)

        # --- Chưa có target → lock vào người gần nhất ---
        if self._follow_target_id < 0 and persons:
            target = min(persons, key=lambda p: p.distance)
            self._follow_target_id = target.track_id
            self._target_last_seen = now
            logger.info("Auto-follow: locked onto track_id=%d", self._follow_target_id)

        # --- Không có ai trong khung hình ---
        if self._follow_target_id < 0:
            return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.70)

        # --- Đang theo dõi target ---
        self._target_last_seen = now
        target_person = next((p for p in persons if p.track_id == self._follow_target_id), None)
        if target_person is None:
            return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.90)

        heading = self._heading_toward(self._follow_target_id, persons)
        dist = target_person.distance

        # Hard stop tuyệt đối
        if dist <= self.hard_stop_dist:
            return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.99)

        # P-Controller: error = dist - target_distance
        #   error > 0 (quá xa)  → vel dương → tiến
        #   error < 0 (quá gần) → vel âm   → lùi
        error = dist - self.follow_target_distance
        if abs(error) < self.follow_deadband:
            vel = 0.0  # dead-band: đứng yên, tránh rung lắc
        else:
            vel = float(np.clip(self.follow_kp * error, -self.follow_vel, self.follow_vel))

        return self._make(
            NavigationMode.FOLLOW,
            vel,
            heading,
            confidence=0.88,
            follow_target_id=self._follow_target_id,
        )

    @staticmethod
    def _make(
        mode: NavigationMode,
        velocity: float,
        heading: float,
        confidence: float = 1.0,
        follow_target_id: int = -1,
    ) -> NavigationCommand:
        return NavigationCommand(
            mode=mode,
            velocity_scale=velocity,
            heading_offset=heading,
            follow_target_id=follow_target_id,
            confidence=confidence,
        ).clip()

    @staticmethod
    def _compute_avoid_heading(blocker: DetectionResult, frame_det: FrameDetections) -> float:
        """Steer away from blocking person: if they are left → steer right."""
        frame_mid = 640 / 2.0  # assume 640px width
        cx = (blocker.bbox[0] + blocker.bbox[2]) / 2.0
        # Nudge opposite to blocker's position
        raw = -(cx - frame_mid) / frame_mid * math.radians(25)
        return float(np.clip(raw, -math.pi / 4, math.pi / 4))

    @staticmethod
    def _heading_toward(target_id: int, persons: list[DetectionResult]) -> float:
        for p in persons:
            if p.track_id == target_id:
                frame_mid = 640 / 2.0
                cx = (p.bbox[0] + p.bbox[2]) / 2.0
                return float(
                    np.clip(
                        (cx - frame_mid) / frame_mid * math.radians(20),
                        -math.pi / 4,
                        math.pi / 4,
                    )
                )
        return 0.0

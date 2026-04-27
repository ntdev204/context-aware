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
        follow_max_vel: float = 0.8,
        follow_min_vel: float = 0.3,
        hard_stop_distance: float = 2.0,
        slow_down_distance: float = 3.0,
        auto_follow: bool = False,
        follow_target_distance: float = 2.0,
        follow_deadband: float = 0.08,
        follow_kp: float = 1.0,
        target_lost_timeout_s: float = 300.0,
        follow_min_distance: float = 0.5,
    ) -> None:
        self.cruise_threshold = cruise_free_space_threshold
        self.cruise_vel = cruise_velocity
        self.cautious_vel = cautious_velocity
        self.avoid_vel = avoid_velocity
        # Follow velocity range — robot scales between these based on context
        self.follow_max_vel = follow_max_vel
        self.follow_min_vel = follow_min_vel
        self.hard_stop_dist = hard_stop_distance
        self.slow_down_dist = slow_down_distance
        self.auto_follow = auto_follow
        self.follow_target_distance = follow_target_distance
        self.follow_deadband = follow_deadband
        self.follow_kp = follow_kp
        self.target_lost_timeout_s = target_lost_timeout_s
        # Emergency stop khi người quá gần trong follow mode (tách khỏi follow_target_distance)
        self.follow_min_distance = follow_min_distance

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
        robot_state: RobotState | None = None,
    ) -> NavigationCommand:
        """Return a NavigationCommand based on pure rule logic."""
        self._observation = observation

        persons = frame_det.persons
        obstacles = frame_det.obstacles
        free_ratio = frame_det.free_space_ratio

        intent_map = {p.track_id: p for p in intent_preds}

        nearest_person_dist = float(observation[1]) * 5.0
        nearest_obstacle_dist = float(observation[3]) * 5.0

        # Trong follow mode: delegate NGAY cho _decide_follow() — nó tự xử lý emergency stop
        # tại follow_min_distance (0.5m). KHÔNG dùng hard_stop_dist (2.0m) ở đây vì sẽ
        # chặn lệnh lùi ra xa khi người đứng trong khoảng (0.5m, 2.0m).
        if self.auto_follow:
            if obstacles and nearest_obstacle_dist < 0.3:
                return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.99)
            
            # Mecanum: dùng strafe (velocity_y) để né tường, KHÔNG xoay đầu
            strafe = 0.0
            if robot_state:
                # Ưu tiên né tường trước (override strafe bám người)
                if robot_state.dist_left < 0.4:
                    strafe = -0.4   # trượt phải khi bên trái có tường
                elif robot_state.dist_right < 0.4:
                    strafe = 0.4    # trượt trái khi bên phải có tường
            
            cmd = self._decide_follow(persons, free_ratio, intent_map)
            # Nếu không đang né tường, dùng strafe để bám người ngang
            if strafe == 0.0:
                strafe = self._lateral_strafe_to_target(self._follow_target_id, persons, free_ratio)
            cmd.velocity_y = strafe
            return cmd

        if persons and nearest_person_dist < self.hard_stop_dist:
            return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.99)
        if obstacles and nearest_obstacle_dist < 0.3:
            return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.99)

        for pred in intent_preds:
            if pred.intent_class == ERRATIC and pred.confidence > 0.6:
                logger.warning("ERRATIC intent detected (track=%d)", pred.track_id)
                return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.95)

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

        if self._follow_target_id >= 0:
            heading = self._heading_toward(self._follow_target_id, persons)
            return self._make(
                NavigationMode.FOLLOW,
                self.follow_max_vel,
                heading,
                confidence=0.80,
                follow_target_id=self._follow_target_id,
            )

        if free_ratio >= self.cruise_threshold and not persons:
            return self._make(NavigationMode.CRUISE, self.cruise_vel, 0.0, confidence=0.90)

        if persons:
            vel = self.cautious_vel
            if nearest_person_dist < self.slow_down_dist:
                vel *= nearest_person_dist / self.slow_down_dist
            return self._make(NavigationMode.CAUTIOUS, vel, 0.0, confidence=0.75)

        return self._make(NavigationMode.CRUISE, self.cruise_vel, 0.0, confidence=0.70)

    def _decide_follow(
        self,
        persons: list[DetectionResult],
        free_ratio: float,
        intent_map: dict,
    ) -> NavigationCommand:
        """Context-aware person following: velocity scales with distance, free space, and intent.

        Velocity formula (forward):
            vel = follow_min_vel
                + (follow_max_vel - follow_min_vel)
                × dist_factor    # |error| / target_dist, capped [0, 1]
                × space_factor   # 0.7 + 0.3 × free_ratio (open space → faster)
                × intent_factor  # 0.65 when target is APPROACHING robot (slow down)
        Reverse (too close): mirrors forward but ignores space_factor.
        """
        now = time.monotonic()
        active_ids = {p.track_id for p in persons}

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
                return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.90)

        if self._follow_target_id < 0 and persons:
            target = min(persons, key=lambda p: p.distance)
            self._follow_target_id = target.track_id
            self._target_last_seen = now
            logger.info("Auto-follow: locked onto track_id=%d", self._follow_target_id)

        if self._follow_target_id < 0:
            return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.70)

        self._target_last_seen = now
        target_person = next((p for p in persons if p.track_id == self._follow_target_id), None)
        if target_person is None:
            return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.90)

        dist = target_person.distance

        DEPTH_INVALID_THRESHOLD = 0.15
        if dist < DEPTH_INVALID_THRESHOLD:
            x1, y1, x2, y2 = target_person.bbox
            bbox_h = max(1, y2 - y1)
            bbox_w = max(1, x2 - x1)
            # Camera bị mù ở < 0.4m, nhưng nếu người rất gần thì bbox sẽ rất lớn (do chỉ nhìn thấy phần thân)
            # Công thức tính 1.7m sẽ sai bét nếu không thấy toàn thân.
            if bbox_h > 350 or bbox_w > 300:
                dist = 0.3  # Chắc chắn ở rất gần (< 0.5m), ép dừng khẩn cấp
                logger.debug("Depth invalid & bbox HUGE (h=%d) -> FORCE STOP (0.3m)", bbox_h)
            else:
                dist = float(np.clip(525.0 * 1.7 / bbox_h, 0.3, 5.0))
                logger.debug("Depth invalid, using bbox fallback: dist=%.2fm", dist)

        # Emergency stop chỉ khi quá gần (follow_min_distance). Khoảng (min, target) → lùi.
        if dist <= self.follow_min_distance:
            logger.warning(
                "Follow emergency stop: person at %.2fm (min=%.2fm)", dist, self.follow_min_distance
            )
            return self._make(NavigationMode.STOP, 0.0, 0.0, confidence=0.99)

        # Mecanum follow: KHÔNG xoay đầu, giữ heading_offset = 0
        # Lateral centering được xử lý bởi velocity_y ở tầng gọi hàm này
        error = dist - self.follow_target_distance
        if abs(error) < self.follow_deadband:
            vel = 0.0
        else:
            vel = self._context_velocity(error, free_ratio, intent_map)

        logger.debug(
            "Follow: dist=%.2fm error=%.2fm vel=%.2f heading=0 (strafe mode)",
            dist, error, vel,
        )
        return self._make(
            NavigationMode.FOLLOW,
            vel,
            0.0,             # heading_offset = 0: robot luôn giữ hướng, dùng vy để bám ngang
            confidence=0.88,
            follow_target_id=self._follow_target_id,
        )

    def _context_velocity(self, error: float, free_ratio: float, intent_map: dict) -> float:
        """Compute context-aware velocity magnitude × direction.

        Args:
            error: dist - follow_target_distance  (+ = too far, - = too close)
            free_ratio: fraction of frame that is free space [0, 1]
            intent_map: track_id → IntentPrediction

        Returns:
            Signed velocity in [-follow_max_vel, -follow_min_vel] ∪ {0} ∪ [follow_min_vel, follow_max_vel]
        """
        going_forward = error > 0

        # 1. Distance factor: how urgently must we move? [0, 1]
        dist_factor = float(
            np.clip(abs(error) / max(self.follow_target_distance * 0.8, 0.01), 0.0, 1.0)
        )

        # 2. Free-space factor: more open space → faster forward (reverse always at full scale)
        space_factor = (0.7 + 0.3 * float(np.clip(free_ratio, 0.0, 1.0))) if going_forward else 1.0

        # 3. Intent factor: target approaching robot → slow down (they'll close the gap themselves)
        intent_factor = 1.0
        target_pred = intent_map.get(self._follow_target_id)
        if target_pred and target_pred.intent_class == APPROACHING and going_forward:
            intent_factor = 0.65
            logger.debug("Intent APPROACHING: reducing forward vel (×0.65)")

        # 4. Combine into magnitude, enforce [min_vel, max_vel] range
        vel_range = self.follow_max_vel - self.follow_min_vel
        vel_magnitude = self.follow_min_vel + vel_range * dist_factor * space_factor * intent_factor
        vel_magnitude = float(np.clip(vel_magnitude, self.follow_min_vel, self.follow_max_vel))

        return vel_magnitude if going_forward else -vel_magnitude

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

    def _lateral_strafe_to_target(self, target_id: int, persons: list[DetectionResult], free_ratio: float) -> float:
        """Tính velocity_y để căn giữa robot theo vị trí ngang của người (Mecanum strafe).
        
        Quy ước ROS:
            linear.y > 0 = trượt TRÁI (robot's left)
            linear.y < 0 = trượt PHẢI (robot's right)
        """
        for p in persons:
            if p.track_id == target_id:
                frame_mid = 640 / 2.0
                cx = (p.bbox[0] + p.bbox[2]) / 2.0
                lateral_err = (cx - frame_mid) / frame_mid  # [-1, 1], + = người bên phải
                
                # Nếu lệch ít (< 10%) thì không trượt ngang để tránh xe lắc lư liên tục (deadband)
                if abs(lateral_err) < 0.10:
                    return 0.0

                # Hệ số không gian: thoáng (1.0) -> trượt nhanh hơn, hẹp (0) -> trượt cẩn thận
                space_factor = 0.7 + 0.3 * float(np.clip(free_ratio, 0.0, 1.0))
                
                # Đảm bảo vy luôn nằm trong khoảng [follow_min_vel, follow_max_vel]
                # giống y hệt như vx để xe di chuyển đồng bộ
                vy_range = self.follow_max_vel - self.follow_min_vel
                vy_mag = self.follow_min_vel + vy_range * abs(lateral_err) * space_factor
                vy_mag = float(np.clip(vy_mag, self.follow_min_vel, self.follow_max_vel))

                # Người bên phải (+ err) -> cần trượt phải (vy âm)
                vy = -vy_mag if lateral_err > 0 else vy_mag
                logger.debug("Lateral strafe: err=%.2f space=%.2f vy=%.2f", lateral_err, space_factor, vy)
                return vy
        return 0.0

    @staticmethod
    def _compute_avoid_heading(blocker: DetectionResult, frame_det: FrameDetections) -> float:
        """Steer away from blocking person: if they are left → steer right."""
        frame_mid = 640 / 2.0
        cx = (blocker.bbox[0] + blocker.bbox[2]) / 2.0
        raw = -(cx - frame_mid) / frame_mid * math.radians(25)
        return float(np.clip(raw, -math.pi / 4, math.pi / 4))

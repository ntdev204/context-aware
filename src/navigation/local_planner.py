"""Short-horizon local planner for gesture follow mode.

The planner is intentionally smaller than Nav2: it works in robot-frame metres,
fuses low obstacles from full 360-degree lidar scan with high obstacles from camera depth,
and recomputes a short plan every perception frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..perception.yolo_detector import DetectionResult, FrameDetections

if TYPE_CHECKING:
    from .context_builder import RobotState


HORIZONTAL_FOV_RAD = math.radians(70.0)
UNKNOWN_CLEARANCE_M = 9.9


@dataclass(frozen=True)
class PlanWaypoint:
    x: float
    y: float


@dataclass(frozen=True)
class PlanObstacle:
    x: float
    y: float
    radius: float
    source: str
    label: str


@dataclass
class LocalPlan:
    status: str = "idle"  # idle|no_target|no_depth|clear|replanned|blocked
    waypoints: list[PlanWaypoint] = field(default_factory=list)
    obstacles: list[PlanObstacle] = field(default_factory=list)
    target_track_id: int = -1
    target_distance: float = 0.0
    target_angle: float = 0.0
    blocked_by: str = ""
    lidar_sectors: tuple[float, float, float, float] | None = None
    lidar_scan: tuple[float, ...] = field(default_factory=tuple)

    @property
    def blocked(self) -> bool:
        return self.status == "blocked"

    @property
    def replanned(self) -> bool:
        return self.status == "replanned"

    def first_waypoint(self) -> PlanWaypoint | None:
        return self.waypoints[0] if self.waypoints else None

    def to_payload(self) -> dict:
        return {
            "status": self.status,
            "target_track_id": self.target_track_id,
            "target_distance": self.target_distance,
            "target_angle": self.target_angle,
            "blocked_by": self.blocked_by,
            "waypoints": [{"x": p.x, "y": p.y} for p in self.waypoints],
            "obstacles": [
                {
                    "x": ob.x,
                    "y": ob.y,
                    "radius": ob.radius,
                    "source": ob.source,
                    "label": ob.label,
                }
                for ob in self.obstacles
            ],
            "lidar_sectors": list(self.lidar_sectors) if self.lidar_sectors else None,
            "lidar_scan_count": len(self.lidar_scan),
        }


class LocalPlanner:
    """Plan a short robot-frame path from the robot toward the followed person."""

    def __init__(
        self,
        enabled: bool = True,
        follow_target_distance: float = 2.0,
        max_plan_distance: float = 5.0,
        robot_radius: float = 0.25,
        safety_margin: float = 0.20,
        corridor_width: float = 0.85,
        detour_offsets: list[float] | tuple[float, ...] = (0.75, 1.10),
        lidar_block_distance: float = 0.65,
        camera_obstacle_radius: float = 0.35,
    ) -> None:
        self.enabled = enabled
        self.follow_target_distance = float(follow_target_distance)
        self.max_plan_distance = float(max_plan_distance)
        self.robot_radius = float(robot_radius)
        self.safety_margin = float(safety_margin)
        self.corridor_width = float(corridor_width)
        self.detour_offsets = [float(v) for v in detour_offsets]
        self.lidar_block_distance = float(lidar_block_distance)
        self.camera_obstacle_radius = float(camera_obstacle_radius)

    def plan(
        self,
        frame_det: FrameDetections,
        target: DetectionResult | None,
        robot_state: RobotState | None,
    ) -> LocalPlan:
        if not self.enabled:
            return LocalPlan(status="idle")
        if target is None:
            return LocalPlan(status="no_target")
        if getattr(target, "stale", False):
            return LocalPlan(status="no_target", target_track_id=target.track_id)

        target_distance = self._valid_distance(target.distance)
        if target_distance is None:
            return LocalPlan(
                status="no_depth",
                target_track_id=target.track_id,
                target_angle=self._angle_from_bbox(target, frame_det.frame_width),
            )

        target_angle = self._angle_from_bbox(target, frame_det.frame_width)
        goal = self._goal_from_target(target_distance, target_angle)
        lidar_scan = self._lidar_scan(robot_state)
        lidar_sectors = self._lidar_sectors(robot_state)
        obstacles = self._camera_obstacles(frame_det, target.track_id)

        if self._path_is_clear([goal], obstacles, lidar_scan, lidar_sectors):
            return LocalPlan(
                status="clear",
                waypoints=[goal],
                obstacles=obstacles,
                target_track_id=target.track_id,
                target_distance=target_distance,
                target_angle=target_angle,
                lidar_sectors=lidar_sectors,
                lidar_scan=lidar_scan or (),
            )

        for waypoints in self._candidate_detours(goal, target_angle, lidar_scan, lidar_sectors):
            if self._path_is_clear(waypoints, obstacles, lidar_scan, lidar_sectors):
                return LocalPlan(
                    status="replanned",
                    waypoints=waypoints,
                    obstacles=obstacles,
                    target_track_id=target.track_id,
                    target_distance=target_distance,
                    target_angle=target_angle,
                    blocked_by="direct_path",
                    lidar_sectors=lidar_sectors,
                    lidar_scan=lidar_scan or (),
                )

        return LocalPlan(
            status="blocked",
            waypoints=[],
            obstacles=obstacles,
            target_track_id=target.track_id,
            target_distance=target_distance,
            target_angle=target_angle,
            blocked_by="no_safe_corridor",
            lidar_sectors=lidar_sectors,
            lidar_scan=lidar_scan or (),
        )

    def _goal_from_target(self, distance: float, angle: float) -> PlanWaypoint:
        distance = min(distance, self.max_plan_distance)
        target_x = distance * math.cos(angle)
        target_y = distance * math.sin(angle)
        goal_x = max(0.0, target_x - self.follow_target_distance)
        return PlanWaypoint(goal_x, target_y)

    def _candidate_detours(
        self,
        goal: PlanWaypoint,
        target_angle: float,
        lidar_scan: tuple[float, ...] | None,
        lidar_sectors: tuple[float, float, float, float] | None,
    ) -> list[list[PlanWaypoint]]:
        first_x = min(max(goal.x * 0.35, 0.25), 0.9)
        front_dist = self._ray_distance(lidar_scan, 0.0) if lidar_scan else None
        if front_dist is None and lidar_sectors:
            front_dist = lidar_sectors[0]
        if front_dist is not None and front_dist < self.lidar_block_distance:
            first_x = 0.0

        side_order = self._side_order(target_angle, lidar_scan, lidar_sectors)
        candidates: list[list[PlanWaypoint]] = []
        for sign in side_order:
            for offset in self.detour_offsets:
                y = sign * offset
                candidates.append(self._dedupe_waypoints([
                    PlanWaypoint(first_x, y),
                    PlanWaypoint(max(goal.x, first_x), y),
                ]))
        return candidates

    def _path_is_clear(
        self,
        waypoints: list[PlanWaypoint],
        obstacles: list[PlanObstacle],
        lidar_scan: tuple[float, ...] | None,
        lidar_sectors: tuple[float, float, float, float] | None,
    ) -> bool:
        prev = PlanWaypoint(0.0, 0.0)
        for point in waypoints:
            if self._segment_blocked_by_camera(prev, point, obstacles):
                return False
            if self._segment_blocked_by_lidar(prev, point, lidar_scan, lidar_sectors):
                return False
            prev = point
        return True

    def _segment_blocked_by_camera(
        self,
        a: PlanWaypoint,
        b: PlanWaypoint,
        obstacles: list[PlanObstacle],
    ) -> bool:
        for ob in obstacles:
            clearance = self.robot_radius + self.safety_margin + ob.radius
            if self._distance_point_to_segment(ob.x, ob.y, a, b) <= clearance:
                return True
        return False

    def _segment_blocked_by_lidar(
        self,
        a: PlanWaypoint,
        b: PlanWaypoint,
        lidar_scan: tuple[float, ...] | None,
        lidar_sectors: tuple[float, float, float, float] | None,
    ) -> bool:
        if lidar_scan is None and lidar_sectors is None:
            return False

        clearance = self.robot_radius + self.safety_margin
        for i in range(1, 13):
            t = i / 12.0
            x = a.x + (b.x - a.x) * t
            y = a.y + (b.y - a.y) * t
            dist = math.hypot(x, y)
            if dist <= 1e-6:
                continue
            angle = math.atan2(y, x)
            sector_dist = self._ray_distance(lidar_scan, angle)
            if sector_dist is None:
                assert lidar_sectors is not None
                sector_dist = self._sector_distance(lidar_sectors, angle)
            if -45.0 <= math.degrees(angle) <= 45.0:
                if abs(y) > self.corridor_width * 0.5:
                    continue
            if sector_dist < UNKNOWN_CLEARANCE_M and dist >= max(0.0, sector_dist - clearance):
                return True
        return False

    def _camera_obstacles(
        self,
        frame_det: FrameDetections,
        target_track_id: int,
    ) -> list[PlanObstacle]:
        obstacles: list[PlanObstacle] = []
        candidates = list(frame_det.obstacles)
        candidates.extend(p for p in frame_det.persons if p.track_id != target_track_id)

        for det in candidates:
            if getattr(det, "stale", False):
                continue
            distance = self._valid_distance(det.distance)
            if distance is None or distance > self.max_plan_distance:
                continue
            angle = self._angle_from_bbox(det, frame_det.frame_width)
            obstacles.append(
                PlanObstacle(
                    x=distance * math.cos(angle),
                    y=distance * math.sin(angle),
                    radius=self.camera_obstacle_radius,
                    source="camera",
                    label=det.class_name,
                )
            )
        return obstacles

    @staticmethod
    def _angle_from_bbox(det: DetectionResult, frame_width: int) -> float:
        frame_width = frame_width or 640
        center_x = (det.bbox[0] + det.bbox[2]) * 0.5
        offset = (center_x - frame_width * 0.5) / max(frame_width * 0.5, 1.0)
        return float(np.clip(offset, -1.0, 1.0) * (HORIZONTAL_FOV_RAD * 0.5))

    @staticmethod
    def _valid_distance(value: float) -> float | None:
        try:
            distance = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(distance) or distance <= 0.15:
            return None
        return distance

    @staticmethod
    def _lidar_sectors(robot_state: RobotState | None) -> tuple[float, float, float, float] | None:
        if robot_state is None:
            return None
        values = (
            getattr(robot_state, "lidar_front", UNKNOWN_CLEARANCE_M),
            getattr(robot_state, "lidar_rear", UNKNOWN_CLEARANCE_M),
            getattr(robot_state, "lidar_left", UNKNOWN_CLEARANCE_M),
            getattr(robot_state, "lidar_right", UNKNOWN_CLEARANCE_M),
        )
        sectors = tuple(LocalPlanner._normalise_lidar_distance(v) for v in values)
        return sectors

    @staticmethod
    def _lidar_scan(robot_state: RobotState | None) -> tuple[float, ...] | None:
        if robot_state is None:
            return None
        values = tuple(
            LocalPlanner._normalise_lidar_distance(v)
            for v in getattr(robot_state, "lidar_scan", ()) or ()
        )
        return values if len(values) >= 36 else None

    @staticmethod
    def _normalise_lidar_distance(value: float) -> float:
        try:
            distance = float(value)
        except (TypeError, ValueError):
            return UNKNOWN_CLEARANCE_M
        if not math.isfinite(distance) or distance <= 0.0:
            return UNKNOWN_CLEARANCE_M
        return min(distance, UNKNOWN_CLEARANCE_M)

    @staticmethod
    def _sector_distance(sectors: tuple[float, float, float, float], angle: float) -> float:
        deg = math.degrees(angle)
        front, rear, left, right = sectors
        if -45.0 <= deg <= 45.0:
            return front
        if 45.0 < deg < 135.0:
            return left
        if -135.0 < deg < -45.0:
            return right
        return rear

    @staticmethod
    def _ray_distance(
        scan: tuple[float, ...] | None,
        angle: float,
        window_deg: float = 4.0,
    ) -> float | None:
        if not scan:
            return None
        n = len(scan)
        deg = math.degrees(angle) % 360.0
        idx = int(round(deg / 360.0 * n)) % n
        half = max(1, int(round(window_deg / max(360.0 / n, 1e-6))))
        values = [scan[(idx + offset) % n] for offset in range(-half, half + 1)]
        valid = [v for v in values if 0.0 < v < UNKNOWN_CLEARANCE_M]
        return min(valid) if valid else UNKNOWN_CLEARANCE_M

    @staticmethod
    def _band_clearance(
        scan: tuple[float, ...] | None,
        start_deg: float,
        end_deg: float,
    ) -> float | None:
        if not scan:
            return None
        n = len(scan)
        start = int(round((start_deg % 360.0) / 360.0 * n)) % n
        end = int(round((end_deg % 360.0) / 360.0 * n)) % n
        if start <= end:
            values = scan[start : end + 1]
        else:
            values = scan[start:] + scan[: end + 1]
        valid = np.asarray([v for v in values if 0.0 < v < UNKNOWN_CLEARANCE_M], dtype=np.float32)
        if valid.size == 0:
            return UNKNOWN_CLEARANCE_M
        return float(np.percentile(valid, 20))

    @staticmethod
    def _side_order(
        target_angle: float,
        lidar_scan: tuple[float, ...] | None,
        lidar_sectors: tuple[float, float, float, float] | None,
    ) -> list[int]:
        if lidar_scan:
            left_clearance = LocalPlanner._band_clearance(lidar_scan, 45.0, 135.0)
            right_clearance = LocalPlanner._band_clearance(lidar_scan, 225.0, 315.0)
            if left_clearance is not None and right_clearance is not None:
                if left_clearance == right_clearance:
                    return [1, -1] if target_angle >= 0.0 else [-1, 1]
                return [1, -1] if left_clearance > right_clearance else [-1, 1]
        if lidar_sectors is None:
            return [1, -1] if target_angle >= 0.0 else [-1, 1]
        _, _, left, right = lidar_sectors
        if left == right:
            return [1, -1] if target_angle >= 0.0 else [-1, 1]
        return [1, -1] if left > right else [-1, 1]

    @staticmethod
    def _dedupe_waypoints(points: list[PlanWaypoint]) -> list[PlanWaypoint]:
        result: list[PlanWaypoint] = []
        for point in points:
            if result and math.hypot(point.x - result[-1].x, point.y - result[-1].y) < 0.05:
                continue
            result.append(point)
        return result

    @staticmethod
    def _distance_point_to_segment(
        px: float,
        py: float,
        a: PlanWaypoint,
        b: PlanWaypoint,
    ) -> float:
        ax, ay, bx, by = a.x, a.y, b.x, b.y
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

"""Context State Builder — constructs the observation vector from perception outputs.

CRITICAL DESIGN RULE (Bottleneck B1):
    The observation space MUST be temporal-ready from day 1.
    Phase 1 uses temporal_stack_size=1 (snapshot), but the ring-buffer
    and get_stacked_observation() API are already in place so Phase 2
    can flip k→3 in config with ZERO code changes.

Observation vector layout (114 floats when k=1):
    [0]      num_persons               (float, 0-10 normalised)
    [1]      nearest_person_distance   (float, metres, clamped 0-5)
    [2]      nearest_person_angle      (float, radians)
    [3]      nearest_obstacle_distance (float, metres)
    [4]      nearest_obstacle_angle    (float, radians)
    [5]      reserved                  (float, currently 0.0)
    [6-69]   occupancy_grid            (8×8 = 64 floats, 0=free 1=occupied)
    [70-93]  person_intents            (3 persons × 8 features)
    [94-96]  robot_velocity            (vx, vy, vθ)
    [97-103] previous_action           (velocity_scale, heading_offset, mode×5 one-hot)
    [104-113] reserved                 (10 floats, currently 0.0)
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..perception.intent_cnn import IntentPrediction
from ..perception.yolo_detector import DetectionResult, FrameDetections
from .nav_command import NavigationCommand

logger = logging.getLogger(__name__)

OBS_DIM = 114
GRID_SIZE = 8  # 8×8 occupancy grid
MAX_PERSONS = 3  # top-k persons included in observation
INTENT_FEATS = 8  # 6 intent probs + dx + dy per person
MAX_DIST = 5.0  # metres — clamp for normalisation
NUM_MODES = 5


@dataclass
class RobotState:
    vx: float = 0.0
    vy: float = 0.0
    vtheta: float = 0.0
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_theta: float = 0.0
    battery_percent: float = 100.0
    nav2_status: str = "idle"
    lidar_front: float = 9.9
    lidar_rear: float = 9.9
    lidar_left: float = 9.9
    lidar_right: float = 9.9
    lidar_scan: tuple[float, ...] = field(default_factory=tuple)
    timestamp: float = field(default_factory=time.time)

    @property
    def lidar_sectors(self) -> tuple[float, float, float, float]:
        """Return [front, rear, left, right] lidar sector distances in metres."""
        return (self.lidar_front, self.lidar_rear, self.lidar_left, self.lidar_right)


class ContextBuilder:
    """Builds temporal-ready observation vectors for the navigation policy.

    Parameters
    ----------
    temporal_stack_size:
        Number of consecutive frames to stack.  k=1 = snapshot (Phase 1).
    state_version:
        Semantic version string — stored in logs so data can be filtered by
        observation layout when replaying for RL training.
    """

    def __init__(
        self,
        temporal_stack_size: int = 1,
        state_version: str = "v2-camera",
        occupancy_grid_size: int = GRID_SIZE,
    ) -> None:
        self.temporal_stack_size = temporal_stack_size
        self.state_version = state_version
        self.grid_size = occupancy_grid_size

        self._history: deque[np.ndarray] = deque(maxlen=temporal_stack_size)
        self._prev_action: NavigationCommand | None = None
        self._robot_state = RobotState()

    # ------------------------------------------------------------------
    # External API
    # ------------------------------------------------------------------
    def update_robot_state(self, state: RobotState) -> None:
        self._robot_state = state

    def update_prev_action(self, cmd: NavigationCommand) -> None:
        self._prev_action = cmd

    def build(
        self,
        frame_det: FrameDetections,
        intent_preds: list[IntentPrediction],
    ) -> np.ndarray:
        """Build one observation snapshot and push to history ring-buffer.

        Returns the stacked observation (shape: (OBS_DIM * k,)).
        """
        obs = self._build_snapshot(frame_det, intent_preds)
        self._history.append(obs)
        return self.get_stacked_observation()

    def get_stacked_observation(self) -> np.ndarray:
        """Return concatenated observation history."""
        if len(self._history) == 0:
            return np.zeros(OBS_DIM * self.temporal_stack_size, dtype=np.float32)

        pad_with = self._history[0]
        padding_needed = self.temporal_stack_size - len(self._history)
        pads = [pad_with] * padding_needed
        frames = pads + list(self._history)
        return np.concatenate(frames, axis=0).astype(np.float32)

    def reset(self) -> None:
        """Clear history (call at episode start)."""
        self._history.clear()
        self._prev_action = None

    # ------------------------------------------------------------------
    # Internal: snapshot builder
    # ------------------------------------------------------------------
    def _build_snapshot(
        self,
        frame_det: FrameDetections,
        intent_preds: list[IntentPrediction],
    ) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        persons = frame_det.persons
        obstacles = frame_det.obstacles

        # -- Scene encoding [0:6] -----------------------------------
        obs[0] = min(len(persons), 10) / 10.0

        if persons:
            nearest_p = self._nearest(persons)
            obs[1] = nearest_p.distance / MAX_DIST
            obs[2] = self._angle_from_bbox(nearest_p, frame_det.frame_width)
        else:
            obs[1] = 1.0  # far
            obs[2] = 0.0

        if obstacles:
            nearest_o = self._nearest(obstacles)
            obs[3] = nearest_o.distance / MAX_DIST
            obs[4] = self._angle_from_bbox(nearest_o, frame_det.frame_width)
        else:
            obs[3] = 1.0
            obs[4] = 0.0

        obs[5] = 0.0

        # -- Occupancy grid [6:70] ----------------------------------
        grid = self._build_occupancy_grid(frame_det)
        obs[6:70] = grid.flatten()

        # -- Human intents [70:94] ----------------------------------
        intent_map = {p.track_id: p for p in intent_preds}
        top_persons = persons[:MAX_PERSONS]

        for i, person in enumerate(top_persons):
            base = 70 + i * INTENT_FEATS
            pred = intent_map.get(person.track_id)
            if pred is not None:
                obs[base : base + 6] = pred.probabilities
                obs[base + 6] = pred.dx
                obs[base + 7] = pred.dy

        # -- Robot state [94:97] ------------------------------------
        rs = self._robot_state
        obs[94] = np.clip(rs.vx, -2.0, 2.0) / 2.0
        obs[95] = np.clip(rs.vy, -2.0, 2.0) / 2.0
        obs[96] = np.clip(rs.vtheta, -3.14, 3.14) / math.pi

        # -- Previous action [97:104] -------------------------------
        if self._prev_action is not None:
            obs[97] = self._prev_action.velocity_scale
            obs[98] = self._prev_action.heading_offset / (math.pi / 4)
            mode_oh = np.zeros(NUM_MODES, dtype=np.float32)
            mode_oh[int(self._prev_action.mode)] = 1.0
            obs[99:104] = mode_oh

        # [104:114] reserved for backward-compatible observation width.
        obs[104:114] = 0.0

        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_occupancy_grid(self, frame_det: FrameDetections) -> np.ndarray:
        """8×8 binary grid; 1 = occupied, 0 = free.
        Maps detection centres into grid cells assuming full-frame coverage.
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        fw = frame_det.frame_width or 1
        fh = frame_det.frame_height or 1
        for det in frame_det.persons + frame_det.obstacles:
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            cx_norm = cx / fw
            cy_norm = cy / fh
            gx = min(self.grid_size - 1, int(cx_norm * self.grid_size))
            gy = min(self.grid_size - 1, int(cy_norm * self.grid_size))
            grid[gy, gx] = 1.0
        return grid

    @staticmethod
    def _nearest(dets: list[DetectionResult]) -> DetectionResult:
        """Return detection with the smallest actual distance (metres).

        Uses DetectionResult.distance which is populated by the hybrid
        estimator (depth camera primary, bbox heuristic fallback).
        """
        return min(dets, key=lambda d: d.distance)

    @staticmethod
    def _angle_from_bbox(det: DetectionResult, frame_width: int) -> float:
        """Horizontal angle estimate from bbox centre (radians).
        Camera FOV assumed ~70°.
        """
        cx = (det.bbox[0] + det.bbox[2]) / 2.0
        center = frame_width / 2.0
        return (cx - center) / center * math.radians(35)

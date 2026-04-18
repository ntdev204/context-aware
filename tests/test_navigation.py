"""Tests for navigation: context builder, heuristic policy, safety monitor."""

from __future__ import annotations

import math
import time

import numpy as np

from src.navigation.context_builder import OBS_DIM, ContextBuilder, RobotState
from src.navigation.heuristic_policy import HeuristicPolicy
from src.navigation.nav_command import NavigationCommand, NavigationMode
from src.navigation.safety_monitor import SafetyMonitor
from src.perception.intent_cnn import (
    APPROACHING,
    CROSSING,
    ERRATIC,
    INTENT_NAMES,
    STATIONARY,
    IntentPrediction,
)
from src.perception.yolo_detector import DetectionResult, FrameDetections

# ── Helpers ──────────────────────────────────────────────────────────────────


def make_det(x1=100, y1=100, x2=200, y2=400, cls="person", tid=1):
    from src.perception.yolo_detector import CLASS_NAMES

    return DetectionResult(
        bbox=(x1, y1, x2, y2),
        class_id=CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 0,
        class_name=cls,
        confidence=0.9,
        track_id=tid,
    )


def make_fd(persons=None, obstacles=None, free_space=0.9):
    fd = FrameDetections(free_space_ratio=free_space)
    fd.persons = persons or []
    fd.obstacles = obstacles or []
    fd.all_detections = fd.persons + fd.obstacles
    return fd


def make_pred(track_id=1, intent=STATIONARY, conf=0.9, dx=0.0, dy=0.0):
    probs = np.zeros(6, dtype=np.float32)
    probs[intent] = conf
    probs /= probs.sum()
    return IntentPrediction(
        track_id=track_id,
        intent_class=intent,
        intent_name=INTENT_NAMES[intent],
        probabilities=probs,
        dx=dx,
        dy=dy,
        confidence=conf,
    )


# ── ContextBuilder ───────────────────────────────────────────────────────────


class TestContextBuilder:
    def test_snapshot_observation_shape(self):
        cb = ContextBuilder(temporal_stack_size=1)
        fd = make_fd()
        obs = cb.build(fd, [])
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_temporal_k3_shape(self):
        cb = ContextBuilder(temporal_stack_size=3)
        fd = make_fd()
        for _ in range(3):
            obs = cb.build(fd, [])
        assert obs.shape == (OBS_DIM * 3,)

    def test_cold_start_padding(self):
        """First frame with k=3 should still return full (306,) vector."""
        cb = ContextBuilder(temporal_stack_size=3)
        fd = make_fd()
        obs = cb.build(fd, [])
        assert obs.shape == (OBS_DIM * 3,)  # padded with zeros

    def test_free_space_ratio_in_obs(self):
        cb = ContextBuilder(temporal_stack_size=1)
        fd = make_fd(free_space=0.75)
        obs = cb.build(fd, [])
        assert abs(obs[5] - 0.75) < 1e-4

    def test_no_persons_distance_is_normalised(self):
        cb = ContextBuilder(temporal_stack_size=1)
        fd = make_fd(persons=[], free_space=1.0)
        obs = cb.build(fd, [])
        assert obs[1] == 1.0  # normalised max distance

    def test_intent_feats_in_obs(self):
        cb = ContextBuilder(temporal_stack_size=1)
        person = make_det()
        fd = make_fd(persons=[person])
        pred = make_pred(track_id=1, intent=APPROACHING, conf=0.95)
        obs = cb.build(fd, [pred])
        # intent features start at index 70
        assert obs[70 + APPROACHING] > 0.5

    def test_robot_state_in_obs(self):
        cb = ContextBuilder(temporal_stack_size=1)
        cb.update_robot_state(RobotState(vx=1.0, vy=0.5, vtheta=0.3))
        fd = make_fd()
        obs = cb.build(fd, [])
        assert abs(obs[94] - 0.5) < 1e-3  # 1.0/2.0 normalised

    def test_reset_clears_history(self):
        cb = ContextBuilder(temporal_stack_size=3)
        fd = make_fd()
        cb.build(fd, [])
        cb.build(fd, [])
        assert len(cb._history) == 2
        cb.reset()
        assert len(cb._history) == 0

    def test_observation_all_finite(self):
        cb = ContextBuilder(temporal_stack_size=1)
        persons = [make_det(200, 100, 300, 400, tid=i) for i in range(5)]
        preds = [make_pred(track_id=i, intent=CROSSING) for i in range(5)]
        fd = make_fd(persons=persons, free_space=0.3)
        obs = cb.build(fd, preds)
        assert np.all(np.isfinite(obs))


# ── HeuristicPolicy ──────────────────────────────────────────────────────────


class TestHeuristicPolicy:
    def _policy(self):
        return HeuristicPolicy(hard_stop_distance=0.5, slow_down_distance=1.0)

    def _obs(self, person_dist=5.0, obstacle_dist=5.0, free=0.9):
        obs = np.zeros(102, dtype=np.float32)
        obs[1] = person_dist / 5.0
        obs[3] = obstacle_dist / 5.0
        obs[5] = free
        return obs

    def test_empty_scene_cruise(self):
        policy = self._policy()
        obs = self._obs()
        cmd = policy.decide(obs, make_fd(free_space=0.95), [])
        assert cmd.mode == NavigationMode.CRUISE
        assert cmd.velocity_scale > 0.7

    def test_person_close_stop(self):
        policy = self._policy()
        person = make_det(300, 50, 380, 700)  # tall bbox = close
        obs = self._obs(person_dist=0.3)
        cmd = policy.decide(obs, make_fd(persons=[person], free_space=0.4), [])
        assert cmd.mode == NavigationMode.STOP
        assert cmd.velocity_scale == 0.0

    def test_erratic_stop(self):
        policy = self._policy()
        obs = self._obs(person_dist=2.0, free=0.5)
        pred = make_pred(track_id=1, intent=ERRATIC, conf=0.9)
        cmd = policy.decide(obs, make_fd(persons=[make_det()]), [pred])
        assert cmd.mode == NavigationMode.STOP

    def test_crossing_avoid(self):
        policy = self._policy()
        obs = self._obs(person_dist=1.5, free=0.5)
        pred = make_pred(track_id=1, intent=CROSSING, conf=0.8)
        cmd = policy.decide(obs, make_fd(persons=[make_det()]), [pred])
        assert cmd.mode == NavigationMode.AVOID
        assert 0.0 < cmd.velocity_scale <= policy.cautious_vel  # AVOID is ≤ cautious

    def test_velocity_clipped_to_one(self):
        policy = HeuristicPolicy(cruise_velocity=2.0)  # intentionally > 1
        obs = self._obs()
        cmd = policy.decide(obs, make_fd(free_space=1.0), [])
        assert cmd.velocity_scale <= 1.0

    def test_heading_clipped(self):
        policy = self._policy()
        person = make_det(0, 100, 50, 400)  # far left → should steer right
        obs = self._obs(person_dist=1.5, free=0.4)
        pred = make_pred(track_id=1, intent=CROSSING, conf=0.9)
        cmd = policy.decide(obs, make_fd(persons=[person]), [pred])
        assert -math.pi / 4 <= cmd.heading_offset <= math.pi / 4

    def test_persons_present_cautious(self):
        policy = self._policy()
        obs = self._obs(person_dist=2.0, free=0.3)
        cmd = policy.decide(obs, make_fd(persons=[make_det()]), [])
        assert cmd.mode == NavigationMode.CAUTIOUS


# ── SafetyMonitor ────────────────────────────────────────────────────────────


class TestSafetyMonitor:
    def _monitor(self):
        return SafetyMonitor(
            hard_stop_person=0.5,
            hard_stop_obstacle=0.3,
            slow_down_distance=1.0,
            watchdog_timeout_ms=500,
        )

    def _cmd(self, mode=NavigationMode.CRUISE, vel=0.8):
        return NavigationCommand(mode=mode, velocity_scale=vel, heading_offset=0.0)

    def test_person_hard_stop(self):
        monitor = self._monitor()
        # Very tall bbox = very close person
        person = make_det(300, 10, 380, 748)
        fd = make_fd(persons=[person])
        cmd = monitor.check(self._cmd(), fd, [])
        assert cmd.mode == NavigationMode.STOP
        assert cmd.velocity_scale == 0.0
        assert cmd.safety_override is True

    def test_erratic_override(self):
        monitor = self._monitor()
        pred = make_pred(track_id=1, intent=ERRATIC, conf=0.8)
        cmd = monitor.check(self._cmd(), make_fd(persons=[make_det()]), [pred])
        assert cmd.mode == NavigationMode.STOP

    def test_watchdog_timeout(self):
        monitor = self._monitor()
        # Manually force last_received to long ago
        monitor._last_robot_state_ts = time.monotonic() - 1.5
        cmd = monitor.check(self._cmd(), make_fd(), [])
        assert cmd.mode == NavigationMode.STOP

    def test_battery_critical(self):
        monitor = self._monitor()
        rs = RobotState(battery_percent=5.0)
        monitor.update_robot_state(rs)
        cmd = monitor.check(self._cmd(), make_fd(), [])
        assert cmd.mode == NavigationMode.STOP

    def test_no_override_safe_scene(self):
        monitor = self._monitor()
        monitor._last_robot_state_ts = time.monotonic()
        cmd = monitor.check(self._cmd(vel=0.8), make_fd(free_space=1.0), [])
        # No persons → no override
        assert cmd.mode == NavigationMode.CRUISE
        assert cmd.velocity_scale <= 1.0

    def test_velocity_always_clipped(self):
        monitor = self._monitor()
        monitor._last_robot_state_ts = time.monotonic()
        cmd = NavigationCommand(mode=NavigationMode.CRUISE, velocity_scale=5.0)
        result = monitor.check(cmd, make_fd(), [])
        assert result.velocity_scale <= 1.0

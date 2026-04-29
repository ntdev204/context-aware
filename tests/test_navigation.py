from __future__ import annotations

import math

import numpy as np

from src.navigation.context_builder import OBS_DIM, ContextBuilder, RobotState
from src.navigation.heuristic_policy import HeuristicPolicy
from src.navigation.nav_command import NavigationMode
from src.perception.intent_cnn import (
    APPROACHING,
    CROSSING,
    ERRATIC,
    INTENT_NAMES,
    STATIONARY,
    IntentPrediction,
)
from src.perception.yolo_detector import DetectionResult, FrameDetections


def make_det(x1=100, y1=100, x2=200, y2=400, cls="person", tid=1, dist=3.0):
    from src.perception.yolo_detector import CLASS_NAMES

    return DetectionResult(
        bbox=(x1, y1, x2, y2),
        class_id=CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 0,
        class_name=cls,
        confidence=0.9,
        track_id=tid,
        distance=dist,
    )


def make_fd(
    persons=None, obstacles=None, free_space=0.9, free_sectors=None, heading=0.0, width=0.0
):
    fd = FrameDetections(free_space_ratio=free_space)
    fd.persons = persons or []
    fd.obstacles = obstacles or []
    fd.all_detections = fd.persons + fd.obstacles
    fd.free_sectors = None if free_sectors is None else np.asarray(free_sectors, dtype=np.float32)
    fd.navigable_heading = heading
    fd.navigable_width = width
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
        cb = ContextBuilder(temporal_stack_size=3)
        fd = make_fd()
        obs = cb.build(fd, [])
        assert obs.shape == (OBS_DIM * 3,)

    def test_reserved_obs_slot_is_zero(self):
        cb = ContextBuilder(temporal_stack_size=1)
        fd = make_fd(free_space=0.75)
        obs = cb.build(fd, [])
        assert obs[5] == 0.0

    def test_reserved_camera_geometry_slots_are_zero(self):
        cb = ContextBuilder(temporal_stack_size=1)
        sectors = np.linspace(0.0, 1.0, 8, dtype=np.float32)
        fd = make_fd(free_sectors=sectors, heading=math.pi / 8, width=math.pi / 4)
        obs = cb.build(fd, [])
        assert np.allclose(obs[104:114], 0.0)

    def test_no_persons_distance_is_normalised(self):
        cb = ContextBuilder(temporal_stack_size=1)
        fd = make_fd(persons=[], free_space=1.0)
        obs = cb.build(fd, [])
        assert obs[1] == 1.0

    def test_intent_feats_in_obs(self):
        cb = ContextBuilder(temporal_stack_size=1)
        person = make_det()
        fd = make_fd(persons=[person])
        pred = make_pred(track_id=1, intent=APPROACHING, conf=0.95)
        obs = cb.build(fd, [pred])
        assert obs[70 + APPROACHING] > 0.5

    def test_robot_state_in_obs(self):
        cb = ContextBuilder(temporal_stack_size=1)
        cb.update_robot_state(RobotState(vx=1.0, vy=0.5, vtheta=0.3))
        fd = make_fd()
        obs = cb.build(fd, [])
        assert abs(obs[94] - 0.5) < 1e-3

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


class TestHeuristicPolicy:
    def _policy(self):
        return HeuristicPolicy(hard_stop_distance=0.5, slow_down_distance=1.0)

    def _obs(self, person_dist=5.0, obstacle_dist=5.0, free=0.9):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[1] = person_dist / 5.0
        obs[3] = obstacle_dist / 5.0
        obs[5] = free
        return obs

    def test_empty_scene_stops_without_authenticated_target(self):
        policy = self._policy()
        obs = self._obs()
        cmd = policy.decide(obs, make_fd(free_space=0.95), [])
        assert cmd.mode == NavigationMode.STOP
        assert cmd.velocity_scale == 0.0

    def test_person_close_stops_without_authenticated_target(self):
        policy = self._policy()
        person = make_det(300, 50, 380, 700)
        obs = self._obs(person_dist=0.3)
        cmd = policy.decide(obs, make_fd(persons=[person], free_space=0.4), [])
        assert cmd.mode == NavigationMode.STOP
        assert cmd.velocity_scale == 0.0

    def test_erratic_does_not_enable_autonomous_motion(self):
        policy = self._policy()
        obs = self._obs(person_dist=2.0, free=0.5)
        pred = make_pred(track_id=1, intent=ERRATIC, conf=0.9)
        cmd = policy.decide(obs, make_fd(persons=[make_det()]), [pred])
        assert cmd.mode == NavigationMode.STOP

    def test_crossing_does_not_enable_avoid_without_authentication(self):
        policy = self._policy()
        obs = self._obs(person_dist=1.5, free=0.5)
        pred = make_pred(track_id=1, intent=CROSSING, conf=0.8)
        cmd = policy.decide(obs, make_fd(persons=[make_det()]), [pred])
        assert cmd.mode == NavigationMode.STOP
        assert cmd.velocity_scale == 0.0

    def test_authenticated_follow_target_moves(self):
        policy = self._policy()
        policy.set_follow_target(1)
        obs = self._obs()
        cmd = policy.decide(obs, make_fd(persons=[make_det(dist=3.0)]), [])
        assert cmd.mode == NavigationMode.FOLLOW
        assert cmd.follow_target_id == 1
        assert cmd.velocity_scale > 0.0

    def test_follow_velocity_clipped_to_one(self):
        policy = HeuristicPolicy(follow_max_vel=2.0)
        policy.set_follow_target(1)
        obs = self._obs()
        cmd = policy.decide(obs, make_fd(persons=[make_det(dist=8.0)], free_space=1.0), [])
        assert cmd.velocity_scale <= 1.0

    def test_follow_keeps_zero_heading_for_mecanum_strafe(self):
        policy = self._policy()
        policy.set_follow_target(1)
        person = make_det(0, 100, 50, 400)
        obs = self._obs(person_dist=1.5, free=0.4)
        cmd = policy.decide(obs, make_fd(persons=[person]), [])
        assert cmd.mode == NavigationMode.FOLLOW
        assert cmd.heading_offset == 0.0

    def test_follow_too_close_disables_lateral_strafe(self):
        policy = self._policy()
        policy.set_follow_target(1)
        person = make_det(0, 100, 50, 400, dist=1.9)
        obs = self._obs(person_dist=1.9, free=0.4)
        cmd = policy.decide(obs, make_fd(persons=[person]), [])
        assert cmd.mode == NavigationMode.FOLLOW
        assert cmd.velocity_y == 0.0

    def test_lost_follow_target_keeps_lock_id(self):
        policy = self._policy()
        policy.set_follow_target(7)
        obs = self._obs()
        cmd = policy.decide(obs, make_fd(persons=[]), [])
        assert cmd.mode == NavigationMode.STOP
        assert cmd.follow_target_id == 7
        assert policy.follow_target_id == 7

    def test_stale_follow_target_stops_but_keeps_lock_id(self):
        policy = self._policy()
        policy.set_follow_target(7)
        person = make_det(tid=7, dist=3.0)
        person.stale = True
        obs = self._obs()
        cmd = policy.decide(obs, make_fd(persons=[person]), [])
        assert cmd.mode == NavigationMode.STOP
        assert cmd.follow_target_id == 7

    def test_persons_present_stops_without_authenticated_target(self):
        policy = self._policy()
        obs = self._obs(person_dist=2.0, free=0.3)
        cmd = policy.decide(obs, make_fd(persons=[make_det()]), [])
        assert cmd.mode == NavigationMode.STOP

    def test_freespace_heading_does_not_enable_cruise_without_authentication(self):
        policy = self._policy()
        obs = self._obs(free=0.95)
        fd = make_fd(
            free_space=0.95,
            free_sectors=[0.9] * 8,
            heading=math.radians(12),
            width=math.radians(40),
        )
        cmd = policy.decide(obs, fd, [])
        assert cmd.mode == NavigationMode.STOP
        assert cmd.heading_offset == 0.0

    def test_front_sectors_do_not_enable_autonomous_motion(self):
        policy = self._policy()
        obs = self._obs(free=0.95)
        fd = make_fd(free_space=0.95, free_sectors=[0.9, 0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.9])
        cmd = policy.decide(obs, fd, [])
        assert cmd.mode == NavigationMode.STOP

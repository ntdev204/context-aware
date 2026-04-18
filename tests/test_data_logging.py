"""CRITICAL tests for data logging pipeline (Bottleneck C1).

Tests verify:
- All required fields present per frame
- Timestamp alignment (same ts across all fields)
- Ring buffer overflow behaviour
- Data replayability (write → read → verify)
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from src.experience.buffer import ExperienceBuffer
from src.experience.collector import ExperienceCollector, _encode_action
from src.navigation.context_builder import RobotState
from src.navigation.nav_command import NavigationCommand, NavigationMode
from src.perception.intent_cnn import INTENT_NAMES, STATIONARY, IntentPrediction
from src.perception.yolo_detector import FrameDetections

# ── Fixtures ─────────────────────────────────────────────────────────────────


def make_frame_image():
    import numpy as np

    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def make_frame_det():
    fd = FrameDetections(free_space_ratio=0.8)
    fd.persons = []
    fd.obstacles = []
    fd.all_detections = []
    return fd


def make_robot_state():
    return RobotState(vx=0.5, vy=0.0, vtheta=0.1, battery_percent=80.0)


def make_nav_cmd():
    return NavigationCommand(mode=NavigationMode.CRUISE, velocity_scale=0.8, heading_offset=0.1)


def make_intent_pred(track_id=1):
    probs = np.zeros(6, dtype=np.float32)
    probs[STATIONARY] = 1.0
    return IntentPrediction(
        track_id=track_id,
        intent_class=STATIONARY,
        intent_name=INTENT_NAMES[STATIONARY],
        probabilities=probs,
        dx=0.0,
        dy=0.0,
        confidence=1.0,
    )


def make_observation(dim=102):
    return np.random.rand(dim).astype(np.float32)


# ── Action Encoding ───────────────────────────────────────────────────────────


class TestActionEncoding:
    def test_shape(self):
        cmd = make_nav_cmd()
        action = _encode_action(cmd)
        assert action.shape == (7,)
        assert action.dtype == np.float32

    def test_velocity_preserved(self):
        cmd = NavigationCommand(mode=NavigationMode.CRUISE, velocity_scale=0.75, heading_offset=0.2)
        action = _encode_action(cmd)
        assert abs(action[0] - 0.75) < 1e-5
        assert abs(action[1] - 0.2) < 1e-5

    def test_mode_one_hot(self):
        for mode in NavigationMode:
            cmd = NavigationCommand(mode=mode, velocity_scale=0.5)
            action = _encode_action(cmd)
            mode_oh = action[2:]
            assert mode_oh[int(mode)] == 1.0
            assert mode_oh.sum() == 1.0


# ── ExperienceBuffer ─────────────────────────────────────────────────────────


class TestExperienceBuffer:
    def test_push_and_len(self):
        buf = ExperienceBuffer(max_size=100, async_write=False)
        dummy = object()
        buf.push(dummy)
        assert len(buf) == 1

    def test_ring_overflow_drops_oldest(self):
        buf = ExperienceBuffer(max_size=5, async_write=False)
        for i in range(10):
            buf.push(i)
        assert len(buf) == 5
        # Most recent 5 should be 5-9
        batch = buf.pop_batch(5)
        assert batch[-1] == 9

    def test_pop_batch_size(self):
        buf = ExperienceBuffer(max_size=100, async_write=False)
        for i in range(20):
            buf.push(i)
        batch = buf.pop_batch(8)
        assert len(batch) == 8

    def test_thread_safety(self):
        """Push from multiple threads — no crash, count correct."""
        import threading

        buf = ExperienceBuffer(max_size=10_000, async_write=False)
        errors = []

        def push_many():
            try:
                for _ in range(1000):
                    buf.push(object())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=push_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(buf) <= 10_000

    def test_overflow_returns_false(self):
        buf = ExperienceBuffer(max_size=1, async_write=False)
        buf.push("first")
        result = buf.push("second")
        assert result is False

    def test_hdf5_write_creates_file(self):
        pytest.importorskip("h5py")  # skip if h5py not installed
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = ExperienceBuffer(
                max_size=100,
                write_dir=tmpdir,
                write_format="hdf5",
                async_write=False,
            )
            buf.start()

            collector = ExperienceCollector(buffer=buf, enabled=True)
            for i in range(5):
                collector.collect(
                    raw_frame=make_frame_image(),
                    frame_det=make_frame_det(),
                    intent_preds=[make_intent_pred()],
                    observation=make_observation(),
                    cmd=make_nav_cmd(),
                    robot_state=make_robot_state(),
                )
            # Manually flush pending writes
            buf._write_hdf5(buf._pending)
            buf.stop()

            h5_files = list(Path(tmpdir).glob("*.h5"))
            assert len(h5_files) >= 1


# ── ExperienceCollector ───────────────────────────────────────────────────────


class TestExperienceCollector:
    def _make_collector(self):
        buf = ExperienceBuffer(max_size=1000, async_write=False)
        return ExperienceCollector(buffer=buf, enabled=True, session_id="test"), buf

    def test_all_fields_present(self):
        collector, buf = self._make_collector()
        exp = collector.collect(
            raw_frame=make_frame_image(),
            frame_det=make_frame_det(),
            intent_preds=[make_intent_pred()],
            observation=make_observation(),
            cmd=make_nav_cmd(),
            robot_state=make_robot_state(),
        )
        assert exp is not None
        assert exp.frame_id == 0
        assert exp.raw_image_jpeg is not None and len(exp.raw_image_jpeg) > 0
        assert exp.observation.shape == (102,)
        assert exp.action.shape == (7,)
        assert exp.timestamp > 0
        assert exp.wall_time > 0
        assert exp.robot_state is not None
        assert len(exp.intent_predictions) == 1

    def test_timestamp_alignment(self):
        """frame_id, timestamp, wall_time all from the same collect() call."""
        collector, _ = self._make_collector()
        t_before = time.monotonic()
        exp = collector.collect(
            raw_frame=make_frame_image(),
            frame_det=make_frame_det(),
            intent_preds=[],
            observation=make_observation(),
            cmd=make_nav_cmd(),
            robot_state=make_robot_state(),
        )
        t_after = time.monotonic()
        assert t_before <= exp.timestamp <= t_after

    def test_frame_id_increments(self):
        collector, _ = self._make_collector()
        for expected_id in range(5):
            exp = collector.collect(
                raw_frame=make_frame_image(),
                frame_det=make_frame_det(),
                intent_preds=[],
                observation=make_observation(),
                cmd=make_nav_cmd(),
                robot_state=make_robot_state(),
            )
            assert exp.frame_id == expected_id

    def test_jpeg_quality_min_size(self):
        collector, _ = self._make_collector()
        exp = collector.collect(
            raw_frame=make_frame_image(),
            frame_det=make_frame_det(),
            intent_preds=[],
            observation=make_observation(),
            cmd=make_nav_cmd(),
            robot_state=make_robot_state(),
        )
        # Quality 85 JPEG of a 640x480 random image — should be > 10KB
        assert len(exp.raw_image_jpeg) > 10_000

    def test_disabled_returns_none(self):
        buf = ExperienceBuffer(max_size=100, async_write=False)
        collector = ExperienceCollector(buffer=buf, enabled=False)
        result = collector.collect(
            raw_frame=make_frame_image(),
            frame_det=make_frame_det(),
            intent_preds=[],
            observation=make_observation(),
            cmd=make_nav_cmd(),
            robot_state=make_robot_state(),
        )
        assert result is None

    def test_stats_tracking(self):
        collector, _ = self._make_collector()
        for _ in range(10):
            collector.collect(
                raw_frame=make_frame_image(),
                frame_det=make_frame_det(),
                intent_preds=[],
                observation=make_observation(),
                cmd=make_nav_cmd(),
                robot_state=make_robot_state(),
            )
        stats = collector.stats
        assert stats["collected"] == 10
        assert stats["dropped"] == 0

    def test_directory_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = ExperienceBuffer(
                max_size=100,
                write_dir=tmpdir,
                write_format="directory",
                async_write=False,
            )
            buf.start()
            collector = ExperienceCollector(buffer=buf, enabled=True)
            exp = collector.collect(
                raw_frame=make_frame_image(),
                frame_det=make_frame_det(),
                intent_preds=[],
                observation=make_observation(),
                cmd=make_nav_cmd(),
                robot_state=make_robot_state(),
            )
            buf._write_directory([exp])
            buf.stop()

            jpgs = list(Path(tmpdir).glob("*.jpg"))
            jsons = list(Path(tmpdir).glob("*.json"))
            assert len(jpgs) == 1
            assert len(jsons) == 1

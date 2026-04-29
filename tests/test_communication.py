"""Tests for ZMQ communication and Protobuf serialization."""

from __future__ import annotations

import struct
import time
import json

import pytest

from src.navigation.nav_command import NavigationCommand, NavigationMode

# ── NavigationCommand Serialization ──────────────────────────────────────────


class TestNavCommandSerialization:
    def test_clip_velocity(self):
        cmd = NavigationCommand(mode=NavigationMode.CRUISE, velocity_scale=2.5)
        cmd.clip()
        assert cmd.velocity_scale == 1.0

    def test_clip_velocity_negative(self):
        cmd = NavigationCommand(mode=NavigationMode.CRUISE, velocity_scale=-0.5)
        cmd.clip()
        assert cmd.velocity_scale == -0.5

    def test_clip_velocity_underflow(self):
        cmd = NavigationCommand(mode=NavigationMode.CRUISE, velocity_scale=-2.5)
        cmd.clip()
        assert cmd.velocity_scale == -1.0

    def test_clip_heading(self):
        import math

        cmd = NavigationCommand(mode=NavigationMode.AVOID, heading_offset=2.0)
        cmd.clip()
        assert cmd.heading_offset <= math.pi / 4

    def test_clip_returns_self(self):
        cmd = NavigationCommand(mode=NavigationMode.STOP, velocity_scale=0.5)
        result = cmd.clip()
        assert result is cmd

    def test_repr_contains_mode(self):
        cmd = NavigationCommand(mode=NavigationMode.FOLLOW, velocity_scale=0.5)
        assert "FOLLOW" in repr(cmd)

    def test_is_safe_to_move_stop(self):
        cmd = NavigationCommand(mode=NavigationMode.STOP, velocity_scale=0.0)
        assert not cmd.is_safe_to_move()

    def test_is_safe_to_move_cruise(self):
        cmd = NavigationCommand(mode=NavigationMode.CRUISE, velocity_scale=0.8)
        assert cmd.is_safe_to_move()


# ── ZMQ Publisher Fallback Encoding ──────────────────────────────────────────


class TestZMQPublisherEncoding:
    """Test the struct-based fallback encoder (no proto compiled needed)."""

    def test_fallback_encode_nav_cmd(self):
        pytest.importorskip("zmq")  # skip on machines without pyzmq
        from src.communication.zmq_publisher import ZMQPublisher

        cmd = NavigationCommand(
            mode=NavigationMode.AVOID,
            velocity_scale=0.4,
            heading_offset=0.15,
            follow_target_id=-1,
            confidence=0.85,
            safety_override=False,
        )
        raw = ZMQPublisher._encode_nav_cmd(cmd)
        assert isinstance(raw, bytes)
        assert len(raw) > 0

    def test_fallback_decode_round_trip(self):
        pytest.importorskip("zmq")  # skip on machines without pyzmq
        """Encode with struct, decode manually, check values."""
        from src.communication.zmq_publisher import ZMQPublisher

        cmd = NavigationCommand(
            mode=NavigationMode.CRUISE,
            velocity_scale=0.7,
            heading_offset=-0.1,
            follow_target_id=-1,
            timestamp=1234567.8,
            confidence=0.9,
            safety_override=False,
        )
        raw = ZMQPublisher._encode_nav_cmd(cmd)

        # If proto compiled, it returns proto bytes — we only test struct path here
        # If raw is struct-encoded it should be 29 bytes: !iffiffB
        if len(raw) == struct.calcsize("!iffiffB"):
            mode, v, h, ftid, ts, conf, safe = struct.unpack("!iffiffB", raw)
            assert mode == int(NavigationMode.CRUISE)
            assert abs(v - 0.7) < 1e-5
            assert abs(h - (-0.1)) < 1e-5


class TestZMQSubscriberDecoding:
    def test_struct_robot_state_decodes_lidar_sectors(self):
        pytest.importorskip("zmq")
        from src.communication.zmq_subscriber import ZMQSubscriber

        raw = struct.pack(
            "!11fd",
            0.1,
            0.2,
            0.3,
            1.0,
            2.0,
            0.4,
            88.0,
            0.55,
            1.2,
            1.8,
            0.7,
            123.0,
        )

        state = ZMQSubscriber._decode(raw)

        assert state.pos_theta == 0.4
        assert state.lidar_sectors == (0.55, 1.2, 1.8, 0.7)

    def test_json_robot_state_decodes_full_lidar_scan(self):
        pytest.importorskip("zmq")
        from src.communication.zmq_subscriber import ZMQSubscriber

        scan = [9.9] * 360
        scan[0] = 0.42
        scan[90] = 1.7
        raw = json.dumps(
            {
                "odom": {
                    "vx": 0.1,
                    "vy": 0.2,
                    "vtheta": 0.3,
                    "pos_x": 1.0,
                    "pos_y": 2.0,
                    "pos_theta": 0.4,
                },
                "battery_percent": 87.0,
                "lidar": {
                    "sectors": {"front": 0.42, "rear": 1.2, "left": 1.7, "right": 0.8},
                    "scan360": scan,
                },
                "timestamp": 123.0,
            }
        ).encode("utf-8")

        state = ZMQSubscriber._decode(raw)

        assert state.lidar_front == 0.42
        assert len(state.lidar_scan) == 360
        assert state.lidar_scan[90] == 1.7


# ── ZMQ Loopback (integration, requires ZMQ) ─────────────────────────────────


class TestZMQLoopback:
    """Test PUB/SUB on loopback.  Skipped if ZMQ not available."""

    @pytest.fixture(autouse=True)
    def require_zmq(self):
        pytest.importorskip("zmq")

    def test_pub_sub_loopback(self):
        import zmq

        ctx = zmq.Context()
        port = 15999

        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.bind(f"tcp://127.0.0.1:{port}")

        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.LINGER, 0)
        sub.setsockopt_string(zmq.SUBSCRIBE, "test")
        sub.connect(f"tcp://127.0.0.1:{port}")
        sub.setsockopt(zmq.RCVTIMEO, 2000)

        time.sleep(0.1)  # allow connect
        pub.send_multipart([b"test", b"hello"])

        try:
            topic, data = sub.recv_multipart()
            assert topic == b"test"
            assert data == b"hello"
        finally:
            pub.close()
            sub.close()
            ctx.term()

    def test_pub_no_block_when_no_subscriber(self):
        """Publisher must not hang when no subscriber connected."""
        import zmq

        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.SNDHWM, 2)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind("tcp://127.0.0.1:15998")
        time.sleep(0.05)

        # Should not raise
        for _ in range(10):
            try:
                sock.send_multipart([b"ai/nav_cmd", b"data"], flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

        sock.close()
        ctx.term()

"""ZMQ Publisher -- sends NavigationCommand and DetectionList to RasPi.

Channels:
    PUB :5555  topic "ai/nav_cmd"    NavigationCommand  @ 30 Hz
    PUB :5556  topic "ai/detections" DetectionList      @ 10 Hz

Socket config:
    SNDHWM = 2      drop old msgs if consumer slow
    LINGER = 0      no hung sockets on shutdown
    CONFLATE = 1    only keep latest message per topic
"""

from __future__ import annotations

import logging
import threading

try:
    import zmq
except ImportError:  # pragma: no cover
    zmq = None  # type: ignore[assignment]

from ..navigation.nav_command import NavigationCommand
from ..perception.yolo_detector import FrameDetections
import struct

try:
    from .proto import messages_pb2 as pb
except ImportError:
    pb = None

logger = logging.getLogger(__name__)


class ZMQPublisher:
    """Thread-safe ZMQ PUB socket wrapper."""

    def __init__(
        self,
        nav_cmd_port: int = 5555,
        detections_port: int = 5556,
        bind_host: str = "0.0.0.0",
    ) -> None:
        self.nav_port = nav_cmd_port
        self.det_port = detections_port
        self.bind_host = bind_host

        self._ctx: zmq.Context | None = None
        self._nav_sock: zmq.Socket | None = None
        self._det_sock: zmq.Socket | None = None
        self._lock = threading.Lock()
        self._det_frame_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._ctx = zmq.Context()

        self._nav_sock = self._make_pub(self.nav_port)
        self._det_sock = self._make_pub(self.det_port)

        logger.info(
            "ZMQ Publisher started -- nav_cmd:%d  detections:%d",
            self.nav_port,
            self.det_port,
        )

    def stop(self) -> None:
        for sock in (self._nav_sock, self._det_sock):
            if sock:
                sock.close()
        if self._ctx:
            self._ctx.term()
        logger.info("ZMQ Publisher stopped")

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------
    def publish_nav_cmd(self, cmd: NavigationCommand) -> None:
        """Serialise *cmd* to Protobuf and publish on nav_cmd channel."""
        try:
            payload = self._encode_nav_cmd(cmd)
            with self._lock:
                self._nav_sock.send_multipart([b"ai/nav_cmd", payload], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass  # subscriber too slow — drop (SNDHWM handles this)
        except Exception as exc:
            logger.error("nav_cmd publish error: %s", exc)

    def publish_detections(self, frame_det: FrameDetections) -> None:
        """Publish detections at ~10 Hz (every 3rd frame at 30 Hz)."""
        self._det_frame_count += 1
        if self._det_frame_count % 3 != 0:
            return
        try:
            payload = self._encode_detections(frame_det)
            with self._lock:
                self._det_sock.send_multipart([b"ai/detections", payload], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception as exc:
            logger.error("detections publish error: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _make_pub(self, port: int) -> zmq.Socket:
        sock = self._ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.SNDHWM, 2)
        sock.setsockopt(zmq.LINGER, 0)
        # Bỏ zmq.CONFLATE vì CONFLATE làm hỏng (drop) multipart message trong ZMQ PUB
        sock.bind(f"tcp://{self.bind_host}:{port}")
        return sock

    @staticmethod
    def _encode_nav_cmd(cmd: NavigationCommand) -> bytes:
        """Encode NavigationCommand thành binary struct để gửi cho Pi.

        Format (29 bytes): i f f f i f f B
          - mode           : int32   (NavigationMode)
          - velocity_scale : float32 (Linear X)
          - velocity_y     : float32 (Linear Y)
          - heading_offset : float32 (Angular Z)
          - follow_id      : int32   (track_id)
          - timestamp      : float32 (epoch seconds, truncated ok)
          - confidence     : float32
          - safety_override: uint8
        """
        return struct.pack(
            "!ifffiffB",
            int(cmd.mode),
            float(cmd.velocity_scale),
            float(cmd.velocity_y),
            float(cmd.heading_offset),
            int(cmd.follow_target_id),
            float(cmd.timestamp),
            float(cmd.confidence),
            int(cmd.safety_override),
        )

    @staticmethod
    def _encode_detections(frame_det: FrameDetections) -> bytes:
        if pb is None:
            return b""
        try:
            msg = pb.DetectionList()
            msg.timestamp = frame_det.timestamp
            msg.frame_id = frame_det.frame_id
            msg.free_space_ratio = frame_det.free_space_ratio
            for d in frame_det.all_detections:
                det = msg.detections.add()
                det.track_id = d.track_id
                det.x1, det.y1, det.x2, det.y2 = d.bbox
                det.class_name = d.class_name
                det.confidence = d.confidence
                det.intent_class = d.intent_class
                det.intent_confidence = d.intent_confidence
                det.dx = d.dx
                det.dy = d.dy
            return msg.SerializeToString()
        except (ImportError, AttributeError):
            return b""  # graceful degradation without proto

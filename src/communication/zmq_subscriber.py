"""ZMQ Subscriber — receives RobotState from RasPi.

Channel:
    SUB :5560  topic "robot/state"  RobotState @ 20 Hz

Runs in a background daemon thread.
If no message received within watchdog_timeout_ms, the safety monitor
is notified automatically.
"""

from __future__ import annotations

import json
import logging
import math
import struct
import threading
import time
from collections.abc import Callable

import zmq

from ..navigation.context_builder import RobotState

try:
    from .proto import messages_pb2 as pb
except ImportError:
    pb = None

logger = logging.getLogger(__name__)


class ZMQSubscriber:
    """Background thread subscriber for RobotState from RasPi."""

    def __init__(
        self,
        robot_state_port: int = 5560,
        rasp_pi_ip: str = "192.168.1.101",
        watchdog_timeout_ms: float = 500.0,
    ) -> None:
        self.port = robot_state_port
        self.rasp_pi_ip = rasp_pi_ip
        self.watchdog_timeout_ms = watchdog_timeout_ms

        self._ctx: zmq.Context | None = None
        self._sock: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        # Shared state (updated by background thread, read by main thread)
        self._latest_state = RobotState()
        self._state_lock = threading.Lock()
        self._last_received_ts = time.monotonic()

        # Optional callbacks
        self._on_state: Callable[[RobotState], None] | None = None
        self._on_timeout: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, on_state: Callable | None = None, on_timeout: Callable | None = None) -> None:
        """Start background receive loop.

        Parameters
        ----------
        on_state:   called with new RobotState whenever a message arrives
        on_timeout: called when watchdog fires (no message within timeout)
        """
        self._on_state = on_state
        self._on_timeout = on_timeout

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 2)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "robot/state")
        self._sock.connect(f"tcp://{self.rasp_pi_ip}:{self.port}")

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True, name="zmq-sub")
        self._thread.start()
        logger.info("ZMQ Subscriber connected to %s:%d", self.rasp_pi_ip, self.port)

    def stop(self) -> None:
        """Stop the subscriber gracefully."""
        if not self._running:
            return

        # Signal thread to stop
        self._running = False

        # Wait for thread to exit poll loop
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        # Now safe to close socket and context
        if self._sock:
            try:
                self._sock.close()
            except Exception as e:
                logger.debug("Error closing socket: %s", e)

        if self._ctx:
            try:
                self._ctx.term()
            except Exception as e:
                logger.debug("Error terminating context: %s", e)

        logger.info("ZMQ Subscriber stopped")

    # ------------------------------------------------------------------
    # Public read
    # ------------------------------------------------------------------
    def get_latest_state(self) -> RobotState:
        with self._state_lock:
            return self._latest_state

    def is_alive(self) -> bool:
        elapsed_ms = (time.monotonic() - self._last_received_ts) * 1000
        return elapsed_ms < self.watchdog_timeout_ms

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _recv_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)

        while self._running:
            socks = dict(poller.poll(timeout=100))  # 100ms poll interval
            if self._sock in socks:
                try:
                    parts = self._sock.recv_multipart(flags=zmq.NOBLOCK)
                    if len(parts) >= 2:
                        state = self._decode(parts[1])
                        with self._state_lock:
                            self._latest_state = state
                        self._last_received_ts = time.monotonic()
                        if self._on_state:
                            self._on_state(state)
                except zmq.Again:
                    pass
                except Exception as exc:
                    logger.error("Subscriber recv error: %s", exc)
            else:
                # Poll timeout — check watchdog
                elapsed_ms = (time.monotonic() - self._last_received_ts) * 1000
                if elapsed_ms > self.watchdog_timeout_ms:
                    if self._on_timeout:
                        self._on_timeout()

    @staticmethod
    def _decode(data: bytes) -> RobotState:
        if data[:1] == b"{":
            return ZMQSubscriber._decode_json(data)

        if pb is not None:
            try:
                msg = pb.RobotState()
                msg.ParseFromString(data)
                return RobotState(
                    vx=msg.vx,
                    vy=msg.vy,
                    vtheta=msg.vtheta,
                    pos_x=msg.pos_x,
                    pos_y=msg.pos_y,
                    pos_theta=getattr(msg, "pos_theta", 0.0),
                    battery_percent=msg.battery_percent,
                    nav2_status=msg.nav2_status,
                    lidar_front=getattr(msg, "lidar_front", 9.9),
                    lidar_rear=getattr(msg, "lidar_rear", 9.9),
                    lidar_left=getattr(msg, "lidar_left", 9.9),
                    lidar_right=getattr(msg, "lidar_right", 9.9),
                    lidar_scan=tuple(getattr(msg, "lidar_scan", ()) or ()),
                    timestamp=msg.timestamp,
                )
            except Exception:
                pass

        # Fallback to struct unpack if data is 52 bytes: 11 floats (44 bytes) + 1 double (8 bytes)
        if len(data) == 52:
            vx, vy, vtheta, px, py, ptheta, batt, _df, _dr, _dl, _dri, ts = struct.unpack(
                "!11fd", data
            )
            return RobotState(
                vx=vx,
                vy=vy,
                vtheta=vtheta,
                pos_x=px,
                pos_y=py,
                pos_theta=ptheta,
                battery_percent=batt,
                lidar_front=_df,
                lidar_rear=_dr,
                lidar_left=_dl,
                lidar_right=_dri,
                nav2_status="idle",
                timestamp=ts,
            )

        # Legacy 36 bytes (7 floats + 1 double)
        if len(data) == 36:
            vx, vy, vtheta, px, py, ptheta, batt, ts = struct.unpack("!7fd", data)
            return RobotState(
                vx=vx,
                vy=vy,
                vtheta=vtheta,
                pos_x=px,
                pos_y=py,
                pos_theta=ptheta,
                battery_percent=batt,
                nav2_status="idle",
                timestamp=ts,
            )
        raise ValueError(f"Unknown state payload length: {len(data)}")

    @staticmethod
    def _decode_json(data: bytes) -> RobotState:
        payload = json.loads(data.decode("utf-8"))
        odom = payload.get("odom", payload)
        lidar = payload.get("lidar", {})
        sectors = lidar.get("sectors", {})
        scan = lidar.get("scan360") or lidar.get("scan") or payload.get("lidar_scan") or []
        scan_tuple = tuple(ZMQSubscriber._valid_distance(v) for v in scan)

        return RobotState(
            vx=float(odom.get("vx", odom.get("v_x", 0.0))),
            vy=float(odom.get("vy", odom.get("v_y", 0.0))),
            vtheta=float(odom.get("vtheta", odom.get("v_z", 0.0))),
            pos_x=float(odom.get("pos_x", odom.get("x", 0.0))),
            pos_y=float(odom.get("pos_y", odom.get("y", 0.0))),
            pos_theta=float(odom.get("pos_theta", odom.get("yaw", 0.0))),
            battery_percent=float(payload.get("battery_percent", payload.get("battery", 100.0))),
            nav2_status=str(payload.get("nav2_status", "idle")),
            lidar_front=ZMQSubscriber._valid_distance(
                sectors.get("front", payload.get("lidar_front", 9.9))
            ),
            lidar_rear=ZMQSubscriber._valid_distance(
                sectors.get("rear", payload.get("lidar_rear", 9.9))
            ),
            lidar_left=ZMQSubscriber._valid_distance(
                sectors.get("left", payload.get("lidar_left", 9.9))
            ),
            lidar_right=ZMQSubscriber._valid_distance(
                sectors.get("right", payload.get("lidar_right", 9.9))
            ),
            lidar_scan=scan_tuple,
            timestamp=float(payload.get("timestamp", time.time())),
        )

    @staticmethod
    def _valid_distance(value) -> float:
        try:
            dist = float(value)
        except (TypeError, ValueError):
            return 9.9
        if not math.isfinite(dist) or dist <= 0.0:
            return 9.9
        return min(dist, 9.9)

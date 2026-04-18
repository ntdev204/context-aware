"""Thread-safe shared state hub between the inference loop and the Edge API."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional


@dataclass
class InferenceMetrics:
    fps: float = 0.0
    inference_ms: float = 0.0
    persons: int = 0
    obstacles: int = 0
    buffer_size: int = 0
    depth_coverage_pct: float = 0.0
    mode: str = "IDLE"
    frame_id: int = 0
    updated_at: float = field(default_factory=time.monotonic)


VALID_MODE_OVERRIDES = frozenset({"STOP", "CRUISE", "CAUTIOUS", "AVOID", "FOLLOW"})


class ServerState:
    """
    Central hub shared between the inference loop (writer) and Edge API (reader/writer).

    All public methods are thread-safe. Do NOT access private attributes directly.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._metrics = InferenceMetrics()
        self._mode_override: Optional[str] = None
        self._runtime_config: Dict[str, Any] = {}
        self._mjpeg_frame: bytes = b""
        self._running = True
        self._start_time = time.monotonic()
        self._latest_detections: Dict[str, Any] = {}

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def update_metrics(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._metrics, key):
                    setattr(self._metrics, key, value)
            self._metrics.updated_at = time.monotonic()

    def get_metrics(self) -> InferenceMetrics:
        with self._lock:
            m = self._metrics
            return InferenceMetrics(
                fps=m.fps,
                inference_ms=m.inference_ms,
                persons=m.persons,
                obstacles=m.obstacles,
                buffer_size=m.buffer_size,
                depth_coverage_pct=m.depth_coverage_pct,
                mode=m.mode,
                frame_id=m.frame_id,
                updated_at=m.updated_at,
            )

    def update_detections(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._latest_detections = payload

    def get_detections(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_detections)

    def set_mode_override(self, mode: Optional[str]) -> None:
        if mode is not None and mode not in VALID_MODE_OVERRIDES:
            raise ValueError(f"Invalid mode '{mode}'. Valid: {VALID_MODE_OVERRIDES}")
        with self._lock:
            self._mode_override = mode

    def get_mode_override(self) -> Optional[str]:
        with self._lock:
            return self._mode_override

    def push_frame(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._mjpeg_frame = jpeg_bytes

    def get_frame(self) -> bytes:
        with self._lock:
            return self._mjpeg_frame

    def update_runtime_config(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            self._runtime_config.update(updates)

    def get_runtime_config(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._runtime_config)

    def set_running(self, running: bool) -> None:
        with self._lock:
            self._running = running

    def is_running(self) -> bool:
        with self._lock:
            return self._running

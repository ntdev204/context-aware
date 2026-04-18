"""Thread-safe camera capture with double buffering.

Supports two backends:
  - "usb"   : standard OpenCV USB/CSI camera (RGB only)
  - "astra" : Orbbec Astra S via OpenNI2 (RGB + depth uint16 mm)

grab() always returns Tuple[np.ndarray | None, np.ndarray | None]
  (rgb_frame, depth_frame) -- depth_frame is None when unavailable
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Captures frames from CSI, USB, or Astra S camera in a background thread.

    Uses double-buffering: producer writes to back-buffer, consumer reads
    from front-buffer — no blocking between capture and inference.
    """

    def __init__(
        self,
        device_id: int = 0,
        backend: str = "usb",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.device_id = device_id
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps

        # "astra" backend streams both RGB and uint16 depth via OpenNI2
        self._has_depth: bool = backend == "astra"

        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._depth_frame: np.ndarray | None = None  # uint16, millimetres
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_count = 0

        # OpenNI2 device handles (populated in start() for astra backend)
        self._oni_device = None
        self._oni_depth_stream = None
        self._oni_color_stream = None
        self._openni2 = None  # cached module reference — set in _open_astra()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._has_depth:
            self._open_astra()
        else:
            self._cap = self._open_capture()

        self._running = True

        # No camera available after all fallbacks — run null loop
        if self._cap is None and not self._has_depth:
            logger.warning("No camera available — publishing null frames")
            self._thread = threading.Thread(target=self._null_loop, daemon=True)
            self._thread.start()
            return

        target = self._astra_capture_loop if self._has_depth else self._capture_loop
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

        # Wait for first RGB frame
        deadline = time.monotonic() + 5.0
        while self._frame is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._frame is None:
            raise RuntimeError("Camera did not produce a frame within 5 s")
        logger.info(
            "Camera started: %dx%d @ %d FPS [%s, depth=%s]",
            self.width,
            self.height,
            self.fps,
            self.backend,
            self._has_depth,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        # Release OpenNI2 streams if used
        if self._oni_depth_stream is not None:
            self._oni_depth_stream.stop()
        if self._oni_color_stream is not None:
            self._oni_color_stream.stop()
        if self._oni_device is not None:
            self._oni_device.close()
        if self._cap is not None:
            self._cap.release()
        logger.info("Camera stopped")

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------
    def grab(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (rgb_frame, depth_frame) — depth is None when unavailable.

        rgb_frame  : BGR uint8 numpy array, or None if camera not ready.
        depth_frame: uint16 numpy array in millimetres, or None.
        """
        with self._lock:
            rgb = self._frame.copy() if self._frame is not None else None
            depth = self._depth_frame.copy() if self._depth_frame is not None else None
        return rgb, depth

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _open_capture(self) -> cv2.VideoCapture:
        if self.backend == "csi":
            pipeline = self._gstreamer_csi_pipeline()
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(self.device_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device_id} [{self.backend}]")
        return cap

    def _capture_loop(self) -> None:
        """Standard OpenCV capture loop (USB / CSI backends)."""
        interval = 1.0 / self.fps
        while self._running:
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Camera read failed, retrying...")
                time.sleep(0.1)
                continue

            with self._lock:
                self._frame = frame
                self._frame_count += 1

            elapsed = time.monotonic() - t0
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _null_loop(self) -> None:
        """Produce black frames when no camera hardware is available."""
        interval = 1.0 / self.fps
        black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        while self._running:
            with self._lock:
                self._frame = black
                self._frame_count += 1
            time.sleep(interval)

    # ------------------------------------------------------------------
    # Astra S (OpenNI2) helpers
    # ------------------------------------------------------------------
    def _open_astra(self) -> None:
        """Initialise OpenNI2 and open colour + depth streams."""
        # Guard: OpenNI2 is not available on Windows — fall back silently
        import platform

        if platform.system() == "Windows":
            logger.warning("Astra backend requested on Windows — falling back to USB (no depth)")
            self._has_depth = False
            self._cap = self._open_capture()
            return

        try:
            import os as _os

            from openni import openni2  # type: ignore[import]

            self._openni2 = openni2

            # Orbbec official OpenNI2 binaries extracted to /usr/lib
            _redist = "/usr/lib"
            openni2.initialize(_redist if _os.path.isdir(_redist) else None)
            self._oni_device = openni2.Device.open_any()

            # Depth stream — uint16 mm
            self._oni_depth_stream = self._oni_device.create_depth_stream()
            self._oni_depth_stream.set_video_mode(
                openni2.c_api.OniVideoMode(
                    pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM,
                    resolutionX=self.width,
                    resolutionY=self.height,
                    fps=self.fps,
                )
            )
            self._oni_depth_stream.start()

            # Colour stream — BGR via OpenCV
            self._oni_color_stream = self._oni_device.create_color_stream()
            self._oni_color_stream.set_video_mode(
                openni2.c_api.OniVideoMode(
                    pixelFormat=openni2.PIXEL_FORMAT_RGB888,
                    resolutionX=self.width,
                    resolutionY=self.height,
                    fps=self.fps,
                )
            )
            self._oni_color_stream.start()

            # Align depth pixels to colour frame coordinates (Astra S hardware feature)
            try:
                self._oni_device.set_image_registration_mode(
                    openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
                )
            except Exception as reg_exc:
                logger.warning(
                    "Depth-to-colour registration not available on this OpenNI2 build: %s",
                    reg_exc,
                )

            logger.info("Astra S OpenNI2 streams opened (%dx%d)", self.width, self.height)

        except Exception as exc:
            # Gracefully degrade to no-depth if OpenNI2 fails
            logger.warning("Astra OpenNI2 init failed (%s) — falling back to USB rgb-only", exc)
            self._has_depth = False
            self.backend = "usb"  # switch away from 'astra' so _open_capture uses V4L2
            try:
                self._cap = self._open_capture()
            except RuntimeError as cam_exc:
                logger.warning("USB camera unavailable (%s) — running without camera", cam_exc)
                self._cap = None

    def _astra_capture_loop(self) -> None:
        """Capture loop for Astra S — reads RGB and depth independently.

        Depth failure per-frame does NOT kill the RGB stream; it simply
        results in depth_frame=None for that frame.
        """
        interval = 1.0 / self.fps
        while self._running:
            t0 = time.monotonic()

            # --- RGB (use cached module — no per-frame import) ---
            try:
                color_frame = self._oni_color_stream.read_frame()
                rgb_buf = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
                rgb = cv2.cvtColor(rgb_buf.reshape(self.height, self.width, 3), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                logger.warning("Astra RGB read failed: %s", exc)
                time.sleep(0.1)
                continue  # RGB failure → skip frame entirely

            # --- Depth (isolated try/except — failure yields None) ---
            depth: np.ndarray | None = None
            try:
                depth_frame = self._oni_depth_stream.read_frame()
                depth_buf = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
                depth = depth_buf.reshape(self.height, self.width)
            except Exception as exc:
                logger.debug("Astra depth read failed (frame skipped): %s", exc)

            with self._lock:
                self._frame = rgb
                self._depth_frame = depth
                self._frame_count += 1

            elapsed = time.monotonic() - t0
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _gstreamer_csi_pipeline(self) -> str:
        return (
            f"nvarguscamerasrc sensor-id={self.device_id} ! "
            f"video/x-raw(memory:NVMM),width={self.width},height={self.height},"
            f"format=NV12,framerate={self.fps}/1 ! "
            "nvvidconv flip-method=0 ! "
            f"video/x-raw,width={self.width},height={self.height},format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink"
        )

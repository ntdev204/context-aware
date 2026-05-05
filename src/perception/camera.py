from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
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

        self._has_depth: bool = backend == "astra"

        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._depth_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_count = 0

        self._oni_device = None
        self._oni_depth_stream = None
        self._oni_color_stream = None
        self._openni2 = None

    def start(self) -> None:
        if self._has_depth:
            self._open_astra()
        else:
            self._cap = self._open_capture()

        self._running = True

        if self._cap is None and not self._has_depth:
            logger.warning("No camera available — publishing null frames")
            self._thread = threading.Thread(target=self._null_loop, daemon=True)
            self._thread.start()
            return

        target = self._astra_capture_loop if self._has_depth else self._capture_loop
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

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
        if self._oni_depth_stream is not None:
            self._oni_depth_stream.stop()
        if self._oni_color_stream is not None:
            self._oni_color_stream.stop()
        if self._oni_device is not None:
            self._oni_device.close()
        if self._cap is not None:
            self._cap.release()
        logger.info("Camera stopped")

    def grab(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            rgb = self._frame.copy() if self._frame is not None else None
            depth = self._depth_frame.copy() if self._depth_frame is not None else None
        return rgb, depth

    @property
    def frame_count(self) -> int:
        return self._frame_count

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
        interval = 1.0 / self.fps
        black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        while self._running:
            with self._lock:
                self._frame = black
                self._frame_count += 1
            time.sleep(interval)

    def _open_astra(self) -> None:
        import platform

        if platform.system() == "Windows":
            logger.warning("Astra backend requested on Windows — falling back to USB (no depth)")
            self._has_depth = False
            self._cap = self._open_capture()
            return

        try:
            import os as _os

            from openni import openni2

            self._openni2 = openni2

            _redist = "/usr/lib"
            openni2.initialize(_redist if _os.path.isdir(_redist) else None)
            self._oni_device = openni2.Device.open_any()

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
            logger.warning("Astra OpenNI2 init failed (%s) — falling back to USB rgb-only", exc)
            self._has_depth = False
            self.backend = "usb"
            try:
                self._cap = self._open_capture()
            except RuntimeError as cam_exc:
                logger.warning("USB camera unavailable (%s) — running without camera", cam_exc)
                self._cap = None

    def _astra_capture_loop(self) -> None:
        interval = 1.0 / self.fps
        while self._running:
            t0 = time.monotonic()

            try:
                color_frame = self._oni_color_stream.read_frame()
                rgb_buf = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
                rgb = cv2.cvtColor(rgb_buf.reshape(self.height, self.width, 3), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                logger.warning("Astra RGB read failed: %s", exc)
                time.sleep(0.1)
                continue

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

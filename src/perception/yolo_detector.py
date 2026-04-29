"""YOLOv11s object detector — TensorRT (.engine) or PyTorch (.pt) backend."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

# Astra S valid depth range in millimetres
_DEPTH_MIN_MM: float = 2000.0
_DEPTH_MAX_MM: float = 8000.0
_DEPTH_MIN_VALID_PIXELS: int = 5  # minimum pixels in ROI that must be valid
_DEPTH_ROI_HALF: int = 5  # half-size of 10×10 sampling window

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "person",
    "obstacle",  # Legacy fallback
    "static_obstacle",  # Tĩnh: kiện hàng, bàn ghế, cột
    "dynamic_hazard",  # Động: xe đẩy hành lý, vali bị rớt, quả bóng
    "door",
    "wall",
]
CLASS_IDS = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class DetectionResult:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coords)
    class_id: int
    class_name: str
    confidence: float
    track_id: int = -1  # assigned by tracker
    distance: float = 0.0  # calibrated physical distance in metres
    distance_source: str = "bbox"  # "depth" | "bbox"
    intent_class: int = -1
    intent_name: str = "UNKNOWN"
    intent_confidence: float = 0.0
    dx: float = 0.0
    dy: float = 0.0


@dataclass
class FrameDetections:
    persons: list[DetectionResult] = field(default_factory=list)
    obstacles: list[DetectionResult] = field(default_factory=list)
    all_detections: list[DetectionResult] = field(default_factory=list)
    free_space_ratio: float = 1.0
    free_mask: np.ndarray | None = None
    obstacle_mask: np.ndarray | None = None
    unknown_mask: np.ndarray | None = None
    free_sectors: np.ndarray | None = None
    navigable_heading: float = 0.0
    navigable_width: float = 0.0
    navigable_width_m: float = 0.0
    freespace_processing_ms: float = 0.0
    timestamp: float = 0.0
    frame_id: int = 0
    inference_ms: float = 0.0
    frame_width: int = 640
    frame_height: int = 480


class YOLODetector:
    """Wraps Ultralytics YOLO — uses TensorRT engine on Jetson, GPU-accelerated PT on dev."""

    def __init__(
        self,
        model_path: str,
        use_tensorrt: bool = False,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        input_size: int | tuple[int, int] = (480, 640),  # (H, W)
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.use_tensorrt = use_tensorrt
        self.conf = confidence_threshold
        self.iou = iou_threshold
        self.input_size = input_size
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load model — call once before inference loop."""
        import torch
        from ultralytics import YOLO

        # Auto-detect device if cuda requested but not available
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        self._model = YOLO(self.model_path, task="detect")

        # Warm-up pass — dummy matches engine input shape (H, W)
        if isinstance(self.input_size, (list, tuple)):
            h, w = self.input_size
            self.input_size = (h, w)  # normalize list → tuple
        else:
            h = w = self.input_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.input_size,
            device=self.device,
        )

        logger.info(
            "YOLO loaded: %s [device=%s, TensorRT=%s]",
            self.model_path,
            self.device,
            self.use_tensorrt,
        )

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        depth_frame: np.ndarray | None = None,
    ) -> FrameDetections:
        """Run inference on *frame* (BGR numpy array). Returns FrameDetections.

        Parameters
        ----------
        depth_frame:
            Optional uint16 depth map in millimetres (same resolution as frame).
            When provided, depth is the primary distance source per-detection.
        """
        if self._model is None:
            raise RuntimeError("Call load() before detect()")

        t0 = time.monotonic()
        results = self._model.predict(
            frame,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.input_size,
            device=self.device,
        )
        inference_ms = (time.monotonic() - t0) * 1000

        h, w = frame.shape[:2]
        frame_det = FrameDetections(
            timestamp=time.time(),
            frame_id=frame_id,
            inference_ms=inference_ms,
            frame_width=w,
            frame_height=h,
        )

        if not results or results[0].boxes is None:
            return frame_det

        boxes = results[0].boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            raw_name = results[0].names[cls_id]
            class_name = self._map_to_nav_class(raw_name)

            if class_name == "ignore":
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

            det = DetectionResult(
                bbox=(x1, y1, x2, y2),
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
            )

            # Depth-first distance estimation
            det.distance, det.distance_source = self._estimate_distance(
                x1, y1, x2, y2, h, w, class_name, depth_frame
            )

            frame_det.all_detections.append(det)

            if class_name == "person":
                frame_det.persons.append(det)
            elif class_name in ("obstacle", "static_obstacle", "dynamic_hazard", "door", "wall"):
                frame_det.obstacles.append(det)
        return frame_det

    @staticmethod
    def _map_to_nav_class(raw_name: str) -> str:
        """Map raw YOLO class name (COCO or Custom) into Robot Navigation class."""
        if raw_name in CLASS_NAMES:
            return raw_name

        if raw_name == "person":
            return "person"

        dynamic_keywords = {"sports ball", "frisbee", "backpack", "suitcase", "handbag", "umbrella"}
        if raw_name in dynamic_keywords:
            return "dynamic_hazard"

        static_keywords = {
            "bench",
            "chair",
            "couch",
            "potted plant",
            "dining table",
            "tv",
            "laptop",
        }
        if raw_name in static_keywords:
            return "static_obstacle"

        return "ignore"

    @staticmethod
    def _estimate_distance(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_h: int,
        frame_w: int,
        class_name: str,
        depth_frame: np.ndarray | None,
    ) -> tuple[float, str]:
        """Hybrid distance estimation: depth camera primary, bbox heuristic fallback.

        Returns
        -------
        (distance_metres, source)  where source is "depth" or "bbox".
        """
        # Depth estimation (Astra S, uint16 mm)
        if depth_frame is not None:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 10x10 median depth around bbox centre
            r0 = max(0, cy - _DEPTH_ROI_HALF)
            r1 = min(depth_frame.shape[0], cy + _DEPTH_ROI_HALF)
            c0 = max(0, cx - _DEPTH_ROI_HALF)
            c1 = min(depth_frame.shape[1], cx + _DEPTH_ROI_HALF)

            roi = depth_frame[r0:r1, c0:c1].astype(np.float32)

            valid_mask = (roi >= _DEPTH_MIN_MM) & (roi <= _DEPTH_MAX_MM)
            valid_pixels = int(np.count_nonzero(valid_mask))

            if valid_pixels >= _DEPTH_MIN_VALID_PIXELS:
                depth_m = float(np.median(roi[valid_mask])) / 1000.0
                return round(depth_m, 3), "depth"

        bbox_height = max(1, y2 - y1)
        k_factor = 1.5 if class_name == "person" else 0.5
        dist_m = max(0.2, k_factor * frame_h / bbox_height)
        return round(dist_m, 3), "bbox"

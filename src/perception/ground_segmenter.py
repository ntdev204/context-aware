"""Depth-based free-space segmentation for the Astra S RGB-D camera."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class FreeSpaceResult:
    free_mask: np.ndarray
    obstacle_mask: np.ndarray
    unknown_mask: np.ndarray
    free_space_ratio: float
    free_sectors: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    navigable_heading: float = 0.0
    navigable_width: float = 0.0
    processing_ms: float = 0.0


class GroundSegmenter:
    """Classify drivable ground from one registered Astra S depth frame.

    The segmenter uses a conservative three-state mask:
    free, obstacle, and unknown. Invalid or out-of-range depth is always unknown,
    never free.
    """

    def __init__(
        self,
        fx: float = 570.0,
        fy: float = 570.0,
        cx: float = 320.0,
        cy: float = 240.0,
        camera_height_m: float = 0.50,
        camera_pitch_deg: float = 0.0,
        depth_min_mm: int = 400,
        depth_max_mm: int = 2000,
        downscale: int = 4,
        ground_tolerance_m: float = 0.08,
        obstacle_height_m: float = 0.10,
        safety_margin_px: int = 18,
        sector_count: int = 8,
        sector_free_threshold: float = 0.55,
        fov_deg: float | None = None,
    ) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.camera_height_m = float(camera_height_m)
        self.camera_pitch_rad = math.radians(float(camera_pitch_deg))
        self.depth_min_mm = int(depth_min_mm)
        self.depth_max_mm = int(depth_max_mm)
        self.downscale = max(1, int(downscale))
        self.ground_tolerance_m = float(ground_tolerance_m)
        self.obstacle_height_m = float(obstacle_height_m)
        self.safety_margin_px = max(0, int(safety_margin_px))
        self.sector_count = max(1, int(sector_count))
        self.sector_free_threshold = float(sector_free_threshold)
        self.fov_rad = math.radians(float(fov_deg)) if fov_deg is not None else None
        self._ray_cache: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    def segment(
        self,
        depth_frame: np.ndarray | None,
        detections: Any | None = None,
        frame_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    ) -> FreeSpaceResult:
        t0 = time.monotonic()

        if depth_frame is None:
            h, w = self._shape_from_frame(frame_shape)
            return self._empty_unknown(h, w, (time.monotonic() - t0) * 1000.0)

        if depth_frame.ndim != 2:
            raise ValueError(f"depth_frame must be 2-D uint16 mm, got shape {depth_frame.shape}")

        depth_h, depth_w = depth_frame.shape
        frame_h, frame_w = self._shape_from_frame(frame_shape, fallback=(depth_h, depth_w))

        small_w = max(1, depth_w // self.downscale)
        small_h = max(1, depth_h // self.downscale)
        depth_small = cv2.resize(depth_frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

        valid = (depth_small >= self.depth_min_mm) & (depth_small <= self.depth_max_mm)
        if not np.any(valid):
            result = self._empty_unknown(depth_h, depth_w, (time.monotonic() - t0) * 1000.0)
            return self._resize_result(result, frame_h, frame_w)

        ray_x, ray_y = self._ray_maps(small_h, small_w, depth_h, depth_w)
        z = depth_small.astype(np.float32) / 1000.0
        y_down = ray_y * z

        cos_p = math.cos(self.camera_pitch_rad)
        sin_p = math.sin(self.camera_pitch_rad)
        vertical_down = cos_p * y_down + sin_p * z
        above_ground = self.camera_height_m - vertical_down

        ground = valid & (np.abs(above_ground) <= self.ground_tolerance_m)
        obstacle = valid & (above_ground >= self.obstacle_height_m)

        obstacle = self._dilate(obstacle, self.safety_margin_px / self.downscale)
        free = ground & ~obstacle
        unknown = ~(free | obstacle)

        free_full = self._resize_mask(free, depth_h, depth_w)
        obstacle_full = self._resize_mask(obstacle, depth_h, depth_w)
        unknown_full = self._resize_mask(unknown, depth_h, depth_w)

        free_full, obstacle_full, unknown_full = self._stamp_detections(
            free_full,
            obstacle_full,
            unknown_full,
            detections,
        )

        if (frame_h, frame_w) != (depth_h, depth_w):
            free_full = self._resize_mask(free_full, frame_h, frame_w)
            obstacle_full = self._resize_mask(obstacle_full, frame_h, frame_w)
            unknown_full = self._resize_mask(unknown_full, frame_h, frame_w)

        free_ratio = self._free_ratio(free_full, obstacle_full)
        sectors = self._sector_ratios(free_full, obstacle_full)
        heading, width = self._navigable_corridor(sectors, frame_w)

        return FreeSpaceResult(
            free_mask=free_full,
            obstacle_mask=obstacle_full,
            unknown_mask=unknown_full,
            free_space_ratio=free_ratio,
            free_sectors=sectors,
            navigable_heading=heading,
            navigable_width=width,
            processing_ms=(time.monotonic() - t0) * 1000.0,
        )

    def apply_to_frame(self, frame_det: Any, result: FreeSpaceResult) -> Any:
        """Attach segmentation output to a FrameDetections-like object."""
        frame_det.free_mask = result.free_mask
        frame_det.obstacle_mask = result.obstacle_mask
        frame_det.unknown_mask = result.unknown_mask
        frame_det.free_space_ratio = result.free_space_ratio
        frame_det.free_sectors = result.free_sectors
        frame_det.navigable_heading = result.navigable_heading
        frame_det.navigable_width = result.navigable_width
        frame_det.freespace_processing_ms = result.processing_ms
        return frame_det

    def _ray_maps(
        self,
        small_h: int,
        small_w: int,
        depth_h: int,
        depth_w: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (small_h, small_w, depth_h, depth_w)
        cached = self._ray_cache.get(key)
        if cached is not None:
            return cached

        scale_x = small_w / max(depth_w, 1)
        scale_y = small_h / max(depth_h, 1)
        fx = self.fx * scale_x
        fy = self.fy * scale_y
        cx = self.cx * scale_x
        cy = self.cy * scale_y

        u = np.arange(small_w, dtype=np.float32)[None, :]
        v = np.arange(small_h, dtype=np.float32)[:, None]
        ray_x = (u - cx) / max(fx, 1e-6)
        ray_y = (v - cy) / max(fy, 1e-6)
        self._ray_cache[key] = (ray_x, ray_y)
        return ray_x, ray_y

    def _stamp_detections(
        self,
        free: np.ndarray,
        obstacle: np.ndarray,
        unknown: np.ndarray,
        detections: Any | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if detections is None:
            return free, obstacle, unknown

        dets = []
        if hasattr(detections, "persons") or hasattr(detections, "obstacles"):
            dets.extend(getattr(detections, "persons", []) or [])
            dets.extend(getattr(detections, "obstacles", []) or [])
        elif isinstance(detections, (list, tuple)):
            dets.extend(detections)

        if not dets:
            return free, obstacle, unknown

        h, w = free.shape
        margin = max(0, self.safety_margin_px)
        for det in dets:
            bbox = getattr(det, "bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0, int(x1) - margin)
            y1 = max(0, int(y1) - margin)
            x2 = min(w, int(x2) + margin)
            y2 = min(h, int(y2) + margin)
            if x2 <= x1 or y2 <= y1:
                continue
            free[y1:y2, x1:x2] = False
            obstacle[y1:y2, x1:x2] = True
            unknown[y1:y2, x1:x2] = False

        return free, obstacle, unknown

    def _sector_ratios(self, free: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
        h, w = free.shape
        sectors = np.zeros(self.sector_count, dtype=np.float32)
        known = free | obstacle
        edges = np.linspace(0, w, self.sector_count + 1, dtype=np.int32)
        for i in range(self.sector_count):
            x0, x1 = int(edges[i]), int(edges[i + 1])
            if x1 <= x0:
                continue
            known_count = int(np.count_nonzero(known[:, x0:x1]))
            if known_count == 0:
                sectors[i] = 0.0
            else:
                sectors[i] = float(np.count_nonzero(free[:, x0:x1]) / known_count)
        return sectors

    def _navigable_corridor(self, sectors: np.ndarray, frame_w: int) -> tuple[float, float]:
        if sectors.size == 0 or float(np.max(sectors)) <= 0.0:
            return 0.0, 0.0

        best = int(np.argmax(sectors))
        left = best
        right = best
        while left - 1 >= 0 and sectors[left - 1] >= self.sector_free_threshold:
            left -= 1
        while right + 1 < sectors.size and sectors[right + 1] >= self.sector_free_threshold:
            right += 1

        fov = self._fov_rad(frame_w)
        sector_width = fov / sectors.size
        center_index = (left + right + 1) / 2.0
        heading = -fov / 2.0 + center_index * sector_width
        width = (right - left + 1) * sector_width
        return float(heading), float(width)

    def _fov_rad(self, frame_w: int) -> float:
        if self.fov_rad is not None:
            return self.fov_rad
        return 2.0 * math.atan(frame_w / (2.0 * max(self.fx, 1e-6)))

    @staticmethod
    def _shape_from_frame(
        frame_shape: tuple[int, int] | tuple[int, int, int] | None,
        fallback: tuple[int, int] = (480, 640),
    ) -> tuple[int, int]:
        if frame_shape is None:
            return fallback
        if len(frame_shape) < 2:
            return fallback
        return int(frame_shape[0]), int(frame_shape[1])

    @staticmethod
    def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
        return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(
            bool
        )

    @staticmethod
    def _dilate(mask: np.ndarray, radius_px: float) -> np.ndarray:
        radius = int(math.ceil(radius_px))
        if radius <= 0 or not np.any(mask):
            return mask
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    @staticmethod
    def _free_ratio(free: np.ndarray, obstacle: np.ndarray) -> float:
        known = free | obstacle
        known_count = int(np.count_nonzero(known))
        if known_count == 0:
            return 0.0
        return float(np.count_nonzero(free) / known_count)

    def _empty_unknown(self, h: int, w: int, processing_ms: float) -> FreeSpaceResult:
        free = np.zeros((h, w), dtype=bool)
        obstacle = np.zeros((h, w), dtype=bool)
        unknown = np.ones((h, w), dtype=bool)
        return FreeSpaceResult(
            free_mask=free,
            obstacle_mask=obstacle,
            unknown_mask=unknown,
            free_space_ratio=0.0,
            free_sectors=np.zeros(self.sector_count, dtype=np.float32),
            navigable_heading=0.0,
            navigable_width=0.0,
            processing_ms=processing_ms,
        )

    def _resize_result(self, result: FreeSpaceResult, h: int, w: int) -> FreeSpaceResult:
        if result.free_mask.shape == (h, w):
            return result
        result.free_mask = self._resize_mask(result.free_mask, h, w)
        result.obstacle_mask = self._resize_mask(result.obstacle_mask, h, w)
        result.unknown_mask = self._resize_mask(result.unknown_mask, h, w)
        return result

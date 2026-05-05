from __future__ import annotations

import math

import cv2
import numpy as np


def draw_freespace_overlay(
    frame: np.ndarray,
    frame_det,
    alpha: float = 0.30,
    copy: bool = True,
) -> np.ndarray:
    vis = frame.copy() if copy else frame
    free = getattr(frame_det, "free_mask", None)
    obstacle = getattr(frame_det, "obstacle_mask", None)
    unknown = getattr(frame_det, "unknown_mask", None)

    if free is not None:
        _blend_mask(vis, _match_mask(free, vis.shape[:2]), (0, 180, 0), alpha)
    if obstacle is not None:
        _blend_mask(vis, _match_mask(obstacle, vis.shape[:2]), (0, 0, 180), 0.18)
    if unknown is not None:
        _blend_mask(vis, _match_mask(unknown, vis.shape[:2]), (80, 80, 80), 0.10)

    _draw_sector_radar(vis, getattr(frame_det, "free_sectors", None))
    _draw_freespace_text(
        vis,
        float(getattr(frame_det, "free_space_ratio", 0.0) or 0.0),
        float(getattr(frame_det, "navigable_width", 0.0) or 0.0),
        float(getattr(frame_det, "navigable_width_m", 0.0) or 0.0),
    )
    return vis


def _match_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask.shape == (h, w):
        return mask.astype(bool, copy=False)
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _blend_mask(
    vis: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float
) -> None:
    if mask.size == 0 or not np.any(mask):
        return
    inv_alpha = 1.0 - alpha
    for channel_idx, target in enumerate(color):
        channel = vis[:, :, channel_idx]
        values = channel[mask].astype(np.float32)
        channel[mask] = (values * inv_alpha + target * alpha).astype(np.uint8)


def _draw_freespace_text(vis: np.ndarray, ratio: float, width_rad: float, width_m: float) -> None:
    text = f"FREE:{ratio * 100:.0f}% | WIDTH:{width_m:.1f}m | CORRIDOR:{math.degrees(width_rad):.0f}deg"
    h, w = vis.shape[:2]
    org = (8, max(48, h - 12))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    x2 = min(w - 4, org[0] + tw + 8)
    y1 = max(0, org[1] - th - baseline - 6)
    cv2.rectangle(vis, (4, y1), (x2, org[1] + baseline + 3), (0, 0, 0), cv2.FILLED)
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 255, 80), 1)


def _draw_sector_radar(vis: np.ndarray, sectors) -> None:
    if sectors is None:
        return
    vals = np.asarray(sectors, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return

    vals = np.clip(vals[:8], 0.0, 1.0)
    h, w = vis.shape[:2]
    size = min(104, max(72, w // 7))
    pad = 10
    x0 = max(0, w - size - pad)
    y0 = max(0, h - size - pad)
    cx = x0 + size // 2
    cy = y0 + size - 8
    radius = size - 18

    panel = vis[y0 : y0 + size, x0 : x0 + size]
    black = np.zeros_like(panel)
    cv2.addWeighted(panel, 0.55, black, 0.45, 0, dst=panel)

    sector_angle = 90.0 / len(vals)
    start_angle = -135.0
    for i, val in enumerate(vals):
        a0 = start_angle + i * sector_angle
        a1 = a0 + sector_angle
        color = _sector_color(float(val))
        points = [(cx, cy)]
        for angle in np.linspace(a0, a1, 5):
            rad = math.radians(angle)
            points.append((int(cx + radius * math.cos(rad)), int(cy + radius * math.sin(rad))))
        pts = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(vis, pts, color)
        cv2.polylines(vis, [pts], True, (20, 20, 20), 1)

    cv2.circle(vis, (cx, cy), 4, (255, 255, 255), cv2.FILLED)


def _sector_color(value: float) -> tuple[int, int, int]:
    if value >= 0.65:
        return (0, 180, 0)
    if value >= 0.35:
        return (0, 180, 180)
    return (0, 0, 190)

from __future__ import annotations

import math

import cv2
import numpy as np


def draw_detections(
    frame: np.ndarray,
    persons: list,
    obstacles: list,
    mode_name: str,
    fps: float,
    follow_target_id: int = -1,
    local_plan=None,
    copy: bool = True,
) -> np.ndarray:
    vis = frame.copy() if copy else frame

    for p in persons:
        x1, y1, x2, y2 = p.bbox
        locked = p.track_id == follow_target_id
        stale = getattr(p, "stale", False)
        state = "LOCK LOST " if locked and stale else "LOCK " if locked else "LOST " if stale else ""
        label = f"{state}ID:{p.track_id} {p.intent_name} {p.intent_confidence:.0%}"
        color = (0, 255, 255) if locked else (0, 200, 0)
        thickness = 1 if stale else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(vis, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(
            vis, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )

    for ob in obstacles:
        x1, y1, x2, y2 = ob.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 220), 2)

    hud = f"{mode_name} | FPS:{fps:.0f}"
    (tw, th), baseline = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(vis, (4, 4), (tw + 12, th + baseline + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(vis, hud, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    _draw_local_plan(vis, local_plan)

    return vis


def _draw_local_plan(vis: np.ndarray, local_plan) -> None:
    if local_plan is None:
        return

    status = getattr(local_plan, "status", "idle")
    if status in ("idle", "no_target"):
        return

    h, w = vis.shape[:2]
    panel_w, panel_h = 190, 150
    x0 = max(8, w - panel_w - 8)
    y0 = max(42, h - panel_h - 8)
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.58, vis, 0.42, 0, vis)
    cv2.rectangle(vis, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1)

    origin = (x0 + panel_w // 2, y0 + panel_h - 22)
    scale = 25.0
    max_x_m = 4.5

    cv2.line(vis, (origin[0], y0 + 10), (origin[0], origin[1]), (70, 70, 70), 1)
    cv2.line(vis, (x0 + 12, origin[1]), (x0 + panel_w - 12, origin[1]), (70, 70, 70), 1)
    cv2.circle(vis, origin, 5, (255, 255, 255), cv2.FILLED)

    lidar = getattr(local_plan, "lidar_sectors", None)
    scan = getattr(local_plan, "lidar_scan", None)
    if scan:
        _draw_lidar_scan360(vis, origin, scale, scan)
    elif lidar:
        _draw_lidar_rays(vis, origin, scale, lidar)

    for ob in getattr(local_plan, "obstacles", []):
        px, py = _plan_to_px(origin, scale, ob.x, ob.y, max_x_m)
        radius = max(3, int(getattr(ob, "radius", 0.3) * scale))
        cv2.circle(vis, (px, py), radius, (0, 0, 220), 1)

    points = []
    for wp in getattr(local_plan, "waypoints", []):
        points.append(_plan_to_px(origin, scale, wp.x, wp.y, max_x_m))
    if points:
        prev = origin
        color = (0, 255, 255) if status == "replanned" else (0, 220, 0)
        if status == "blocked":
            color = (0, 0, 255)
        for pt in points:
            cv2.line(vis, prev, pt, color, 2)
            cv2.circle(vis, pt, 3, color, cv2.FILLED)
            prev = pt

    label = f"PLAN:{status.upper()}"
    color = (0, 0, 255) if status == "blocked" else (0, 255, 255)
    cv2.putText(vis, label, (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(vis, label, (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def _draw_lidar_rays(
    vis: np.ndarray,
    origin: tuple[int, int],
    scale: float,
    lidar: tuple[float, float, float, float] | list[float],
) -> None:
    angles = [0.0, math.pi, math.pi / 2.0, -math.pi / 2.0]
    colors = [(0, 220, 0), (0, 160, 160), (0, 220, 0), (0, 220, 0)]
    for distance, angle, color in zip(lidar[:4], angles, colors, strict=False):
        try:
            dist = float(distance)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(dist) or dist <= 0.0:
            continue
        dist = min(dist, 4.5)
        x = math.cos(angle) * dist
        y = math.sin(angle) * dist
        pt = _plan_to_px(origin, scale, x, y, 4.5)
        cv2.line(vis, origin, pt, color, 1)


def _draw_lidar_scan360(
    vis: np.ndarray,
    origin: tuple[int, int],
    scale: float,
    scan: tuple[float, ...] | list[float],
) -> None:
    n = len(scan)
    if n == 0:
        return
    step = max(1, n // 36)
    for idx in range(0, n, step):
        try:
            dist = float(scan[idx])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(dist) or dist <= 0.0 or dist >= 9.9:
            continue
        angle = math.radians(idx * 360.0 / n)
        x = math.cos(angle) * min(dist, 4.5)
        y = math.sin(angle) * min(dist, 4.5)
        pt = _plan_to_px(origin, scale, x, y, 4.5)
        cv2.line(vis, origin, pt, (70, 160, 70), 1)


def _plan_to_px(
    origin: tuple[int, int],
    scale: float,
    x_m: float,
    y_m: float,
    max_x_m: float,
) -> tuple[int, int]:
    x_m = max(-1.0, min(max_x_m, float(x_m)))
    y_m = max(-3.0, min(3.0, float(y_m)))
    px = int(origin[0] - y_m * scale)
    py = int(origin[1] - x_m * scale)
    return px, py


def encode_jpeg(frame: np.ndarray, quality: int = 70) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None

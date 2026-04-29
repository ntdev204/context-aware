from __future__ import annotations

import cv2
import numpy as np


def draw_detections(
    frame: np.ndarray,
    persons: list,
    obstacles: list,
    mode_name: str,
    fps: float,
    copy: bool = True,
) -> np.ndarray:
    vis = frame.copy() if copy else frame

    for p in persons:
        x1, y1, x2, y2 = p.bbox
        stale = getattr(p, "stale", False)
        state = "LOST " if stale else ""
        label = f"{state}ID:{p.track_id} {p.intent_name} {p.intent_confidence:.0%}"
        color = (0, 200, 0)
        thickness = 1 if stale else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(vis, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(vis, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    for ob in obstacles:
        x1, y1, x2, y2 = ob.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 220), 2)

    hud = f"{mode_name} | FPS:{fps:.0f}"
    (tw, th), baseline = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(vis, (4, 4), (tw + 12, th + baseline + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(vis, hud, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return vis


def encode_jpeg(frame: np.ndarray, quality: int = 70) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None

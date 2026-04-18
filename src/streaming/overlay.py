"""Frame annotation utilities for the Edge API video stream."""

from __future__ import annotations

import cv2
import numpy as np


def draw_detections(
    frame: np.ndarray,
    persons: list,
    obstacles: list,
    mode_name: str,
    fps: float,
) -> np.ndarray:
    """Return an annotated copy of frame. Does not mutate the input."""
    vis = frame.copy()

    for p in persons:
        x1, y1, x2, y2 = p.bbox
        label = f"ID:{p.track_id} {p.intent_name} {p.intent_confidence:.0%}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(vis, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(vis, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    for ob in obstacles:
        x1, y1, x2, y2 = ob.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 220), 2)

    hud = f"{mode_name} | FPS:{fps:.0f}"
    # Measure text size so the background bar fits
    (tw, th), baseline = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # Draw solid dark bar behind text
    cv2.rectangle(vis, (4, 4), (tw + 12, th + baseline + 10), (0, 0, 0), cv2.FILLED)
    # Draw text on top of bar
    cv2.putText(vis, hud, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return vis


def encode_jpeg(frame: np.ndarray, quality: int = 70) -> bytes | None:
    """Encode a frame to JPEG bytes. Returns None on failure."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None

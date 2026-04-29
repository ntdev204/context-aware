from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp

    _HAS_MEDIAPIPE = True
except ImportError:
    mp = None
    _HAS_MEDIAPIPE = False


@dataclass
class GestureResult:
    gesture: str = "none"
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    fingers: int = 0


class GestureDetector:
    def __init__(self, enabled: bool = True, min_confidence: float = 0.65) -> None:
        self.enabled = enabled
        self.min_confidence = min_confidence
        self._hands = None
        if enabled and _HAS_MEDIAPIPE:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
            logger.info("Gesture detector backend: mediapipe")
        elif enabled:
            logger.warning("mediapipe not installed; using OpenCV skin-contour gesture fallback")

    def detect(self, frame: np.ndarray) -> GestureResult:
        if not self.enabled:
            return GestureResult()
        if self._hands is not None:
            return self._detect_mediapipe(frame)
        return self._detect_contour(frame)

    def _detect_mediapipe(self, frame: np.ndarray) -> GestureResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return GestureResult()

        hand = result.multi_hand_landmarks[0]
        h, w = frame.shape[:2]
        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]
        bbox = (
            max(0, int(min(xs) * w)),
            max(0, int(min(ys) * h)),
            min(w, int(max(xs) * w)),
            min(h, int(max(ys) * h)),
        )

        fingers = self._count_mediapipe_fingers(hand.landmark)
        if fingers >= 4:
            return GestureResult("open_palm", 0.9, bbox, fingers)
        if fingers <= 1:
            return GestureResult("fist", 0.85, bbox, fingers)
        return GestureResult("none", 0.5, bbox, fingers)

    @staticmethod
    def _count_mediapipe_fingers(landmarks) -> int:
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        fingers = sum(1 for tip, pip in zip(tips, pips) if landmarks[tip].y < landmarks[pip].y)
        thumb_open = abs(landmarks[4].x - landmarks[17].x) > abs(landmarks[3].x - landmarks[17].x)
        return fingers + int(thumb_open)

    def _detect_contour(self, frame: np.ndarray) -> GestureResult:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 180, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 30, 60]), np.array([180, 180, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return GestureResult()
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < max(1800, w * h * 0.006):
            return GestureResult()

        x, y, bw, bh = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour, returnPoints=False)
        fingers = 0
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    _, _, _, depth = defects[i, 0]
                    if depth / 256.0 > 14:
                        fingers += 1

        if fingers >= 4:
            return GestureResult("open_palm", 0.62, (x, y, x + bw, y + bh), 5)
        if fingers <= 1 and area > w * h * 0.012:
            return GestureResult("fist", 0.58, (x, y, x + bw, y + bh), 0)
        return GestureResult("none", 0.4, (x, y, x + bw, y + bh), fingers)

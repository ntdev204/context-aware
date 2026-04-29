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
        frame_area = float(h * w)
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

        best_open: tuple[float, GestureResult] | None = None
        best_fist: tuple[float, GestureResult] | None = None

        min_area = max(1400.0, frame_area * 0.004)
        max_area = frame_area * 0.08

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if bw <= 0 or bh <= 0:
                continue
            aspect = bw / float(bh)
            if aspect < 0.45 or aspect > 2.2:
                continue

            hull_points = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull_points))
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            extent = area / float(max(1, bw * bh))
            fingers = self._count_contour_defects(contour)

            bbox = (x, y, x + bw, y + bh)
            area_factor = min(area / max_area, 1.0)

            if fingers >= 3 and solidity < 0.88:
                score = 0.55 + min(fingers, 5) * 0.06 + area_factor * 0.08
                result = GestureResult("open_palm", min(score, 0.92), bbox, min(fingers + 1, 5))
                if best_open is None or score > best_open[0]:
                    best_open = (score, result)

            if fingers <= 1 and solidity > 0.72 and extent > 0.36:
                score = 0.50 + solidity * 0.22 + extent * 0.12 + area_factor * 0.06
                result = GestureResult("fist", min(score, 0.86), bbox, 0)
                if best_fist is None or score > best_fist[0]:
                    best_fist = (score, result)

        if best_open and (not best_fist or best_open[0] >= best_fist[0] + 0.05):
            return best_open[1]
        if best_fist:
            return best_fist[1]
        return GestureResult()

    @staticmethod
    def _count_contour_defects(contour: np.ndarray) -> int:
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) <= 3:
            return 0
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0

        count = 0
        for i in range(defects.shape[0]):
            start_idx, end_idx, far_idx, depth = defects[i, 0]
            start = contour[start_idx][0]
            end = contour[end_idx][0]
            far = contour[far_idx][0]

            a = np.linalg.norm(end - start)
            b = np.linalg.norm(far - start)
            c = np.linalg.norm(end - far)
            if b <= 1e-3 or c <= 1e-3:
                continue
            angle = np.degrees(np.arccos(np.clip((b * b + c * c - a * a) / (2 * b * c), -1.0, 1.0)))
            if depth / 256.0 > 10 and angle < 95:
                count += 1
        return count

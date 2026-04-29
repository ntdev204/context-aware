from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2

logger = logging.getLogger(__name__)

try:
    from mediapipe.python.solutions import hands as mp_hands

    _HAS_MEDIAPIPE = True
    _MEDIAPIPE_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    mp_hands = None
    _HAS_MEDIAPIPE = False
    _MEDIAPIPE_IMPORT_ERROR = exc


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
        if enabled and not _HAS_MEDIAPIPE:
            raise RuntimeError(
                "mediapipe is required for gesture detection. "
                "Rebuild the Jetson image with the updated requirements."
            ) from _MEDIAPIPE_IMPORT_ERROR

        if enabled:
            self._hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
            logger.info("Gesture detector backend: mediapipe")

    def detect(self, frame: np.ndarray) -> GestureResult:
        if not self.enabled:
            return GestureResult()
        return self._detect_mediapipe(frame)

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

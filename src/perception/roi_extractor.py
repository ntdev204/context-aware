from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .yolo_detector import DetectionResult, FrameDetections

logger = logging.getLogger(__name__)

CNN_INPUT_W = 128
CNN_INPUT_H = 256


@dataclass
class PersonROI:
    image: np.ndarray
    bbox: tuple[int, int, int, int]
    track_id: int
    relative_position: tuple[float, float]
    distance_estimate: float = 0.0


class ROIExtractor:
    def __init__(
        self,
        output_width: int = CNN_INPUT_W,
        output_height: int = CNN_INPUT_H,
        padding_ratio: float = 0.10,
    ) -> None:
        self.output_width = output_width
        self.output_height = output_height
        self.padding_ratio = padding_ratio

    def extract(self, frame: np.ndarray, frame_det: FrameDetections) -> list[PersonROI]:
        h, w = frame.shape[:2]
        rois: list[PersonROI] = []

        for person in frame_det.persons:
            roi = self._crop_person(frame, person, h, w)
            if roi is not None:
                rois.append(roi)

        return rois

    def _crop_person(
        self,
        frame: np.ndarray,
        person: DetectionResult,
        frame_h: int,
        frame_w: int,
    ) -> PersonROI | None:
        x1, y1, x2, y2 = person.bbox
        bw = x2 - x1
        bh = y2 - y1

        if bw < 10 or bh < 10:
            return None

        pad_x = int(bw * self.padding_ratio)
        pad_y = int(bh * self.padding_ratio)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(frame_w, x2 + pad_x)
        cy2 = min(frame_h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        resized = cv2.resize(
            crop, (self.output_width, self.output_height), interpolation=cv2.INTER_LINEAR
        )

        cx_norm = ((x1 + x2) / 2.0) / frame_w
        cy_norm = ((y1 + y2) / 2.0) / frame_h

        dist_estimate = max(0.0, 1.0 - (bh / frame_h))

        return PersonROI(
            image=resized,
            bbox=person.bbox,
            track_id=person.track_id,
            relative_position=(cx_norm, cy_norm),
            distance_estimate=dist_estimate,
        )

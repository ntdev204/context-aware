"""ByteTrack-based multi-object tracker for stable track_id assignment.

Primary backend: supervision.ByteTrack (pip install supervision).
Fallback: minimal IoU tracker with min_hits warm-up to suppress ghost tracks.
"""

from __future__ import annotations

import logging

import numpy as np

from .yolo_detector import DetectionResult, FrameDetections

logger = logging.getLogger(__name__)

# Prefer supervision.ByteTrack — works with any ultralytics version.
try:
    import supervision as sv

    _HAS_SUPERVISION = True
    logger.info("ByteTrack backend: supervision.ByteTrack")
except ImportError:  # pragma: no cover
    _HAS_SUPERVISION = False
    logger.warning(
        "supervision not installed — using fallback IoU tracker. Run: pip install supervision"
    )


class _FallbackTracker:
    """Minimal IoU tracker with min_hits warm-up (used when supervision is unavailable).

    A detection must be matched at least *min_hits* consecutive frames before
    its track_id is exposed — this suppresses ghost tracks from single-frame
    noise, matching ByteTrack's behaviour.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        # track entry: {"bbox", "age", "hits"}
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(self, detections: list[DetectionResult]) -> list[DetectionResult]:
        if not detections:
            self._age_tracks()
            return []

        if not self._tracks:
            for det in detections:
                self._tracks[self._next_id] = {"bbox": det.bbox, "age": 0, "hits": 1}
                self._next_id += 1
            # None of the new tracks have enough hits yet
            self._age_tracks()
            return []

        matched_ids: set[int] = set()
        for det in detections:
            best_id, best_iou = -1, self.iou_threshold
            for tid, track in self._tracks.items():
                iou = self._iou(det.bbox, track["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id != -1:
                self._tracks[best_id]["bbox"] = det.bbox
                self._tracks[best_id]["age"] = 0
                self._tracks[best_id]["hits"] += 1
                matched_ids.add(best_id)
                # Only expose track_id once the track is confirmed
                if self._tracks[best_id]["hits"] >= self.min_hits:
                    det.track_id = best_id
            else:
                self._tracks[self._next_id] = {"bbox": det.bbox, "age": 0, "hits": 1}
                self._next_id += 1
                # track_id stays -1 (unconfirmed)

        self._age_tracks()
        return [d for d in detections if d.track_id != -1]

    def _age_tracks(self) -> None:
        dead = [tid for tid, t in self._tracks.items() if t["age"] >= self.max_age]
        for tid in dead:
            del self._tracks[tid]
        for t in self._tracks.values():
            t["age"] += 1

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)


class Tracker:
    """Wraps supervision.ByteTrack (preferred) or IoU fallback to assign stable track_ids."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._impl = self._build_tracker()

    def _build_tracker(self):
        if _HAS_SUPERVISION:
            return sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=self.max_age,
                minimum_matching_threshold=1.0 - self.iou_threshold,
                frame_rate=30,
                minimum_consecutive_frames=self.min_hits,
            )
        return _FallbackTracker(
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold,
        )

    def update(self, frame_det: FrameDetections, frame_shape: tuple) -> FrameDetections:
        """Assign track_ids to all person detections in *frame_det*."""
        if not frame_det.persons:
            return frame_det

        if _HAS_SUPERVISION:
            frame_det.persons = self._update_supervision(frame_det.persons)
        else:
            frame_det.persons = self._impl.update(frame_det.persons)

        # Mirror confirmed track_ids into all_detections list
        person_tracks = {d.bbox: d.track_id for d in frame_det.persons}
        for det in frame_det.all_detections:
            if det.class_name == "person":
                det.track_id = person_tracks.get(det.bbox, -1)

        return frame_det

    def _update_supervision(self, persons: list[DetectionResult]) -> list[DetectionResult]:
        """Use supervision.ByteTrack — accepts sv.Detections natively."""
        xyxy = np.array([list(d.bbox) for d in persons], dtype=np.float32)
        confs = np.array([d.confidence for d in persons], dtype=np.float32)
        class_ids = np.array([d.class_id for d in persons], dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=class_ids,
        )
        tracked = self._impl.update_with_detections(sv_dets)

        confirmed: list[DetectionResult] = []
        for i in range(len(tracked)):
            tx1, ty1, tx2, ty2 = (int(v) for v in tracked.xyxy[i])
            tid = int(tracked.tracker_id[i])
            for det in persons:
                dx1, dy1, dx2, dy2 = det.bbox
                if abs(dx1 - tx1) < 10 and abs(dy1 - ty1) < 10:
                    det.track_id = tid
                    confirmed.append(det)
                    break
        return confirmed

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from .yolo_detector import DetectionResult, FrameDetections

logger = logging.getLogger(__name__)

try:
    import supervision as sv

    _HAS_SUPERVISION = True
    logger.info("ByteTrack backend: supervision.ByteTrack")
except ImportError:
    _HAS_SUPERVISION = False
    logger.warning(
        "supervision not installed — using fallback IoU tracker. Run: pip install supervision"
    )


class _FallbackTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(self, detections: list[DetectionResult]) -> list[DetectionResult]:
        if not detections:
            self._age_tracks()
            return []

        if not self._tracks:
            for det in detections:
                self._tracks[self._next_id] = {"bbox": det.bbox, "age": 0, "hits": 1}
                if self.min_hits <= 1:
                    det.track_id = self._next_id
                self._next_id += 1
            self._age_tracks()
            return [d for d in detections if d.track_id != -1]

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
                if self._tracks[best_id]["hits"] >= self.min_hits:
                    det.track_id = best_id
            else:
                self._tracks[self._next_id] = {"bbox": det.bbox, "age": 0, "hits": 1}
                if self.min_hits <= 1:
                    det.track_id = self._next_id
                self._next_id += 1

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
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        hold_missing: int = 10,
        bbox_smoothing_alpha: float = 0.65,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.hold_missing = max(0, int(hold_missing))
        self.bbox_smoothing_alpha = float(np.clip(bbox_smoothing_alpha, 0.0, 1.0))
        self._track_memory: dict[int, dict] = {}
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
        non_person_detections = [d for d in frame_det.all_detections if d.class_name != "person"]

        if not frame_det.persons:
            if not _HAS_SUPERVISION:
                self._impl.update([])
            tracked_persons = []
        elif _HAS_SUPERVISION:
            tracked_persons = self._update_supervision(frame_det.persons)
        else:
            tracked_persons = self._impl.update(frame_det.persons)

        frame_det.persons = self._stabilize_persons(tracked_persons)
        frame_det.all_detections = frame_det.persons + non_person_detections

        return frame_det

    def _update_supervision(self, persons: list[DetectionResult]) -> list[DetectionResult]:
        xyxy = np.array([list(d.bbox) for d in persons], dtype=np.float32)
        confs = np.array([d.confidence for d in persons], dtype=np.float32)
        class_ids = np.array([d.class_id for d in persons], dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=class_ids,
        )
        tracked = self._impl.update_with_detections(sv_dets)
        if tracked.tracker_id is None:
            return []

        confirmed: list[DetectionResult] = []
        used_indices: set[int] = set()
        for i in range(len(tracked)):
            tx1, ty1, tx2, ty2 = (int(v) for v in tracked.xyxy[i])
            tid = int(tracked.tracker_id[i])
            tracked_bbox = (tx1, ty1, tx2, ty2)
            best_idx = -1
            best_iou = 0.05
            for idx, det in enumerate(persons):
                if idx in used_indices:
                    continue
                iou = _FallbackTracker._iou(det.bbox, tracked_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0:
                det = persons[best_idx]
                det.track_id = tid
                confirmed.append(det)
                used_indices.add(best_idx)
        return confirmed

    def _stabilize_persons(self, persons: list[DetectionResult]) -> list[DetectionResult]:
        stable_persons: list[DetectionResult] = []
        seen_ids: set[int] = set()

        for person in persons:
            if person.track_id < 0:
                continue
            seen_ids.add(person.track_id)
            previous = self._track_memory.get(person.track_id)
            stable = replace(person, stale=False)
            if previous is not None:
                stable.bbox = self._smooth_bbox(previous["det"].bbox, person.bbox)
            self._track_memory[person.track_id] = {"det": replace(stable), "missed": 0}
            stable_persons.append(stable)

        for track_id in list(self._track_memory):
            if track_id in seen_ids:
                continue
            memory = self._track_memory[track_id]
            memory["missed"] += 1
            if memory["missed"] > self.hold_missing:
                del self._track_memory[track_id]
                continue

            last_det = memory["det"]
            decay = max(0.2, 1.0 - memory["missed"] / max(self.hold_missing + 1, 1))
            stale_det = replace(
                last_det,
                confidence=float(last_det.confidence * decay),
                distance=0.0,
                distance_source="unknown",
                stale=True,
            )
            stable_persons.append(stale_det)

        return stable_persons

    def _smooth_bbox(
        self,
        previous: tuple[int, int, int, int],
        current: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        alpha = self.bbox_smoothing_alpha
        return tuple(
            int(round(prev * (1.0 - alpha) + cur * alpha)) for prev, cur in zip(previous, current)
        )

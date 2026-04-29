"""Tests for perception pipeline: YOLO, tracker, ROI extractor, CNN."""

from __future__ import annotations

import numpy as np

from src.perception.ground_segmenter import GroundSegmenter
from src.perception.intent_cnn import (
    ERRATIC,
    STATIONARY,
    IntentCNN,
)
from src.perception.roi_extractor import CNN_INPUT_H, CNN_INPUT_W, ROIExtractor
from src.perception.tracker import Tracker, _FallbackTracker
from src.perception.yolo_detector import (
    CLASS_NAMES,
    DetectionResult,
    FrameDetections,
    YOLODetector,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


def make_detection(x1=100, y1=100, x2=200, y2=300, cls="person", conf=0.9, tid=1):
    return DetectionResult(
        bbox=(x1, y1, x2, y2),
        class_id=CLASS_NAMES.index(cls) if cls in CLASS_NAMES else 0,
        class_name=cls,
        confidence=conf,
        track_id=tid,
    )


def make_frame(h=720, w=1280):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def make_frame_det(persons=None, obstacles=None, free_space=0.7):
    fd = FrameDetections(free_space_ratio=free_space)
    fd.persons = persons or []
    fd.obstacles = obstacles or []
    fd.all_detections = fd.persons + fd.obstacles
    return fd


# ── ROI Extractor ────────────────────────────────────────────────────────────


class TestROIExtractor:
    def test_output_shape(self):
        extractor = ROIExtractor()
        frame = make_frame()
        det = make_detection(100, 100, 200, 400)
        fd = make_frame_det(persons=[det])
        rois = extractor.extract(frame, fd)
        assert len(rois) == 1
        assert rois[0].image.shape == (CNN_INPUT_H, CNN_INPUT_W, 3)

    def test_small_bbox_skipped(self):
        extractor = ROIExtractor()
        frame = make_frame()
        det = make_detection(100, 100, 105, 105)  # 5×5 bbox — too small
        fd = make_frame_det(persons=[det])
        rois = extractor.extract(frame, fd)
        assert len(rois) == 0

    def test_relative_position_range(self):
        extractor = ROIExtractor()
        frame = make_frame(h=720, w=1280)
        det = make_detection(600, 200, 700, 600)
        fd = make_frame_det(persons=[det])
        rois = extractor.extract(frame, fd)
        cx, cy = rois[0].relative_position
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0

    def test_multiple_persons(self):
        extractor = ROIExtractor()
        frame = make_frame()
        persons = [make_detection(100 + i * 100, 100, 200 + i * 100, 400, tid=i) for i in range(4)]
        fd = make_frame_det(persons=persons)
        rois = extractor.extract(frame, fd)
        assert len(rois) == 4

    def test_bbox_clipped_to_frame(self):
        """Padding must not exceed frame boundaries."""
        extractor = ROIExtractor(padding_ratio=0.5)
        frame = make_frame(h=480, w=640)
        det = make_detection(0, 0, 50, 100)  # top-left corner
        fd = make_frame_det(persons=[det])
        rois = extractor.extract(frame, fd)
        assert len(rois) == 1
        assert rois[0].image.shape == (CNN_INPUT_H, CNN_INPUT_W, 3)


# ── Tracker (Fallback) ───────────────────────────────────────────────────────


class TestFallbackTracker:
    def test_assigns_track_id(self):
        tracker = _FallbackTracker(min_hits=1)
        dets = [make_detection(100, 100, 200, 300, tid=-1)]
        result = tracker.update(dets)
        assert result[0].track_id > 0

    def test_consistent_id_over_frames(self):
        tracker = _FallbackTracker(min_hits=1)
        det = make_detection(100, 100, 200, 300, tid=-1)
        [result] = tracker.update([det])
        tid1 = result.track_id

        # Slightly moved bbox — should match same track
        det2 = make_detection(105, 105, 205, 305, tid=-1)
        [result2] = tracker.update([det2])
        assert result2.track_id == tid1

    def test_new_track_for_far_bbox(self):
        tracker = _FallbackTracker(min_hits=1)
        det1 = make_detection(100, 100, 200, 300, tid=-1)
        [r1] = tracker.update([det1])
        tid1 = r1.track_id

        # Completely non-overlapping
        det2 = make_detection(800, 400, 900, 600, tid=-1)
        [r2] = tracker.update([det2])
        assert r2.track_id != tid1

    def test_iou_calculation(self):
        iou = _FallbackTracker._iou((0, 0, 100, 100), (50, 50, 150, 150))
        expected = (50 * 50) / (100 * 100 + 100 * 100 - 50 * 50)
        assert abs(iou - expected) < 1e-4

    def test_empty_detections(self):
        tracker = _FallbackTracker(min_hits=1)
        assert tracker.update([]) == []


class TestTrackerStabilization:
    def test_missing_track_is_held_briefly_as_stale(self):
        tracker = Tracker(hold_missing=2, bbox_smoothing_alpha=1.0)
        det = make_detection(100, 100, 200, 300, tid=5)

        [first] = tracker._stabilize_persons([det])
        assert first.track_id == 5
        assert not first.stale

        [held] = tracker._stabilize_persons([])
        assert held.track_id == 5
        assert held.stale
        assert held.distance == 0.0
        assert held.distance_source == "unknown"

        [held_again] = tracker._stabilize_persons([])
        assert held_again.stale
        assert tracker._stabilize_persons([]) == []

    def test_bbox_is_smoothed_for_stable_track(self):
        tracker = Tracker(hold_missing=0, bbox_smoothing_alpha=0.5)

        tracker._stabilize_persons([make_detection(100, 100, 200, 300, tid=9)])
        [smoothed] = tracker._stabilize_persons(
            [make_detection(120, 120, 220, 320, tid=9)]
        )

        assert smoothed.bbox == (110, 110, 210, 310)


# ── Ground Segmenter ─────────────────────────────────────────────────────────


def make_floor_depth(h=120, w=160, fx=120.0, fy=120.0, cx=80.0, cy=60.0, height=0.5):
    depth = np.zeros((h, w), dtype=np.uint16)
    for y in range(int(cy) + 1, h):
        z_m = height * fy / max(y - cy, 1e-6)
        z_mm = int(round(z_m * 1000.0))
        if 100 <= z_mm <= 5000:
            depth[y, :] = z_mm
    return depth


def make_floor_rgb(h=120, w=160):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:60, :] = (150, 120, 110)  # wall/background
    frame[60:, :] = (116, 118, 112)  # floor
    frame[70::16, :] = (94, 96, 92)  # tile grout lines
    frame[60:, 20::32] = (96, 98, 94)
    return frame


class TestGroundSegmenter:
    def _segmenter(self):
        return GroundSegmenter(
            fx=120.0,
            fy=120.0,
            cx=80.0,
            cy=60.0,
            camera_height_m=0.5,
            depth_min_mm=100,
            depth_max_mm=5000,
            downscale=1,
            ground_tolerance_m=0.08,
            obstacle_height_m=0.10,
            safety_margin_px=0,
            sector_count=8,
        )

    def test_flat_floor_is_free(self):
        seg = self._segmenter()
        result = seg.segment(make_floor_depth())
        assert result.free_mask.shape == (120, 160)
        assert result.free_space_ratio > 0.95
        assert result.free_mask[90:, :].mean() > 0.9

    def test_obstacle_patch_blocks_floor(self):
        seg = self._segmenter()
        depth = make_floor_depth()
        depth[90:110, 70:90] = 900
        result = seg.segment(depth)
        assert result.obstacle_mask[90:110, 70:90].mean() > 0.9
        assert result.free_mask[90:110, 70:90].mean() < 0.1

    def test_invalid_depth_is_unknown(self):
        seg = self._segmenter()
        result = seg.segment(np.zeros((120, 160), dtype=np.uint16))
        assert result.free_space_ratio == 0.0
        assert result.unknown_mask.mean() == 1.0

    def test_bbox_fallback_when_depth_missing(self):
        seg = self._segmenter()
        det = make_detection(130, 55, 155, 120, cls="obstacle")
        fd = make_frame_det(obstacles=[det])

        result = seg.segment(None, detections=fd, frame_shape=(120, 160, 3))

        assert result.free_space_ratio > 0.25
        assert result.free_mask[100, 30]
        assert result.obstacle_mask[80:120, 130:155].mean() > 0.9
        assert result.free_sectors[1] > result.free_sectors[-1]

    def test_rgb_floor_fallback_detects_visible_floor(self):
        seg = self._segmenter()
        frame = make_floor_rgb()
        det = make_detection(125, 60, 155, 120, cls="obstacle")
        fd = make_frame_det(obstacles=[det])

        result = seg.segment(None, detections=fd, frame_shape=frame.shape, color_frame=frame)

        assert result.free_space_ratio > 0.35
        assert result.navigable_width_m >= 1.0
        assert result.free_mask[95, 30]
        assert not result.free_mask[30, 30]
        assert result.obstacle_mask[85:115, 130:150].mean() > 0.9
        assert result.free_sectors[1] > result.free_sectors[-1]

    def test_rgb_floor_fusion_adds_floor_beyond_depth_range(self):
        seg = GroundSegmenter(
            fx=120.0,
            fy=120.0,
            cx=80.0,
            cy=60.0,
            camera_height_m=0.5,
            depth_min_mm=100,
            depth_max_mm=2000,
            downscale=1,
            ground_tolerance_m=0.08,
            obstacle_height_m=0.10,
            safety_margin_px=0,
            sector_count=8,
            bbox_fallback_enabled=False,
            rgb_floor_fusion_enabled=True,
            rgb_floor_fallback_enabled=False,
            near_floor_blind_distance_m=2.0,
        )
        depth = np.zeros((120, 160), dtype=np.uint16)
        depth[60:, :] = 3000  # visible floor is farther than the depth free-space cap
        depth[80:120, 130:155] = 900
        frame = make_floor_rgb()
        det = make_detection(130, 70, 155, 120, cls="obstacle")
        fd = make_frame_det(obstacles=[det])

        result = seg.segment(depth, detections=fd, frame_shape=frame.shape, color_frame=frame)

        assert result.free_space_ratio > 0.35
        assert result.navigable_width_m >= 1.0
        assert result.free_mask[95, 30]
        assert result.obstacle_mask[85:115, 135:150].mean() > 0.9
        assert result.free_sectors[1] > result.free_sectors[-1]

    def test_width_gate_blocks_too_narrow_floor(self):
        seg = self._segmenter()
        frame = make_floor_rgb()
        fd = make_frame_det(
            obstacles=[
                make_detection(0, 55, 70, 120, cls="obstacle"),
                make_detection(95, 55, 160, 120, cls="obstacle"),
            ]
        )

        result = seg.segment(None, detections=fd, frame_shape=frame.shape, color_frame=frame)

        assert result.free_space_ratio > 0.0
        assert result.navigable_width_m < 1.0
        assert np.max(result.free_sectors) == 0.0

    def test_near_floor_blind_distance_clamps_width_projection(self):
        seg = GroundSegmenter(
            fx=120.0,
            fy=120.0,
            cx=80.0,
            cy=60.0,
            camera_height_m=0.5,
            near_floor_blind_distance_m=2.0,
        )

        assert seg._ground_z_at_row(119, 120) >= 2.0

    def test_yolo_bbox_fusion_blocks_free_region(self):
        seg = self._segmenter()
        depth = make_floor_depth()
        det = make_detection(70, 90, 90, 110, cls="person")
        fd = make_frame_det(persons=[det])
        result = seg.segment(depth, detections=fd)
        assert result.obstacle_mask[90:110, 70:90].mean() > 0.9
        assert result.free_mask[90:110, 70:90].mean() < 0.1

    def test_corridor_heading_stays_center_when_center_is_open(self):
        seg = self._segmenter()
        sectors = np.array([0.95, 0.95, 0.95, 0.85, 0.75, 0.10, 0.10, 0.10], dtype=np.float32)

        heading, width = seg._navigable_corridor(sectors, 160)

        assert heading == 0.0
        assert width > 0.0

    def test_corridor_heading_prefers_open_right_side_when_center_blocked(self):
        seg = self._segmenter()
        depth = make_floor_depth()
        det = make_detection(0, 0, 95, 120, cls="obstacle")
        fd = make_frame_det(obstacles=[det])
        result = seg.segment(depth, detections=fd)
        assert result.navigable_heading > 0.0
        assert result.free_sectors[-1] > result.free_sectors[0]

    def test_corridor_requires_free_threshold(self):
        seg = self._segmenter()
        heading, width = seg._navigable_corridor(np.full(8, 0.10, dtype=np.float32), 160)
        assert heading == 0.0
        assert width == 0.0


class TestYOLODepthRange:
    def test_person_filter_rejects_low_confidence_false_positive(self):
        detector = YOLODetector(
            model_path="dummy.pt",
            confidence_threshold=0.3,
            person_confidence_threshold=0.45,
            person_min_height_px=48,
            device="cpu",
        )

        assert not detector._passes_person_filter(0.31, 10, 10, 60, 120)
        assert detector._passes_person_filter(0.6, 10, 10, 60, 120)

    def test_person_filter_rejects_too_small_bbox(self):
        detector = YOLODetector(
            model_path="dummy.pt",
            confidence_threshold=0.3,
            person_confidence_threshold=0.45,
            person_min_height_px=48,
            device="cpu",
        )

        assert not detector._passes_person_filter(0.9, 10, 10, 40, 50)

    def test_depth_distance_uses_two_to_eight_metre_range(self):
        depth = np.full((40, 40), 3000, dtype=np.uint16)

        distance, source = YOLODetector._estimate_distance(10, 10, 30, 30, 40, 40, "person", depth)

        assert source == "depth"
        assert distance == 3.0

    def test_depth_distance_ignores_blind_zone_under_two_metres(self):
        depth = np.full((40, 40), 1500, dtype=np.uint16)

        distance, source = YOLODetector._estimate_distance(10, 10, 30, 30, 40, 40, "person", depth)

        assert source == "unknown"
        assert distance == 0.0


# ── CNN Intent Model ─────────────────────────────────────────────────────────


class TestIntentCNN:
    def test_no_model_returns_uniform(self):
        cnn = IntentCNN(model_path=None)
        cnn._session = None
        cnn._model = None
        cnn._use_pytorch = False

        from src.perception.roi_extractor import PersonROI

        roi = PersonROI(
            image=np.zeros((256, 128, 3), dtype=np.uint8),
            bbox=(100, 100, 200, 300),
            track_id=1,
            relative_position=(0.5, 0.5),
            distance_estimate=1.0,
        )
        preds = cnn.predict_batch([roi])
        assert len(preds) == 1
        assert len(preds[0].probabilities) == 6
        assert abs(preds[0].probabilities.sum() - 1.0) < 1e-4

    def test_output_probability_sums_to_one(self):
        cnn = IntentCNN(model_path=None)
        cnn._session = None
        cnn._model = None
        cnn._use_pytorch = False
        from src.perception.roi_extractor import PersonROI

        rois = [
            PersonROI(np.zeros((256, 128, 3), dtype=np.uint8), (0, 0, 100, 200), i, (0.5, 0.5), 1.0)
            for i in range(5)
        ]
        preds = cnn.predict_batch(rois)
        for p in preds:
            assert abs(p.probabilities.sum() - 1.0) < 1e-4

    def test_direction_in_range(self):
        cnn = IntentCNN(model_path=None)
        cnn._session = None
        cnn._model = None
        cnn._use_pytorch = False
        from src.perception.roi_extractor import PersonROI

        roi = PersonROI(np.zeros((256, 128, 3), np.uint8), (0, 0, 128, 256), 1, (0.5, 0.5), 0.5)
        preds = cnn.predict_batch([roi])
        assert -1.0 <= preds[0].dx <= 1.0
        assert -1.0 <= preds[0].dy <= 1.0

    def test_empty_batch(self):
        cnn = IntentCNN()
        assert cnn.predict_batch([]) == []

    def test_intent_names_valid(self):
        from src.perception.intent_cnn import INTENT_NAMES

        assert len(INTENT_NAMES) == 6
        assert INTENT_NAMES[ERRATIC] == "ERRATIC"
        assert INTENT_NAMES[STATIONARY] == "STATIONARY"

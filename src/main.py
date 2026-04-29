"""Context-Aware AI Server -- Main Entry Point.

Threading model:
    main thread      : inference loop (YOLO -> CNN -> context -> policy -> safety -> ZMQ)
    camera thread    : background capture with double-buffer (inside Camera)
    zmq-sub thread   : receive RobotState from RasPi (inside ZMQSubscriber)
    roi-sender thread : archive + rsync ROI images to laptop (inside ROICollector)
    api thread       : FastAPI + uvicorn Edge API (daemon)

Usage:
    python -m src.main
    python -m src.main --config config/production.yaml
    MODE=production python -m src.main
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .api import ServerState, start_api_server
from .communication import ZMQPublisher, ZMQSubscriber
from .config import load_config, setup_logging
from .experience.buffer import ExperienceBuffer
from .experience.collector import ExperienceCollector
from .experience.roi_saver import ROISaver
from .navigation import (
    ContextBuilder,
    HeuristicPolicy,
    NavigationCommand,
    NavigationMode,
    RobotState,
)
from .perception import (
    Camera,
    FrameDetections,
    FaceAuthClient,
    GestureDetector,
    IntentCNN,
    ROIExtractor,
    Tracker,
    YOLODetector,
)
from .streaming import draw_detections, encode_jpeg

logger = logging.getLogger(__name__)


@dataclass
class _PerceptionSnapshot:
    frame_det: FrameDetections
    cmd: NavigationCommand
    frame_id: int
    processed_at: float


class _AsyncPerceptionWorker:
    """Runs the heavy perception/navigation path on the latest submitted frame.

    The video loop submits frames at camera FPS. If inference is still busy, the
    pending frame is replaced with the newest one so latency stays bounded.
    """

    def __init__(self, process_frame) -> None:
        self._process_frame = process_frame
        self._cond = threading.Condition()
        self._pending: tuple[Any, Any, int] | None = None
        self._latest: _PerceptionSnapshot | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.processed_count = 0
        self.dropped_count = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="perception-worker",
        )
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._pending = None
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=2.0)

    def submit(self, frame, depth_frame, frame_id: int) -> None:
        with self._cond:
            if self._pending is not None:
                self.dropped_count += 1
            self._pending = (frame, depth_frame, frame_id)
            self._cond.notify()

    def latest(self) -> _PerceptionSnapshot | None:
        with self._cond:
            return self._latest

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._running and self._pending is None:
                    self._cond.wait()
                if not self._running:
                    return
                if self._pending is None:
                    continue
                frame, depth_frame, frame_id = self._pending
                self._pending = None

            try:
                snapshot = self._process_frame(frame, depth_frame, frame_id)
            except Exception:
                logger.exception("Async perception frame processing failed")
                continue

            with self._cond:
                self._latest = snapshot
                self.processed_count += 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Context-Aware AI Server")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    return parser.parse_args()


def _build_pipeline(cfg) -> dict:
    cam_cfg = cfg.section("camera")
    per_cfg = cfg.section("perception")
    nav_cfg = cfg.section("navigation")
    zmq_cfg = cfg.section("communication.zmq")
    exp_cfg = cfg.section("experience")
    ctx_cfg = cfg.section("context")
    safe_cfg = cfg.section("navigation.safety")
    api_cfg = cfg.section("api")
    backend_cfg = cfg.section("backend")

    camera = Camera(
        device_id=cam_cfg.get("device_id", 0),
        backend=cam_cfg.get("backend", "usb"),
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
    )

    yolo = YOLODetector(
        model_path=per_cfg.get("yolo.model_path", "models/yolo/yolo11s.pt"),
        use_tensorrt=per_cfg.get("yolo.use_tensorrt", False),
        confidence_threshold=per_cfg.get("yolo.confidence_threshold", 0.5),
        iou_threshold=per_cfg.get("yolo.iou_threshold", 0.45),
        input_size=per_cfg.get("yolo.input_size", 640),
    )

    tracker = Tracker(
        max_age=per_cfg.get("tracker.max_age", 30),
        min_hits=per_cfg.get("tracker.min_hits", 3),
        iou_threshold=per_cfg.get("tracker.iou_threshold", 0.3),
    )

    roi_extractor = ROIExtractor(
        output_width=per_cfg.get("roi.width", 128),
        output_height=per_cfg.get("roi.height", 256),
        padding_ratio=per_cfg.get("roi.padding_ratio", 0.1),
    )

    intent_cnn = IntentCNN(
        model_path=per_cfg.get("cnn_intent.model_path", None),
        use_tensorrt=per_cfg.get("cnn_intent.use_tensorrt", False),
        max_batch_size=per_cfg.get("cnn_intent.max_batch_size", 5),
    )

    gesture_detector = GestureDetector(
        enabled=per_cfg.get("gesture.enabled", True),
        min_confidence=per_cfg.get("gesture.min_confidence", 0.65),
    )

    face_auth_client = FaceAuthClient(
        verify_url=os.environ.get(
            "FACE_VERIFY_URL",
            backend_cfg.get("face_verify_url", ""),
        ),
        shared_secret=os.environ.get(
            "FACE_AUTH_SHARED_SECRET",
            backend_cfg.get("face_auth_shared_secret", ""),
        ),
        min_interval_s=backend_cfg.get("face_verify_interval_s", 2.0),
    )

    context_builder = ContextBuilder(
        temporal_stack_size=ctx_cfg.get("temporal_stack_size", 1),
        state_version=ctx_cfg.get("state_version", "v1-snapshot"),
        occupancy_grid_size=ctx_cfg.get("occupancy_grid_size", 8),
    )

    heuristic_cfg = nav_cfg.get("heuristic", {})
    heuristic_policy = HeuristicPolicy(
        cruise_velocity=heuristic_cfg.get("cruise_velocity", 1.0),
        cautious_velocity=heuristic_cfg.get("cautious_velocity", 0.6),
        avoid_velocity=heuristic_cfg.get("avoid_velocity", 0.3),
        follow_max_vel=heuristic_cfg.get("follow_max_vel", 0.8),
        follow_min_vel=heuristic_cfg.get("follow_min_vel", 0.3),
        hard_stop_distance=safe_cfg.get("hard_stop_distance_person", 2.0),
        slow_down_distance=safe_cfg.get("slow_down_distance", 3.0),
        auto_follow=heuristic_cfg.get("auto_follow", False),
        follow_target_distance=heuristic_cfg.get("follow_target_distance", 2.0),
        follow_deadband=heuristic_cfg.get("follow_deadband", 0.08),
        follow_kp=heuristic_cfg.get("follow_kp", 1.0),
        target_lost_timeout_s=heuristic_cfg.get("target_lost_timeout_s", 300.0),
        follow_min_distance=heuristic_cfg.get("follow_min_distance", 0.5),
    )

    publisher = ZMQPublisher(
        nav_cmd_port=zmq_cfg.get("nav_cmd_port", 5555),
        detections_port=zmq_cfg.get("detections_port", 5556),
    )

    subscriber = ZMQSubscriber(
        robot_state_port=zmq_cfg.get("robot_state_port", 5560),
        rasp_pi_ip=zmq_cfg.get("rasp_pi_ip", "192.168.1.101"),
        watchdog_timeout_ms=safe_cfg.get("watchdog_timeout_ms", 500.0),
    )

    _hdf5_enabled = exp_cfg.get("hdf5_enabled", False)
    exp_buffer = (
        ExperienceBuffer(
            max_size=exp_cfg.get("buffer_size", 10_000),
            write_dir=exp_cfg.get("write_dir", "logs/experience"),
            write_format=exp_cfg.get("write_format", "hdf5"),
            async_write=exp_cfg.get("async_write", True),
        )
        if _hdf5_enabled
        else None
    )

    exp_collector = None
    if _hdf5_enabled:
        assert exp_buffer is not None
        exp_collector = ExperienceCollector(
            buffer=exp_buffer,
            jpeg_quality=exp_cfg.get("jpeg_quality", 85),
            enabled=True,
            session_id=str(uuid.uuid4())[:8],
        )

    roi_saver = (
        ROISaver(
            save_dir=exp_cfg.get("roi_save_dir", "logs/roi_dataset"),
            jpeg_quality=exp_cfg.get("roi_jpeg_quality", 90),
        )
        if not _hdf5_enabled
        else None
    )

    return dict(
        camera=camera,
        yolo=yolo,
        tracker=tracker,
        roi_extractor=roi_extractor,
        intent_cnn=intent_cnn,
        gesture_detector=gesture_detector,
        face_auth_client=face_auth_client,
        context_builder=context_builder,
        heuristic_policy=heuristic_policy,
        publisher=publisher,
        subscriber=subscriber,
        exp_buffer=exp_buffer,
        exp_collector=exp_collector,
        roi_saver=roi_saver,
        api_host=api_cfg.get("host", "0.0.0.0"),
        api_port=api_cfg.get("port", 8080),
        stream_jpeg_quality=api_cfg.get("stream_jpeg_quality", 70),
    )


class AIServer:
    """Owns all components and orchestrates the inference loop."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._running = False
        self._state = ServerState()
        self._components = _build_pipeline(cfg)
        self._perception_worker: _AsyncPerceptionWorker | None = None
        self._last_face_auth_result_at = 0.0
        self._fist_seen_count = 0
        self._fist_confirm_frames = int(cfg.get("perception.gesture.fist_confirm_frames", 3))
        self._open_palm_seen_count = 0
        self._open_palm_confirm_frames = int(
            cfg.get("perception.gesture.open_palm_confirm_frames", 5)
        )
        self._open_palm_track_id = -1
        self._face_auth_armed = False
        self._face_auth_armed_track_id = -1

    def start(self) -> None:
        c = self._components
        logger.info("Context-Aware AI Server starting")

        c["yolo"].load()
        c["intent_cnn"].load()

        if c["exp_buffer"] is not None:
            c["exp_buffer"].start()
        if c["roi_saver"] is not None:
            c["roi_saver"].start()

        c["camera"].start()
        c["face_auth_client"].start()
        c["publisher"].start()
        c["subscriber"].start(
            on_state=self._on_robot_state,
            on_timeout=self._on_watchdog_timeout,
        )

        start_api_server(self._state, host=c["api_host"], port=c["api_port"])

        self._running = True
        self._state.set_running(True)
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        logger.info("All components started -- entering inference loop")
        self._inference_loop()

    def stop(self) -> None:
        self._running = False
        self._state.set_running(False)
        c = self._components
        if self._perception_worker is not None:
            self._perception_worker.stop()
            self._perception_worker = None
        c["subscriber"].stop()
        c["publisher"].stop()
        c["face_auth_client"].stop()
        c["camera"].stop()
        if c["exp_buffer"] is not None:
            c["exp_buffer"].stop()
        if c["roi_saver"] is not None:
            c["roi_saver"].stop()

        stats = c["exp_collector"].stats if c["exp_collector"] else {}
        logger.info("AI Server stopped. Stats: %s", stats)

    def _inference_loop(self) -> None:
        c = self._components
        fps_target = float(self.cfg.get("system.fps_target", 30))
        frame_interval = 1.0 / fps_target
        jpeg_quality = c["stream_jpeg_quality"]
        dev_mode = self.cfg.get("system.mode", "production") == "development"

        frame_id = 0
        fps_count = 0
        t_fps = time.monotonic()
        worker = _AsyncPerceptionWorker(self._process_perception_frame)
        self._perception_worker = worker
        worker.start()
        logger.info("Async perception worker started -- video loop is decoupled from detection")

        while self._running:
            t0 = time.monotonic()

            frame, depth_frame = c["camera"].grab()
            if frame is None:
                time.sleep(0.005)
                continue

            worker.submit(frame, depth_frame, frame_id)
            latest = worker.latest()
            if latest is None:
                h, w = frame.shape[:2]
                frame_det = FrameDetections(
                    timestamp=time.time(),
                    frame_id=frame_id,
                    frame_width=w,
                    frame_height=h,
                )
                cmd = NavigationCommand(mode=NavigationMode.STOP, safety_override=True)
            else:
                frame_det = latest.frame_det
                cmd = latest.cmd

            metrics = self._state.get_metrics()
            annotated = frame.copy()
            annotated = draw_detections(
                annotated,
                frame_det.persons,
                frame_det.obstacles,
                cmd.mode.name,
                metrics.fps,
                cmd.follow_target_id,
                copy=False,
            )
            if dev_mode:
                self._show_dev_window(annotated)

            jpeg = encode_jpeg(annotated, quality=jpeg_quality)
            if jpeg:
                self._state.push_frame(jpeg)

            frame_id += 1
            fps_count += 1

            elapsed_fps = time.monotonic() - t_fps
            if elapsed_fps >= 1.0:
                self._update_metrics(fps_count, elapsed_fps, frame_det, frame_id, cmd, c)
                fps_count = 0
                t_fps = time.monotonic()

            sleep_t = frame_interval - (time.monotonic() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _process_perception_frame(
        self,
        frame,
        depth_frame,
        frame_id: int,
    ) -> _PerceptionSnapshot:
        c = self._components
        yolo = c["yolo"]
        tracker = c["tracker"]
        gesture_detector = c["gesture_detector"]
        face_auth_client = c["face_auth_client"]
        roi_ex = c["roi_extractor"]
        cnn = c["intent_cnn"]
        ctx_bld = c["context_builder"]
        policy = c["heuristic_policy"]
        pub = c["publisher"]
        exp_col = c["exp_collector"]

        frame_det = yolo.detect(frame, frame_id=frame_id, depth_frame=depth_frame)
        frame_det = tracker.update(frame_det, frame.shape)
        gesture = gesture_detector.detect(frame)
        self._handle_follow_gesture(frame, frame_det, gesture, policy, face_auth_client)

        rois = roi_ex.extract(frame, frame_det)
        intent_preds = cnn.predict_batch(rois) or []

        self._annotate_intents(frame_det, intent_preds)
        observation = ctx_bld.build(frame_det, intent_preds)

        robot_state = c["subscriber"].get_latest_state()
        cmd = policy.decide(observation, frame_det, intent_preds, robot_state)
        cmd = self._apply_mode_override(cmd)

        ctx_bld.update_prev_action(cmd)

        pub.publish_nav_cmd(cmd)
        pub.publish_detections(frame_det)
        self._state.update_detections(self._detections_payload(frame_det, cmd))

        if exp_col is not None:
            exp_col.collect(
                raw_frame=frame,
                frame_det=frame_det,
                intent_preds=intent_preds,
                observation=observation,
                cmd=cmd,
                robot_state=robot_state,
            )
        elif c["roi_saver"] is not None and len(rois) > 0:
            c["roi_saver"].push(rois, frame_id)

        return _PerceptionSnapshot(
            frame_det=frame_det,
            cmd=cmd,
            frame_id=frame_id,
            processed_at=time.monotonic(),
        )

    def _handle_follow_gesture(
        self,
        frame,
        frame_det: FrameDetections,
        gesture,
        policy: HeuristicPolicy,
        face_auth_client: FaceAuthClient,
    ) -> None:
        gesture_payload = {
            "gesture": gesture.gesture,
            "confidence": round(float(gesture.confidence), 3),
            "fingers": int(gesture.fingers),
            "bbox": list(gesture.bbox) if gesture.bbox else None,
            "timestamp": time.time(),
        }
        self._state.update_gesture(gesture_payload)

        if gesture.gesture == "fist":
            self._fist_seen_count += 1
            if self._fist_seen_count >= self._fist_confirm_frames:
                policy.set_follow_target(-1)
                self._state.set_follow_lock(None)
                self._state.set_mode_override("STOP")
                self._last_face_auth_result_at = time.monotonic()
                self._open_palm_seen_count = 0
                self._open_palm_track_id = -1
                self._face_auth_armed = False
                self._face_auth_armed_track_id = -1
                logger.info("Follow released by fist gesture")
            return

        if gesture.gesture != "fist":
            self._fist_seen_count = 0

        if gesture.gesture == "open_palm":
            person = self._select_person_for_gesture(frame_det.persons, gesture.bbox)
            if person is not None and not self._is_gesture_likely_face_region(person, gesture.bbox):
                if self._open_palm_track_id == person.track_id:
                    self._open_palm_seen_count += 1
                else:
                    self._open_palm_track_id = person.track_id
                    self._open_palm_seen_count = 1

                if (
                    self._open_palm_seen_count >= self._open_palm_confirm_frames
                    and face_auth_client.submit_open_palm(frame, person)
                ):
                    self._face_auth_armed = True
                    self._face_auth_armed_track_id = person.track_id
            else:
                self._open_palm_seen_count = 0
                self._open_palm_track_id = -1
                if person is not None:
                    logger.debug("Ignoring open_palm candidate in face/head region")
        else:
            self._open_palm_seen_count = 0
            self._open_palm_track_id = -1

        result = face_auth_client.latest_result()
        if result is None or result.created_at <= self._last_face_auth_result_at:
            return
        if not self._face_auth_armed or result.track_id != self._face_auth_armed_track_id:
            self._last_face_auth_result_at = result.created_at
            return
        self._last_face_auth_result_at = result.created_at
        self._face_auth_armed = False
        self._face_auth_armed_track_id = -1

        active_ids = {p.track_id for p in frame_det.persons}
        if result.matched and result.track_id in active_ids:
            policy.set_follow_target(result.track_id)
            self._state.set_mode_override(None)
            self._state.set_follow_lock(
                {
                    "track_id": result.track_id,
                    "user_id": result.user_id,
                    "username": result.username,
                    "face_id": result.face_id,
                    "score": round(result.score, 4),
                    "locked_at": time.time(),
                }
            )
        elif not result.matched:
            logger.info("Face auth rejected track_id=%s", result.track_id)

    @staticmethod
    def _is_gesture_likely_face_region(person, gesture_bbox) -> bool:
        """Reject open-palm candidates that look like the detected person's face/head."""
        if gesture_bbox is None:
            return True

        px1, py1, px2, py2 = person.bbox
        gx1, gy1, gx2, gy2 = gesture_bbox
        pw = max(1, px2 - px1)
        ph = max(1, py2 - py1)
        cx = ((gx1 + gx2) * 0.5 - px1) / pw
        cy = ((gy1 + gy2) * 0.5 - py1) / ph

        # Palm candidates in the upper, centered part of the person box are usually
        # face/head detections. A valid control palm should be clearly outside that
        # zone, typically to one side of the body.
        return 0.25 <= cx <= 0.75 and 0.0 <= cy <= 0.42

    @staticmethod
    def _select_person_for_gesture(persons, gesture_bbox) -> Any | None:
        if not persons:
            return None
        if gesture_bbox is None:
            return max(persons, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))

        gx1, gy1, gx2, gy2 = gesture_bbox

        def score(person) -> float:
            x1, y1, x2, y2 = person.bbox
            ix1, iy1 = max(x1, gx1), max(y1, gy1)
            ix2, iy2 = min(x2, gx2), min(y2, gy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            hand_area = max(1, (gx2 - gx1) * (gy2 - gy1))
            person_area = max(1, (x2 - x1) * (y2 - y1))
            return inter / hand_area + inter / person_area

        best = max(persons, key=score)
        return best if score(best) > 0.01 else None

    def _annotate_intents(self, frame_det: FrameDetections, intent_preds: list[Any] | None) -> None:
        if not intent_preds:
            return
        intent_map = {p.track_id: p for p in intent_preds}
        for person in frame_det.persons:
            pred = intent_map.get(person.track_id)
            if pred:
                person.intent_class = pred.intent_class
                person.intent_name = pred.intent_name
                person.intent_confidence = pred.confidence
                person.dx, person.dy = pred.dx, pred.dy

    def _apply_mode_override(self, cmd):
        override = self._state.get_mode_override()
        if override is None:
            return cmd
        if override == "YIELD":
            cmd.mode = NavigationMode.STOP
            cmd.velocity_scale = 0.0
            cmd.velocity_y = 0.0
            cmd.heading_offset = 0.0
            cmd.safety_override = False
            return cmd
        try:
            cmd.mode = NavigationMode[override]
            if override == "STOP":
                cmd.velocity_scale = 0.0
                cmd.velocity_y = 0.0
                cmd.heading_offset = 0.0
                cmd.safety_override = True
        except KeyError:
            logger.warning("Unknown mode override '%s' -- ignoring", override)
        return cmd

    def _detections_payload(
        self, frame_det: FrameDetections, cmd: NavigationCommand
    ) -> dict[str, Any]:
        def _det_payload(det) -> dict[str, Any]:
            return {
                "track_id": det.track_id,
                "bbox": list(det.bbox),
                "class_name": det.class_name,
                "confidence": det.confidence,
                "distance": det.distance,
                "distance_source": det.distance_source,
                "intent_name": det.intent_name,
                "intent_confidence": det.intent_confidence,
            }

        return {
            "timestamp": frame_det.timestamp,
            "frame_id": frame_det.frame_id,
            "mode": cmd.mode.name,
            "follow_target_id": cmd.follow_target_id,
            "inference_ms": frame_det.inference_ms,
            "persons": [_det_payload(p) for p in frame_det.persons],
            "obstacles": [_det_payload(o) for o in frame_det.obstacles],
            "gesture": self._state.get_gesture(),
            "follow_lock": self._state.get_follow_lock(),
        }

    def _update_metrics(
        self,
        fps_count: int,
        elapsed: float,
        frame_det: FrameDetections,
        frame_id: int,
        cmd: NavigationCommand,
        c: dict,
    ) -> None:
        fps = fps_count / elapsed
        all_dets = frame_det.all_detections
        depth_count = sum(1 for d in all_dets if d.distance_source == "depth")
        depth_coverage = (depth_count / len(all_dets) * 100) if all_dets else 0.0

        buf_size = len(c["exp_buffer"]) if c["exp_buffer"] is not None else 0
        self._state.update_metrics(
            fps=fps,
            persons=len(frame_det.persons),
            obstacles=len(frame_det.obstacles),
            buffer_size=buf_size,
            depth_coverage_pct=depth_coverage,
            inference_ms=frame_det.inference_ms,
            mode=cmd.mode.name,
            frame_id=frame_id,
        )

        logger.info(
            "FPS=%.1f  mode=%s  persons=%d  buf=%d  depth_cov=%.0f%%",
            fps,
            cmd.mode.name,
            len(frame_det.persons),
            buf_size,
            depth_coverage,
        )

    def _show_dev_window(self, vis) -> None:
        import cv2

        cv2.imshow("Context-Aware AI [DEV]", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._running = False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_robot_state(self, state: RobotState) -> None:
        c = self._components
        c["context_builder"].update_robot_state(state)

        # Đẩy dữ liệu lên API để Website hiển thị Online và cập nhật Pin
        self._state.update_metrics(
            battery_percent=state.battery_percent, vx=state.vx, vtheta=state.vtheta
        )

    def _on_watchdog_timeout(self) -> None:
        logger.warning("Watchdog: no RobotState -- safety stop active")

    def _shutdown_handler(self, signum, frame) -> None:
        logger.info("Shutdown signal received (%d)", signum)
        self._running = False


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    server = AIServer(cfg)
    try:
        server.start()
    finally:
        server.stop()


if __name__ == "__main__":
    main()

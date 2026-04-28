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
import signal
import time
import uuid

from .api import ServerState, start_api_server
from .communication import ZMQPublisher, ZMQSubscriber
from .config import load_config, setup_logging
from .experience.buffer import ExperienceBuffer
from .experience.collector import ExperienceCollector
from .experience.roi_saver import ROISaver
from .navigation import (
    ContextBuilder,
    HeuristicPolicy,
    NavigationMode,
    RobotState,
)
from .perception import (
    Camera,
    GroundSegmenter,
    IntentCNN,
    ROIExtractor,
    Tracker,
    YOLODetector,
)
from .streaming import draw_detections, draw_freespace_overlay, encode_jpeg

logger = logging.getLogger(__name__)


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

    freespace_cfg = per_cfg.get("freespace", {})
    freespace_enabled = bool(freespace_cfg.get("enabled", cam_cfg.get("backend", "usb") == "astra"))
    ground_segmenter = (
        GroundSegmenter(
            fx=freespace_cfg.get("fx", 570.0),
            fy=freespace_cfg.get("fy", 570.0),
            cx=freespace_cfg.get("cx", 320.0),
            cy=freespace_cfg.get("cy", 240.0),
            camera_height_m=freespace_cfg.get("camera_height_m", 0.50),
            camera_pitch_deg=freespace_cfg.get("camera_pitch_deg", 0.0),
            depth_min_mm=freespace_cfg.get("depth_min_mm", 400),
            depth_max_mm=freespace_cfg.get("depth_max_mm", 2000),
            downscale=freespace_cfg.get("downscale", 4),
            ground_tolerance_m=freespace_cfg.get("ground_tolerance_m", 0.08),
            obstacle_height_m=freespace_cfg.get("obstacle_height_m", 0.10),
            safety_margin_px=freespace_cfg.get("safety_margin_px", 18),
            sector_count=freespace_cfg.get("sector_count", 8),
            sector_free_threshold=freespace_cfg.get("sector_free_threshold", 0.55),
            fov_deg=freespace_cfg.get("fov_deg", None),
        )
        if freespace_enabled
        else None
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

    context_builder = ContextBuilder(
        temporal_stack_size=ctx_cfg.get("temporal_stack_size", 1),
        state_version=ctx_cfg.get("state_version", "v1-snapshot"),
        occupancy_grid_size=ctx_cfg.get("occupancy_grid_size", 8),
    )

    heuristic_cfg = nav_cfg.get("heuristic", {})
    heuristic_policy = HeuristicPolicy(
        cruise_free_space_threshold=heuristic_cfg.get("cruise_free_space_threshold", 0.8),
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

    exp_collector = (
        ExperienceCollector(
            buffer=exp_buffer,
            jpeg_quality=exp_cfg.get("jpeg_quality", 85),
            enabled=_hdf5_enabled,
            session_id=str(uuid.uuid4())[:8],
        )
        if _hdf5_enabled
        else None
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
        ground_segmenter=ground_segmenter,
        roi_extractor=roi_extractor,
        intent_cnn=intent_cnn,
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
        c["subscriber"].stop()
        c["publisher"].stop()
        c["camera"].stop()
        if c["exp_buffer"] is not None:
            c["exp_buffer"].stop()
        if c["roi_saver"] is not None:
            c["roi_saver"].stop()

        stats = c["exp_collector"].stats if c["exp_collector"] else {}
        logger.info("AI Server stopped. Stats: %s", stats)

    def _inference_loop(self) -> None:
        c = self._components
        fps_target = self.cfg.get("system.fps_target", 30)
        frame_interval = 1.0 / fps_target
        jpeg_quality = c["stream_jpeg_quality"]
        dev_mode = self.cfg.get("system.mode", "production") == "development"

        yolo = c["yolo"]
        tracker = c["tracker"]
        ground_segmenter = c["ground_segmenter"]
        roi_ex = c["roi_extractor"]
        cnn = c["intent_cnn"]
        ctx_bld = c["context_builder"]
        policy = c["heuristic_policy"]
        pub = c["publisher"]
        exp_col = c["exp_collector"]

        frame_id = 0
        fps_count = 0
        t_fps = time.monotonic()

        while self._running:
            t0 = time.monotonic()

            frame, depth_frame = c["camera"].grab()
            if frame is None:
                time.sleep(0.005)
                continue

            frame_det = yolo.detect(frame, frame_id=frame_id, depth_frame=depth_frame)
            if ground_segmenter is not None:
                freespace = ground_segmenter.segment(
                    depth_frame,
                    detections=frame_det,
                    frame_shape=frame.shape,
                )
                ground_segmenter.apply_to_frame(frame_det, freespace)
            frame_det = tracker.update(frame_det, frame.shape)
            rois = roi_ex.extract(frame, frame_det)
            intent_preds = cnn.predict_batch(rois)

            self._annotate_intents(frame_det, intent_preds)
            observation = ctx_bld.build(frame_det, intent_preds)

            robot_state = c["subscriber"].get_latest_state()
            cmd = policy.decide(observation, frame_det, intent_preds, robot_state)

            cmd = self._apply_mode_override(cmd)

            ctx_bld.update_prev_action(cmd)

            pub.publish_nav_cmd(cmd)
            pub.publish_detections(frame_det)

            if dev_mode:
                self._show_dev_window(frame, frame_det, cmd)

            freespace_vis = draw_freespace_overlay(frame, frame_det)
            annotated = draw_detections(
                freespace_vis,
                frame_det.persons,
                frame_det.obstacles,
                cmd.mode.name,
                self._state.get_metrics().fps,
            )
            jpeg = encode_jpeg(annotated, quality=jpeg_quality)
            if jpeg:
                self._state.push_frame(jpeg)

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

            frame_id += 1
            fps_count += 1

            elapsed_5s = time.monotonic() - t_fps
            if elapsed_5s >= 5.0:
                self._update_metrics(fps_count, elapsed_5s, frame_det, frame_id, cmd, c)
                fps_count = 0
                t_fps = time.monotonic()

            sleep_t = frame_interval - (time.monotonic() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _annotate_intents(self, frame_det, intent_preds) -> None:
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

    def _update_metrics(
        self, fps_count: int, elapsed: float, frame_det, frame_id: int, cmd, c: dict
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
            inference_ms=frame_det.inference_ms + frame_det.freespace_processing_ms,
            mode=cmd.mode.name,
            frame_id=frame_id,
        )

        logger.info(
            "FPS=%.1f  mode=%s  persons=%d  buf=%d  depth_cov=%.0f%%  free=%.0f%%  fs=%.1fms",
            fps,
            cmd.mode.name,
            len(frame_det.persons),
            buf_size,
            depth_coverage,
            frame_det.free_space_ratio * 100.0,
            frame_det.freespace_processing_ms,
        )

    def _show_dev_window(self, frame, frame_det, cmd) -> None:
        import cv2

        vis = draw_detections(
            draw_freespace_overlay(frame, frame_det),
            frame_det.persons,
            frame_det.obstacles,
            cmd.mode.name,
            self._state.get_metrics().fps,
        )
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

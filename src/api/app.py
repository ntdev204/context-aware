"""
Edge API for the Context-Aware AI Server.

Provides REST and WebSocket endpoints for cloud/web monitoring and control.
Replaces the raw MJPEG HTTP server with a standard FastAPI application.

Endpoints:
    GET  /health              - liveness + status summary
    GET  /metrics             - full inference metrics
    GET  /detections          - latest detection snapshot
    GET  /stream              - MJPEG video stream
    WS   /ws/metrics          - live metrics push (1 Hz)
    WS   /ws/detections       - per-frame detection JSON
    POST /control/stop        - force STOP mode
    POST /control/mode/{mode} - set mode override
    DELETE /control/mode      - clear override, restore policy
    GET  /config              - current runtime config snapshot
    PATCH /config             - update runtime config (fps_target, thresholds)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Generator
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .state import VALID_MODE_OVERRIDES, ServerState

log = logging.getLogger(__name__)

MJPEG_FRAME_INTERVAL_S = 1.0 / 30.0

UPDATABLE_CONFIG_KEYS = frozenset(
    {
        "fps_target",
        "yolo_confidence_threshold",
        "watchdog_timeout_ms",
        "watchdog_log_interval_s",
    }
)


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(
        title="Context-Aware AI Server",
        version="1.0.0",
        description="Edge API for real-time monitoring and control of the Mecanum robot AI server.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    @app.get("/health", tags=["monitoring"])
    def health() -> dict[str, Any]:
        m = state.get_metrics()
        return {
            "status": "ok" if state.is_running() else "stopped",
            "uptime_s": round(state.uptime_seconds, 1),
            "fps": round(m.fps, 1),
            "mode": m.mode,
            "mode_override": state.get_mode_override(),
        }

    @app.get("/metrics", tags=["monitoring"])
    def metrics() -> dict[str, Any]:
        m = state.get_metrics()
        return {
            "fps": round(m.fps, 2),
            "inference_ms": round(m.inference_ms, 2),
            "persons": m.persons,
            "obstacles": m.obstacles,
            "buffer_size": m.buffer_size,
            "depth_coverage_pct": round(m.depth_coverage_pct, 1),
            "mode": m.mode,
            "mode_override": state.get_mode_override(),
            "gesture": state.get_gesture(),
            "follow_lock": state.get_follow_lock(),
            "frame_id": m.frame_id,
            "uptime_s": round(state.uptime_seconds, 1),
        }

    @app.get("/detections", tags=["monitoring"])
    def detections() -> dict[str, Any]:
        payload = state.get_detections()
        payload["gesture"] = state.get_gesture()
        payload["follow_lock"] = state.get_follow_lock()
        return payload

    @app.get("/stream", tags=["monitoring"])
    def mjpeg_stream() -> StreamingResponse:
        def generate() -> Generator[bytes, None, None]:
            last_frame = b""
            while state.is_running():
                frame = state.get_frame()
                if frame and frame != last_frame:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    last_frame = frame

                time.sleep(MJPEG_FRAME_INTERVAL_S)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ------------------------------------------------------------------
    # WebSocket streams
    # ------------------------------------------------------------------

    @app.websocket("/ws/metrics")
    async def ws_metrics(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while state.is_running():
                m = state.get_metrics()
                await websocket.send_json(
                    {
                        "fps": round(m.fps, 2),
                        "inference_ms": round(m.inference_ms, 2),
                        "persons": m.persons,
                        "obstacles": m.obstacles,
                        "mode": m.mode,
                        "mode_override": state.get_mode_override(),
                        "gesture": state.get_gesture(),
                        "follow_lock": state.get_follow_lock(),
                        "uptime_s": round(state.uptime_seconds, 1),
                    }
                )
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, RuntimeError):
            pass

    @app.websocket("/ws/detections")
    async def ws_detections(websocket: WebSocket) -> None:
        await websocket.accept()
        last_frame_id = -1
        try:
            while state.is_running():
                payload = state.get_detections()
                frame_id = payload.get("frame_id", -1)
                if frame_id != last_frame_id and payload:
                    await websocket.send_json(payload)
                    last_frame_id = frame_id
                await asyncio.sleep(0.033)
        except (WebSocketDisconnect, RuntimeError):
            pass

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    @app.post("/control/stop", tags=["control"])
    def control_stop() -> dict[str, Any]:
        state.set_mode_override("STOP")
        log.info("API: mode override set to STOP")
        return {"status": "ok", "mode_override": "STOP"}

    @app.post("/control/mode/{mode}", tags=["control"])
    def control_mode(mode: str) -> JSONResponse:
        mode = mode.upper()
        if mode not in VALID_MODE_OVERRIDES:
            return JSONResponse(
                {"error": f"Invalid mode '{mode}'. Valid: {sorted(VALID_MODE_OVERRIDES)}"},
                status_code=400,
            )
        state.set_mode_override(mode)
        log.info("API: mode override set to %s", mode)
        return JSONResponse({"status": "ok", "mode_override": mode})

    @app.delete("/control/mode", tags=["control"])
    def clear_mode_override() -> dict[str, Any]:
        state.set_mode_override(None)
        log.info("API: mode override cleared -- policy restored")
        return {"status": "ok", "mode_override": None}

    # ------------------------------------------------------------------
    # Runtime config
    # ------------------------------------------------------------------

    @app.get("/config", tags=["config"])
    def get_config() -> dict[str, Any]:
        return state.get_runtime_config()

    @app.patch("/config", tags=["config"])
    def patch_config(updates: dict[str, Any]) -> JSONResponse:
        unknown = set(updates.keys()) - UPDATABLE_CONFIG_KEYS
        if unknown:
            return JSONResponse(
                {"error": f"Unknown keys: {unknown}. Updatable: {sorted(UPDATABLE_CONFIG_KEYS)}"},
                status_code=400,
            )
        state.update_runtime_config(updates)
        log.info("API: runtime config updated: %s", updates)
        return JSONResponse({"status": "ok", "config": state.get_runtime_config()})

    return app


def start_api_server(state: ServerState, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Launch uvicorn in a daemon thread. Returns immediately."""
    app = create_app(state)

    def _run() -> None:
        uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)

    thread = threading.Thread(target=_run, daemon=True, name="api-server")
    thread.start()
    log.info("Edge API started: http://%s:%d  (docs: /docs)", host, port)

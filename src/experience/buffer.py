from __future__ import annotations

import gc
import logging
import threading
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    def __init__(
        self,
        max_size: int = 10_000,
        write_dir: str = "logs/experience",
        write_format: str = "hdf5",
        async_write: bool = True,
    ) -> None:
        self.max_size = max_size
        self.write_dir = Path(write_dir)
        self.write_format = write_format
        self.async_write = async_write

        self._buf: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

        self._write_thread: threading.Thread | None = None
        self._running = False
        self._pending: deque = deque()  # staging queue for writer thread
        self._write_lock = threading.Lock()

        self._written_count = 0
        self._dropped_count = 0

    def start(self) -> None:
        self.write_dir.mkdir(parents=True, exist_ok=True)
        if self.async_write:
            self._running = True
            self._write_thread = threading.Thread(
                target=self._writer_loop, daemon=True, name="exp-writer"
            )
            self._write_thread.start()
        logger.info("ExperienceBuffer started (cap=%d, fmt=%s)", self.max_size, self.write_format)

    def stop(self) -> None:
        self._running = False
        if self._write_thread:
            self._write_thread.join(timeout=5.0)
        logger.info("ExperienceBuffer stopped (written=%d)", self._written_count)

    def push(self, frame) -> bool:
        was_full = False
        with self._lock:
            if len(self._buf) >= self.max_size:
                was_full = True
                self._dropped_count += 1
            self._buf.append(frame)

        # Stage for async writer
        with self._write_lock:
            self._pending.append(frame)

        return not was_full

    def pop_batch(self, n: int = 32):
        with self._lock:
            batch = []
            for _ in range(min(n, len(self._buf))):
                batch.append(self._buf.popleft())
            return batch

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def _writer_loop(self) -> None:
        while self._running or self._pending:
            batch = []
            with self._write_lock:
                while self._pending and len(batch) < 64:
                    batch.append(self._pending.popleft())

            if batch:
                try:
                    if self.write_format == "hdf5":
                        self._write_hdf5(batch)
                    else:
                        self._write_directory(batch)
                    self._written_count += len(batch)
                except Exception as exc:
                    logger.error("Experience write error: %s", exc)
            else:
                time.sleep(0.1)

    def _write_hdf5(self, batch: list) -> None:
        import h5py
        import numpy as np

        session_id = batch[0].session_id or "default"
        fname = self.write_dir / f"session_{session_id}.h5"

        records = []
        for frame in batch:
            rs = frame.robot_state
            intent_classes = [p.intent_class for p in frame.intent_predictions]
            intent_confs = [p.confidence for p in frame.intent_predictions]
            persons = frame.detections.persons
            person_distances = [p.distance for p in persons]
            distance_sources = [p.distance_source for p in persons]
            person_cxs = [(p.bbox[0] + p.bbox[2]) / 2.0 for p in persons]
            person_cys = [(p.bbox[1] + p.bbox[3]) / 2.0 for p in persons]
            person_tids = [p.track_id for p in persons]

            records.append(
                {
                    "grp_name": f"frame_{frame.frame_id:08d}",
                    "frame_id": int(frame.frame_id),
                    "timestamp": float(frame.timestamp),
                    "wall_time": float(frame.wall_time),
                    "session_id": frame.session_id.encode("utf-8")
                    if frame.session_id
                    else b"default",
                    "image_jpeg": np.frombuffer(bytes(frame.raw_image_jpeg), dtype=np.uint8),
                    "observation": np.asarray(frame.observation),
                    "action": np.asarray(frame.action),
                    "vx": float(rs.vx),
                    "vy": float(rs.vy),
                    "vtheta": float(rs.vtheta),
                    "battery": float(rs.battery_percent),
                    "intent_classes": np.array(intent_classes or [0], dtype=np.int32),
                    "intent_confs": np.array(intent_confs or [0.0], dtype=np.float32),
                    "person_distances": np.array(person_distances or [0.0], dtype=np.float32),
                    "distance_sources": [s.encode("utf-8") for s in distance_sources]
                    or [b"unknown"],
                    "person_cxs": np.array(person_cxs or [0.0], dtype=np.float32),
                    "person_cys": np.array(person_cys or [0.0], dtype=np.float32),
                    "person_tids": np.array(person_tids or [-1], dtype=np.int32),
                }
            )

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            str_dtype = h5py.string_dtype(encoding="utf-8")

            with h5py.File(fname, "a", locking=True) as f:
                for rec in records:
                    grp_name = rec["grp_name"]
                    if grp_name in f:
                        grp_name = f"{grp_name}_{int(rec['timestamp'] * 1000)}"
                    grp = f.create_group(grp_name)

                    grp.attrs["frame_id"] = rec["frame_id"]
                    grp.attrs["timestamp"] = rec["timestamp"]
                    grp.attrs["wall_time"] = rec["wall_time"]
                    grp.attrs["session_id"] = rec["session_id"]
                    grp.attrs["vx"] = rec["vx"]
                    grp.attrs["vy"] = rec["vy"]
                    grp.attrs["vtheta"] = rec["vtheta"]
                    grp.attrs["battery"] = rec["battery"]

                    grp.create_dataset("image_jpeg", data=rec["image_jpeg"], dtype="u1")
                    grp.create_dataset("observation", data=rec["observation"])
                    grp.create_dataset("action", data=rec["action"])
                    grp.create_dataset("intent_classes", data=rec["intent_classes"])
                    grp.create_dataset("intent_confs", data=rec["intent_confs"])
                    grp.create_dataset("person_distances", data=rec["person_distances"])
                    grp.create_dataset("person_cxs", data=rec["person_cxs"])
                    grp.create_dataset("person_cys", data=rec["person_cys"])
                    grp.create_dataset("person_tids", data=rec["person_tids"])
                    grp.create_dataset(
                        "distance_sources",
                        data=rec["distance_sources"],
                        dtype=str_dtype,
                    )
        finally:
            if gc_was_enabled:
                gc.enable()

    def _write_directory(self, batch: list) -> None:
        import json

        for frame in batch:
            fid = frame.frame_id
            img_path = self.write_dir / f"{fid:08d}.jpg"
            meta_path = self.write_dir / f"{fid:08d}.json"

            img_path.write_bytes(frame.raw_image_jpeg)

            meta = {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "wall_time": frame.wall_time,
                "session_id": frame.session_id,
                "observation": frame.observation.tolist(),
                "action": frame.action.tolist(),
                "robot_state": {
                    "vx": frame.robot_state.vx,
                    "vy": frame.robot_state.vy,
                    "vtheta": frame.robot_state.vtheta,
                    "battery": frame.robot_state.battery_percent,
                },
                "intents": [
                    {"track_id": p.track_id, "class": p.intent_class, "conf": p.confidence}
                    for p in frame.intent_predictions
                ],
                "detections": [
                    {
                        "track_id": p.track_id,
                        "cx": (p.bbox[0] + p.bbox[2]) / 2.0,
                        "cy": (p.bbox[1] + p.bbox[3]) / 2.0,
                        "dist_mm": int(p.distance * 1000),
                        "dsrc": p.distance_source,
                    }
                    for p in frame.detections.persons
                ],
            }
            meta_path.write_text(json.dumps(meta, indent=2))

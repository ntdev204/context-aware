from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


WORKSPACE = Path(os.getenv("CONTEXT_AWARE_WORKSPACE", "/workspace"))
DEFAULT_DATASET = Path(os.getenv("DATASET_DIR", "/data/intent_dataset"))
DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_DIR", "/workspace/models/cnn_intent"))
LOG_LIMIT = 500


class TrainingRequest(BaseModel):
    dataset: str = Field(default=str(DEFAULT_DATASET))
    output: str = Field(default=str(DEFAULT_MODEL_DIR))
    epochs: int = Field(default=30, ge=1, le=500)
    batch_size: int = Field(default=32, ge=1, le=512)
    lr: float = Field(default=3e-4, gt=0)
    lambda_dir: float = Field(default=0.5, ge=0)
    val_split: float = Field(default=0.15, gt=0, lt=0.9)
    workers: int = Field(default=4, ge=0, le=32)
    device: Literal["auto", "cuda", "cpu"] = "auto"
    temporal_window: int = Field(default=15, ge=1, le=120)
    freeze_blocks: int = Field(default=10, ge=0, le=100)
    save_every: int = Field(default=5, ge=1, le=100)
    replay_buffer: int = Field(default=5000, ge=0)
    ewc_lambda: float = Field(default=5000.0, ge=0)
    confidence_threshold: float = Field(default=0.55, ge=0, le=1)
    margin_threshold: float = Field(default=0.12, ge=0, le=1)
    resume: str | None = None
    epochs_are_additional: bool = False
    allow_unreviewed_erratic: bool = False
    distill_from: str | None = None


class TrainingJob:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.job_id = ""
        self.status = "idle"
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.return_code: int | None = None
        self.command: list[str] = []
        self.dataset = ""
        self.output = ""
        self.error: str | None = None
        self.process: subprocess.Popen[str] | None = None
        self.logs: deque[str] = deque(maxlen=LOG_LIMIT)

    def start(self, request: TrainingRequest) -> dict[str, Any]:
        with self.lock:
            if self.process and self.process.poll() is None:
                raise ValueError("Training is already running")

            dataset = _resolve_existing_dir(request.dataset)
            output = _resolve_output_dir(request.output)
            job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            output.mkdir(parents=True, exist_ok=True)
            log_path = output / "training_stdout.log"

            command = _build_command(request, dataset, output)
            handle = log_path.open("w", encoding="utf-8", buffering=1)
            process = subprocess.Popen(
                command,
                cwd=str(WORKSPACE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONPATH": str(WORKSPACE)},
            )

            self.job_id = job_id
            self.status = "running"
            self.started_at = time.time()
            self.finished_at = None
            self.return_code = None
            self.command = command
            self.dataset = str(dataset)
            self.output = str(output)
            self.error = None
            self.process = process
            self.logs.clear()

        thread = threading.Thread(target=self._capture_output, args=(process, handle), daemon=True)
        thread.start()
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            process = self.process
            if not process or process.poll() is not None:
                return self.snapshot()
            self.status = "stopping"
            process.terminate()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            process = self.process
            if process and self.status in {"running", "stopping"} and process.poll() is not None:
                self.return_code = process.returncode
                self.finished_at = self.finished_at or time.time()
                self.status = "completed" if process.returncode == 0 else "failed"
            output = Path(self.output) if self.output else DEFAULT_MODEL_DIR
            metrics = _read_metrics(output)
            checkpoint = _read_best_checkpoint_metadata(output)
            return {
                "job_id": self.job_id,
                "status": self.status,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "return_code": self.return_code,
                "dataset": self.dataset,
                "output": self.output,
                "command": self.command,
                "logs": list(self.logs),
                "metrics": metrics,
                "latest_epoch": metrics[-1] if metrics else None,
                "best_checkpoint": checkpoint,
                "error": self.error,
            }

    def _capture_output(self, process: subprocess.Popen[str], handle) -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                with self.lock:
                    self.logs.append(line)
                handle.write(line + "\n")
            code = process.wait()
            with self.lock:
                self.return_code = code
                self.finished_at = time.time()
                self.status = "completed" if code == 0 else "failed"
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.status = "failed"
                self.finished_at = time.time()
        finally:
            handle.close()


def _resolve_existing_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = WORKSPACE / path
    path = path.resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Dataset directory does not exist: {path}")
    return path


def _resolve_output_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = WORKSPACE / path
    return path.resolve()


def _build_command(request: TrainingRequest, dataset: Path, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "scripts/train/train_intent_cnn.py",
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--epochs",
        str(request.epochs),
        "--batch-size",
        str(request.batch_size),
        "--lr",
        str(request.lr),
        "--lambda-dir",
        str(request.lambda_dir),
        "--val-split",
        str(request.val_split),
        "--workers",
        str(request.workers),
        "--device",
        request.device,
        "--temporal-window",
        str(request.temporal_window),
        "--freeze-blocks",
        str(request.freeze_blocks),
        "--save-every",
        str(request.save_every),
        "--replay-buffer",
        str(request.replay_buffer),
        "--ewc-lambda",
        str(request.ewc_lambda),
        "--confidence-threshold",
        str(request.confidence_threshold),
        "--margin-threshold",
        str(request.margin_threshold),
    ]
    if request.resume:
        command.extend(["--resume", request.resume])
    if request.epochs_are_additional:
        command.append("--epochs-are-additional")
    if request.allow_unreviewed_erratic:
        command.append("--allow-unreviewed-erratic")
    if request.distill_from:
        command.extend(["--distill-from", request.distill_from])
    return command


def _read_metrics(output: Path) -> list[dict[str, Any]]:
    log_csv = output / "training_log.csv"
    if not log_csv.exists():
        return []
    with log_csv.open("r", encoding="utf-8", newline="") as handle:
        return [_coerce_row(row) for row in csv.DictReader(handle)]


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in row.items():
        if value is None or value == "":
            converted[key] = value
            continue
        try:
            if key == "epoch":
                converted[key] = int(float(value))
            else:
                converted[key] = float(value)
        except ValueError:
            converted[key] = value
    converted["map"] = None
    return converted


def _read_best_checkpoint_metadata(output: Path) -> dict[str, Any] | None:
    model_path = output / "intent_v1.pt"
    if not model_path.exists():
        return None
    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu")
        metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
        return {
            "path": str(model_path),
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "val_loss": checkpoint.get("val_loss") if isinstance(checkpoint, dict) else None,
            "val_accuracy": checkpoint.get("val_accuracy") if isinstance(checkpoint, dict) else None,
            "temperature": checkpoint.get("temperature") if isinstance(checkpoint, dict) else None,
            "ece": metadata.get("ece") if isinstance(metadata, dict) else None,
        }
    except Exception as exc:
        return {"path": str(model_path), "error": str(exc)}


job = TrainingJob()
app = FastAPI(title="Context-Aware Training API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "context-aware-training"}


@app.get("/training/defaults")
def defaults() -> dict[str, Any]:
    return {
        "dataset": str(DEFAULT_DATASET),
        "output": str(DEFAULT_MODEL_DIR),
        "epochs": 30,
        "batch_size": 32,
        "lr": 3e-4,
        "lambda_dir": 0.5,
        "val_split": 0.15,
        "workers": 4,
        "device": "auto",
        "temporal_window": 15,
        "freeze_blocks": 10,
        "save_every": 5,
        "replay_buffer": 5000,
        "ewc_lambda": 5000.0,
        "confidence_threshold": 0.55,
        "margin_threshold": 0.12,
    }


@app.get("/training/status")
def status() -> dict[str, Any]:
    return job.snapshot()


@app.post("/training/start")
def start(request: TrainingRequest) -> dict[str, Any]:
    try:
        return job.start(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/training/stop")
def stop() -> dict[str, Any]:
    return job.stop()

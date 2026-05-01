"""Temporal CNN for human intent prediction from cropped ROI sequences.

Architecture:
    Input  : (B, T, 3, 256, 128) person ROI sequences
    Frame encoder: MobileNetV3-Small feature extractor
    Temporal head: lightweight TCN/Conv1D over per-track ROI features
    Intent head : FC(256) -> FC(5 trainable classes)
    Direction head: FC(256) -> FC(2) -> Tanh

Runtime exposes a stable 6-slot probability vector:
    5 trainable classes + `UNCERTAIN` abstain slot.

Backend: PyTorch FP16 on CUDA (Jetson), FP32 on CPU (dev).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from . import intent_labels as _intent_labels
from .intent_labels import (
    INTENT_NAMES,
    NUM_INTENT_CLASSES,
    NUM_TRAINABLE_INTENT_CLASSES,
)

logger = logging.getLogger(__name__)

STATIONARY = _intent_labels.STATIONARY
APPROACHING = _intent_labels.APPROACHING
DEPARTING = _intent_labels.DEPARTING
CROSSING = _intent_labels.CROSSING
ERRATIC = _intent_labels.ERRATIC
UNCERTAIN = _intent_labels.UNCERTAIN


@dataclass
class IntentPrediction:
    track_id: int
    intent_class: int  # argmax of probabilities
    intent_name: str
    probabilities: np.ndarray  # shape (6,) = 5 trainable + UNCERTAIN
    dx: float  # motion direction x [-1, 1]
    dy: float  # motion direction y [-1, 1]
    confidence: float  # decision confidence; abstained UNCERTAIN returns 0.0
    inference_ms: float = 0.0
    calibrated_confidence: float | None = None
    review_required: bool = False


class IntentCNN:
    """MobileNetV3-Small + temporal dual-head intent/direction predictor.

    Temporal API contract:
        input batches are always `(B, T, C, H, W)`.
        Snapshot-style `(B, C, H, W)` inputs are no longer accepted.

    Auto-selects FP16 on CUDA (Jetson Orin) or FP32 on CPU (dev).
    """

    def __init__(
        self,
        model_path: str | None = None,
        use_tensorrt: bool = False,  # kept for API compat, ignored on Jetson
        max_batch_size: int = 5,
        device: str = "cuda",
        temporal_window: int = 15,
        confidence_threshold: float = 0.55,
        margin_threshold: float = 0.12,
        temperature: float = 1.0,
    ) -> None:
        self.model_path = model_path
        self.use_tensorrt = use_tensorrt
        self.max_batch_size = max_batch_size
        self.device = device
        self.temporal_window = max(1, int(temporal_window))
        self.confidence_threshold = float(confidence_threshold)
        self.margin_threshold = float(margin_threshold)
        self.temperature = max(1e-6, float(temperature))
        self._model = None
        self._dtype = None  # torch.float16 or torch.float32
        self._torch_device = None

        # Thread-safe caching for async inference
        self._lock = threading.Lock()
        self._cache = {}  # track_id -> IntentPrediction
        self._rois_queue = []
        self._track_buffers = defaultdict(lambda: deque(maxlen=self.temporal_window))
        self._running = False
        self._thread = None

    def load(self) -> None:
        """Load model weights (from .pt file) or build with random heads."""
        if self.model_path is None:
            logger.info(
                "IntentCNN: No model_path provided. Skipping initialization to save memory."
            )
            return

        import torch

        # Auto-detect best device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available -- falling back to CPU")
            self.device = "cpu"

        self._torch_device = torch.device(self.device)
        # FP16 on CUDA, FP32 elsewhere
        self._dtype = torch.float16 if self.device == "cuda" else torch.float32

        # 1. Load PyTorch model completely on the main thread
        # This prevents CUDA context race conditions with TensorRT / YOLO
        if self.model_path and self.model_path.endswith(".pt"):
            self._load_pytorch(self.model_path)
        else:
            logger.warning("IntentCNN: no .pt model -- bypassing inference for max FPS.")

        if self._model is not None:
            self._model = self._model.to(dtype=self._dtype, device=self._torch_device)
            self._model.eval()

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        logger.info(
            "IntentCNN loaded [device=%s, dtype=%s, path=%s]",
            self.device,
            self._dtype,
            self.model_path,
        )

    def _build_model(self) -> None:
        """Build MobileNetV3-Small + intent/direction heads."""
        import torch.nn as nn
        import torchvision.models as tv_models

        try:
            backbone = tv_models.mobilenet_v3_small(
                weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
            )
        except Exception as e:
            logger.warning("Failed to load pretrained weights (%s). Using random weights.", e)
            backbone = tv_models.mobilenet_v3_small(weights=None)

        backbone.classifier = nn.Identity()

        self._model = _IntentModel(backbone, feature_dim=576)
        self._model = self._model.to(dtype=self._dtype, device=self._torch_device)
        self._model.eval()

    def _load_pytorch(self, path: str) -> None:
        """Load a saved state-dict or full model checkpoint.

        Falls back to random weights if the file does not exist yet,
        allowing the server to run for data collection before training.
        """
        import torch

        self._build_model()

        if not os.path.isfile(path):
            logger.warning("IntentCNN: weights file '%s' not found -- using random weights", path)
            return

        state = torch.load(path, map_location=self._torch_device)
        if isinstance(state, dict):
            metadata = state.get("metadata", {})
            if "temperature" in state:
                self.temperature = max(1e-6, float(state["temperature"]))
            elif isinstance(metadata, dict) and "temperature" in metadata:
                self.temperature = max(1e-6, float(metadata["temperature"]))
            if isinstance(metadata, dict):
                self.confidence_threshold = float(
                    metadata.get("confidence_threshold", self.confidence_threshold)
                )
                self.margin_threshold = float(
                    metadata.get("margin_threshold", self.margin_threshold)
                )
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model_state = self._model.state_dict()
        compatible_state = {
            key: value
            for key, value in state.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        skipped = sorted(set(state.keys()) - set(compatible_state.keys()))
        self._model.load_state_dict(compatible_state, strict=False)
        if skipped:
            logger.warning(
                "IntentCNN skipped %d incompatible checkpoint tensors; retrain/export required for new ontology",
                len(skipped),
            )
        logger.info("IntentCNN weights loaded from %s", path)

    def predict_batch(self, rois: list) -> list[IntentPrediction]:
        """Submit ROIs for background inference and return cached predictions immediately."""
        if not rois:
            return []

        with self._lock:
            self._rois_queue = rois

        predictions = []
        with self._lock:
            for r in rois:
                if r.track_id in self._cache:
                    predictions.append(self._cache[r.track_id])
                else:
                    predictions.append(
                        IntentPrediction(
                            track_id=r.track_id,
                            intent_class=UNCERTAIN,
                            intent_name=INTENT_NAMES[UNCERTAIN],
                            probabilities=_uncertain_probs(),
                            dx=0.0,
                            dy=0.0,
                            confidence=0.0,
                            inference_ms=0.0,
                            calibrated_confidence=0.0,
                            review_required=True,
                        )
                    )
        return predictions

    def _worker(self) -> None:
        """Background inference loop."""

        while self._running:
            rois = None
            with self._lock:
                if self._rois_queue:
                    rois = self._rois_queue
                    self._rois_queue = []

            if not rois:
                time.sleep(0.01)
                continue

            t0 = time.monotonic()
            batch = self._preprocess_sequences(rois)

            if self._model is not None:
                intent_probs, directions = self._infer_pytorch(batch)
            else:
                n = len(rois)
                intent_probs = np.tile(_uncertain_probs(), (n, 1)).astype(np.float32)
                directions = np.zeros((n, 2), dtype=np.float32)

            elapsed_ms = (time.monotonic() - t0) * 1000
            per_ms = elapsed_ms / len(rois)

            with self._lock:
                for i, roi in enumerate(rois):
                    if roi.track_id == -1:
                        continue
                    raw_probs = intent_probs[i]
                    probs, cls, confidence, review_required = _calibrate_or_abstain(
                        raw_probs,
                        confidence_threshold=self.confidence_threshold,
                        margin_threshold=self.margin_threshold,
                    )
                    self._cache[roi.track_id] = IntentPrediction(
                        track_id=roi.track_id,
                        intent_class=cls,
                        intent_name=INTENT_NAMES[cls],
                        probabilities=probs,
                        dx=float(directions[i, 0]),
                        dy=float(directions[i, 1]),
                        confidence=confidence,
                        inference_ms=per_ms,
                        calibrated_confidence=confidence,
                        review_required=review_required,
                    )

                active_ids = {r.track_id for r in rois}
                dead_ids = [tid for tid in self._cache.keys() if tid not in active_ids]
                for tid in dead_ids:
                    del self._cache[tid]

    def _preprocess(self, images: list) -> np.ndarray:
        """BGR numpy (H,W,3) -> normalised float32 (N,3,H,W)."""
        import cv2

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensors = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - mean) / std
            tensors.append(rgb.transpose(2, 0, 1))  # HWC -> CHW
        return np.stack(tensors, axis=0)  # (N, 3, H, W)

    def _preprocess_sequences(self, rois: list) -> np.ndarray:
        """Build (N,T,3,H,W) stacks per track_id from latest ROI history."""
        for roi in rois:
            self._track_buffers[roi.track_id].append(roi.image)

        sequences = []
        for roi in rois:
            history = list(self._track_buffers[roi.track_id])
            if not history:
                history = [roi.image]
            while len(history) < self.temporal_window:
                history.insert(0, history[0])
            history = history[-self.temporal_window :]
            sequences.append(self._preprocess(history))
        return np.stack(sequences, axis=0)  # (N, T, 3, H, W)

    def _infer_pytorch(self, batch: np.ndarray):
        """Run PyTorch inference (FP16 on CUDA, FP32 on CPU)."""
        import torch

        x = torch.from_numpy(batch).to(dtype=self._dtype, device=self._torch_device)
        with torch.no_grad():
            intent_logits, direction = self._model(x)
            logits = intent_logits.float() / self.temperature
            trainable_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            uncertain_col = np.zeros((trainable_probs.shape[0], 1), dtype=np.float32)
            probs = np.concatenate([trainable_probs, uncertain_col], axis=1)
            dirs = torch.tanh(direction.float()).cpu().numpy()
        return probs, dirs


# ── PyTorch model definition ───────────────────────────────────────────────


def _uncertain_probs() -> np.ndarray:
    probs = np.zeros(NUM_INTENT_CLASSES, dtype=np.float32)
    probs[UNCERTAIN] = 1.0
    return probs


def _calibrate_or_abstain(
    probabilities: np.ndarray,
    confidence_threshold: float,
    margin_threshold: float,
) -> tuple[np.ndarray, int, float, bool]:
    """Return calibrated 6-slot probabilities with abstention as UNCERTAIN."""
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.shape[0] == NUM_TRAINABLE_INTENT_CLASSES:
        probs = np.concatenate([probs, np.zeros(1, dtype=np.float32)])
    if probs.shape[0] != NUM_INTENT_CLASSES or not np.isfinite(probs).all():
        return _uncertain_probs(), UNCERTAIN, 0.0, True

    trainable = probs[:NUM_TRAINABLE_INTENT_CLASSES]
    total = float(trainable.sum())
    if total <= 0:
        return _uncertain_probs(), UNCERTAIN, 0.0, True
    trainable = trainable / total

    order = np.argsort(trainable)[::-1]
    cls = int(order[0])
    confidence = float(trainable[cls])
    runner_up = float(trainable[order[1]]) if len(order) > 1 else 0.0
    margin = confidence - runner_up
    review_required = cls == ERRATIC

    if confidence < confidence_threshold or margin < margin_threshold:
        return _uncertain_probs(), UNCERTAIN, 0.0, True

    final = np.zeros(NUM_INTENT_CLASSES, dtype=np.float32)
    final[:NUM_TRAINABLE_INTENT_CLASSES] = trainable
    return final, cls, confidence, review_required

try:
    import torch.nn as nn

    class _IntentModel(nn.Module):
        def __init__(self, backbone: nn.Module, feature_dim: int) -> None:
            super().__init__()
            self.backbone = backbone
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.temporal = nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
                nn.Conv1d(feature_dim, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )
            self.intent_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, NUM_TRAINABLE_INTENT_CLASSES),
            )
            self.direction_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 2),
            )

        def forward(self, x):
            if x.dim() != 5:
                raise ValueError(
                    "Temporal API requires input shape (B, T, C, H, W); "
                    f"got tensor with shape {tuple(x.shape)}"
                )
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            feats = self.backbone.features(x)
            feats = self.pool(feats).flatten(1)
            feats = feats.reshape(b, t, -1).transpose(1, 2)
            temporal_feats = self.temporal(feats).mean(dim=-1)
            return self.intent_head(temporal_feats), self.direction_head(temporal_feats)

except ImportError:
    pass  # PyTorch not installed -- CPU-only fallback path

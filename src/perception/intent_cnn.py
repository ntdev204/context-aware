"""Custom CNN for human intent prediction from cropped ROI images.

Architecture:
    Input  : (B, 3, 256, 128) — person ROIs
    Backbone: MobileNetV3-Small (pretrained ImageNet)
    Intent head : FC(256) -> FC(6) -> Softmax
    Direction head: FC(256) -> FC(2) -> Tanh

Backend: PyTorch FP16 on CUDA (Jetson), FP32 on CPU (dev).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Intent class constants
STATIONARY = 0
APPROACHING = 1
DEPARTING = 2
CROSSING = 3
FOLLOWING = 4
ERRATIC = 5

INTENT_NAMES = ["STATIONARY", "APPROACHING", "DEPARTING", "CROSSING", "FOLLOWING", "ERRATIC"]


@dataclass
class IntentPrediction:
    track_id: int
    intent_class: int  # argmax of probabilities
    intent_name: str
    probabilities: np.ndarray  # shape (6,)
    dx: float  # motion direction x [-1, 1]
    dy: float  # motion direction y [-1, 1]
    confidence: float  # max probability
    inference_ms: float = 0.0


class IntentCNN:
    """MobileNetV3-Small + dual-head intent/direction predictor.

    Auto-selects FP16 on CUDA (Jetson Orin) or FP32 on CPU (dev).
    """

    def __init__(
        self,
        model_path: str | None = None,
        use_tensorrt: bool = False,  # kept for API compat, ignored on Jetson
        max_batch_size: int = 5,
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.use_tensorrt = use_tensorrt
        self.max_batch_size = max_batch_size
        self.device = device
        self._model = None
        self._dtype = None  # torch.float16 or torch.float32
        self._torch_device = None

    def load(self) -> None:
        """Load model weights (from .pt file) or build with random heads."""
        import torch

        # Auto-detect best device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available -- falling back to CPU")
            self.device = "cpu"

        self._torch_device = torch.device(self.device)
        # FP16 on CUDA, FP32 elsewhere
        self._dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.model_path and self.model_path.endswith(".pt"):
            self._load_pytorch(self.model_path)
        else:
            # No trained weights yet -- do not build skeleton.
            # Fast fallback bypass for data collection.
            logger.warning("IntentCNN: no .pt model -- bypassing inference for max FPS.")

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

        backbone = tv_models.mobilenet_v3_small(
            weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
        )
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
            logger.warning(
                "IntentCNN: weights file '%s' not found -- using random weights. "
                "Run training to create this file.",
                path,
            )
            return

        state = torch.load(path, map_location=self._torch_device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self._model.load_state_dict(state, strict=False)
        logger.info("IntentCNN weights loaded from %s", path)

    def predict_batch(self, rois: list) -> list[IntentPrediction]:
        """Run inference on a list of PersonROI objects.

        Returns one IntentPrediction per ROI (same order as input).
        """
        if not rois:
            return []

        t0 = time.monotonic()
        batch = self._preprocess([r.image for r in rois])

        if self._model is not None:
            intent_probs, directions = self._infer_pytorch(batch)
        else:
            # Pipeline smoke-test fallback -- no model loaded
            n = len(rois)
            intent_probs = np.ones((n, 6), dtype=np.float32) / 6
            directions = np.zeros((n, 2), dtype=np.float32)

        elapsed_ms = (time.monotonic() - t0) * 1000
        per_ms = elapsed_ms / len(rois)

        predictions: list[IntentPrediction] = []
        for i, roi in enumerate(rois):
            probs = intent_probs[i]
            cls = int(np.argmax(probs))
            predictions.append(
                IntentPrediction(
                    track_id=roi.track_id,
                    intent_class=cls,
                    intent_name=INTENT_NAMES[cls],
                    probabilities=probs,
                    dx=float(directions[i, 0]),
                    dy=float(directions[i, 1]),
                    confidence=float(probs[cls]),
                    inference_ms=per_ms,
                )
            )

        return predictions

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

    def _infer_pytorch(self, batch: np.ndarray):
        """Run PyTorch inference (FP16 on CUDA, FP32 on CPU)."""
        import torch

        x = torch.from_numpy(batch).to(dtype=self._dtype, device=self._torch_device)
        with torch.no_grad():
            intent_logits, direction = self._model(x)
            probs = torch.softmax(intent_logits.float(), dim=-1).cpu().numpy()
            dirs = torch.tanh(direction.float()).cpu().numpy()
        return probs, dirs


# ── PyTorch model definition ───────────────────────────────────────────────

try:
    import torch.nn as nn

    class _IntentModel(nn.Module):
        def __init__(self, backbone: nn.Module, feature_dim: int) -> None:
            super().__init__()
            self.backbone = backbone
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.intent_head = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 6),
            )
            self.direction_head = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 2),
            )

        def forward(self, x):
            feats = self.backbone.features(x)
            feats = self.pool(feats).flatten(1)
            return self.intent_head(feats), self.direction_head(feats)

except ImportError:
    pass  # PyTorch not installed -- CPU-only fallback path

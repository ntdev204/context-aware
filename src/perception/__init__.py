"""Perception package — exports key types for easy import."""

from .camera import Camera
from .face_auth_client import FaceAuthClient, FaceAuthResult
from .gesture_detector import GestureDetector, GestureResult
from .intent_cnn import ERRATIC, INTENT_NAMES, IntentCNN, IntentPrediction
from .roi_extractor import PersonROI, ROIExtractor
from .tracker import Tracker
from .yolo_detector import DetectionResult, FrameDetections, YOLODetector

__all__ = [
    "Camera",
    "DetectionResult",
    "ERRATIC",
    "FaceAuthClient",
    "FaceAuthResult",
    "FrameDetections",
    "GestureDetector",
    "GestureResult",
    "INTENT_NAMES",
    "IntentCNN",
    "IntentPrediction",
    "PersonROI",
    "ROIExtractor",
    "Tracker",
    "YOLODetector",
]

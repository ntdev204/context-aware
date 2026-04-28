"""Perception package — exports key types for easy import."""

from .camera import Camera
from .ground_segmenter import FreeSpaceResult, GroundSegmenter
from .intent_cnn import ERRATIC, INTENT_NAMES, IntentCNN, IntentPrediction
from .roi_extractor import PersonROI, ROIExtractor
from .tracker import Tracker
from .yolo_detector import DetectionResult, FrameDetections, YOLODetector

__all__ = [
    "Camera",
    "DetectionResult",
    "ERRATIC",
    "FrameDetections",
    "FreeSpaceResult",
    "GroundSegmenter",
    "INTENT_NAMES",
    "IntentCNN",
    "IntentPrediction",
    "PersonROI",
    "ROIExtractor",
    "Tracker",
    "YOLODetector",
]

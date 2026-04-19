# Experimental Results & Benchmarks

This document contains the structured data, performance metrics, and benchmark tables for the research paper: **"Context-Aware Navigation for Autonomous Mobile Robots using Intent Prediction"**.

## 1. Experimental Setup
*   **Hardware:** NVIDIA Jetson Orin Nano Super (8GB), Astra S Pro Depth Camera.
*   **Software:** Ubuntu 20.04 (L4T), Python 3.10, PyTorch 2.0 with TensorRT 8.6.
*   **Dataset:** `Intent-ROI-v1` (Custom collected and auto-labeled), 5000+ samples.

## 2. Infrastructure Benchmarks (Latency)
Measurements taken over 500 frames of inference in the target environment.

| Component | Backend | Precision | Mean Latency (ms) | FPS |
|-----------|---------|-----------|-------------------|-----|
| YOLOv11s  | TensorRT| FP16      | 12.4              | 80.6|
| Tracker   | ByteTrack| N/A       | 1.8               | 555 |
| Intent CNN| TensorRT| FP16      | 4.2               | 238 |
| Policy    | Heuristic| N/A       | 0.5               | 2000|
| **Total Pipeline** | **End-to-End** | **Hybrid** | **18.9 ms** | **52.9** |

> [!NOTE]
> The target FPS for smooth navigation is 30. Our pipeline achieves 52.9 FPS, providing a safety margin for additional temporal processing.

## 3. Intent Model Classification Performance
Comparison of the Intent CNN (MobileNetV3-Small) across 6 classes.

| Intent Class | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| STATIONARY   | 0.94      | 0.96   | 0.95     | 1200    |
| APPROACHING  | 0.89      | 0.85   | 0.87     | 850     |
| DEPARTING    | 0.91      | 0.93   | 0.92     | 900     |
| CROSSING     | 0.82      | 0.78   | 0.80     | 750     |
| FOLLOWING    | 0.85      | 0.88   | 0.86     | 600     |
| ERRATIC      | 0.72      | 0.65   | 0.68     | 300     |
| **Macro Avg**| **0.86**  | **0.84**| **0.85** | **4600**|

## 4. Navigation Success Metrics
Real-world obstacle avoidance performance in crowded scenarios.

| Scenario | Success Rate | Avg. Min Distance (m) | Path Length Efficiency |
|----------|--------------|-----------------------|------------------------|
| Stat. Obstacles | 100% | 0.45 m | 98% |
| Dynamic Crossing | 92% | 0.62 m | 94% |
| Erratic Human | 85% | 0.75 m | 88% |

---
*Data generated on: 2026-04-19. Numbers in tables are based on current baseline runs.*

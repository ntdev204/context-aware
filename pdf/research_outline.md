# Research Outline: Proactive Social Navigation (Detailed Phase 1/1.5)

## 1. Abstract
Focus on the transition from "obstacle avoidance" to "social compliance". Highlight the Intent CNN and Heuristic Policy as the core of Phase 1.

## 2. Introduction
- **Socially-Aware Navigation:** The need for robots to understand human social cues.
- **The "Freezing Robot" Problem:** Why traditional methods fail in dense human environments.
- **Contributions:** Intent Prediction Network, Rule-based Proactive Policy, Edge Deployment.

## 3. System Architecture (Detailed)
- **Modular Data Bus:** ZMQ + Protobuf for high-speed module communication.
- **Perception Pipeline:**
    - Person detection (YOLO).
    - Identity persistence (IoU Tracker).
    - Behavior analysis (Intent CNN).
- **Control Loop:** Heuristic Policy derived from intent labels.

## 4. Methodology
### 4.1. Intent Prediction Network
- **Backbone:** MobileNetV3-Small (Pretrained).
- **Dual-Head Output:**
    - Classification: Stationry, Approaching, Departing, Crossing, Following, Erratic.
    - Motion Direction: (dx, dy) vector.
- **Inference Optimization:** FP16 CUDA execution.

### 4.2. Heuristic Control Logic
- **Priority Rules:** Safety Stop > Behavior Avoidance > Cruise.
- **Intent-Driven Steering:** Adjusting heading based on "Crossing" predictions before threshold violations.

## 5. Experimental Design (Phase 1.5)
- **Testbed:** Omni-directional robot platform.
- **Scenario A:** Intersecting paths (Crossing intent).
- **Scenario B:** Unpredictable behavior (Erratic intent).
- **Metrics:** TTC (Time To Collision), Personal Space Infractions, Path Smoothness, Latency.

## 6. Discussion of Phase 1 Results
- Improvement over purely reactive baselines.
- Feasibility of real-time intent processing on edge hardware.
- Data logging for future RL phases.

## 7. Future Work
- Transition to Deep Reinforcement Learning (Phase 2).
- Multi-human context fusion.

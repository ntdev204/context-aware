# Context-Aware AI Server — System Design

> **Project**: Context-Aware Navigation for Mecanum Robot
> **Version**: 2.0
> **Date**: 2026-04-18 (updated from v1.0 2026-04-13)
> **Status**: Active — Gap Analysis Integrated

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware Specifications & Constraints](#2-hardware-specifications--constraints)
3. [Architecture Overview](#3-architecture-overview)
4. [Jetson AI Server — Inference Pipeline](#4-jetson-ai-server--inference-pipeline)
5. [Custom CNN — Human Intent Prediction](#5-custom-cnn--human-intent-prediction)
6. [RL Policy — Context-Aware Navigation](#6-rl-policy--context-aware-navigation)
7. [Communication Architecture](#7-communication-architecture)
8. [Local Laptop — VLM-Guided Training](#8-local-laptop--vlm-guided-training)
9. [Raspberry Pi 4 — Integration Interface](#9-raspberry-pi-4--integration-interface)
10. [Memory Budget Analysis](#10-memory-budget-analysis)
11. [Deployment & Model Lifecycle](#11-deployment--model-lifecycle)
12. [Directory Structure](#12-directory-structure)
13. [Architecture Decision Records](#13-architecture-decision-records)
14. [Risk Analysis & Mitigations](#14-risk-analysis--mitigations)
15. [Bottleneck Analysis & Mitigation](#15-bottleneck-analysis--mitigation)
16. [Research Gap Analysis & Novel Contributions](#16-research-gap-analysis--novel-contributions)
17. [Development Strategy](#17-development-strategy)

---

## 1. System Overview

Hệ thống context-aware navigation gồm **3 node** phối hợp hoạt động:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM TOPOLOGY                             │
│                                                                     │
│  ┌──────────────────┐   ZMQ (RT)   ┌──────────────────┐            │
│  │  JETSON ORIN NANO │◄────────────►│  RASPBERRY PI 4  │            │
│  │     (AI Server)   │  <5ms RTT    │ (ROS2 Controller)│            │
│  │                   │              │                   │            │
│  │ • YOLOv11s .engine│              │ • SLAM            │            │
│  │ • Custom CNN      │              │ • Nav2            │            │
│  │ • RL Policy       │              │ • Motor Control   │            │
│  └────────┬──────────┘              └───────────────────┘            │
│           │ gRPC                                                     │
│           │ (WiFi/Ethernet)                                          │
│  ┌────────▼──────────┐                                              │
│  │   LOCAL LAPTOP    │                                              │
│  │ (Training Station)│                                              │
│  │                   │                                              │
│  │ • Gemma 4 VLM     │                                              │
│  │ • RL Training     │                                              │
│  │ • Model Export    │                                              │
│  └───────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Node Responsibilities

| Node                       | Role                | Models / Services           | ROS2     |
| -------------------------- | ------------------- | --------------------------- | -------- |
| **Jetson Orin Nano Super** | AI Inference Server | YOLOv11s + CNN + RL Policy  | ❌ Không |
| **Raspberry Pi 4**         | High-Level Control  | SLAM + Nav2 + Motor Control | ✅ Có    |
| **Local Laptop**           | Training Station    | Gemma 4 VLM + RL Training   | ❌ Không |

---

## 2. Hardware Specifications & Constraints

### Jetson Orin Nano Super (AI Server)

| Spec           | Value                                                 |
| -------------- | ----------------------------------------------------- |
| AI Performance | 67 TOPS (sparse) / 33 TOPS (dense)                    |
| GPU            | Ampere, 1024 CUDA cores, 32 Tensor cores, Max 1020MHz |
| CPU            | 6-core Arm Cortex-A78AE, Max 1.7GHz                   |
| Memory         | **8GB LPDDR5** (shared CPU+GPU), 102GB/s              |
| Storage        | NVMe SSD (recommended) + SD card                      |
| Power Modes    | 7W / 15W / **25W** (recommended for multi-model)      |
| Video Decoder  | 1x 4K60 H.265                                         |
| JetPack        | 6.x (CUDA 12.x, TensorRT 10.x, cuDNN 9.x)             |

> ⚠️ **WARNING**
> **8GB shared memory** là constraint nghiêm trọng nhất. Toàn bộ thiết kế phải tối ưu quanh giới hạn này. Xem [Memory Budget Analysis](#10-memory-budget-analysis).

### Raspberry Pi 4 (Controller)

| Spec    | Value                        |
| ------- | ---------------------------- |
| CPU     | Quad-core Cortex-A72, 1.8GHz |
| RAM     | 4GB / 8GB                    |
| ROS2    | Humble                       |
| Sensors | LiDAR, IMU, Encoders         |

### Local Laptop (Training)

| Spec    | Requirement                                 |
| ------- | ------------------------------------------- |
| GPU     | NVIDIA discrete GPU (RTX 3060+ recommended) |
| VRAM    | ≥ 8GB (for Gemma 4 E4B)                     |
| RAM     | ≥ 16GB                                      |
| Gemma 4 | E4B (9.6GB) or E2B (7.2GB) via Ollama       |

---

## 3. Architecture Overview

### Data Flow — Real-time Inference Loop

```
                        JETSON AI SERVER (Inference Loop @ 15-30 FPS)
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   Camera (CSI/USB)                                                   │
│       │                                                              │
│       ▼                                                              │
│   ┌──────────────┐     Detections     ┌────────────────────┐        │
│   │  YOLOv11s    │────────────────────►│  Detection Filter  │        │
│   │  TensorRT    │                     │  & ROI Extractor   │        │
│   │  (.engine)   │                     └─────────┬──────────┘        │
│   └──────────────┘                               │                   │
│                                         Cropped ROIs (persons)       │
│                                                  │                   │
│                                                  ▼                   │
│                                        ┌──────────────────┐         │
│                                        │   Custom CNN      │         │
│                                        │   Intent Predict  │         │
│                                        │   (TensorRT)      │         │
│                                        └────────┬─────────┘         │
│                                                  │                   │
│                          Intent Vectors          │                   │
│                                                  ▼                   │
│   ┌──────────────────────────────────────────────────────┐           │
│   │              Context State Builder                    │           │
│   │                                                       │           │
│   │  context = {                                          │           │
│   │    frame_embedding,    # Encoded current frame        │           │
│   │    detections,         # [person, obstacle, free]     │           │
│   │    human_intents,      # Per-person intent vectors    │           │
│   │    occupancy_map,      # Simple grid from detections  │           │
│   │    robot_velocity,     # Current vx, vy, vθ          │           │
│   │    action_history,     # Last N actions               │           │
│   │  }                                                    │           │
│   └──────────────────────────┬───────────────────────────┘           │
│                              │                                       │
│                              ▼                                       │
│                    ┌──────────────────┐                              │
│                    │    RL Policy     │                              │
│                    │   (TensorRT)     │                              │
│                    └────────┬─────────┘                              │
│                             │                                        │
│                    NavigationCommand                                  │
│                             │                                        │
└─────────────────────────────┼────────────────────────────────────────┘
                              │
                    ZMQ PUB (< 5ms)
                              │
                              ▼
                    ┌──────────────────┐
                    │  RASPBERRY PI 4  │
                    │  Nav2 / Motor    │
                    └──────────────────┘
```

### Processing Timeline — Single Frame

```
Time (ms)   0        5        10       15       20       25       30
            │────────│────────│────────│────────│────────│────────│
            │ Camera │ YOLO   │  ROI   │  CNN   │Context │   RL   │ ZMQ
            │ Grab   │Infer   │Extract │Intent  │ Build  │ Policy │ Send
            │  ~2ms  │ ~6ms   │  ~1ms  │ ~3ms   │  ~1ms  │ ~2ms   │~1ms
            │        │        │        │        │        │        │
            └────────────────── Total: ~16ms ≈ 60 FPS ─────────────┘
                     (Target: 30 FPS with safety margin)
```

---

## 4. Jetson AI Server — Inference Pipeline

### 4.1 YOLOv11s — Object Detection

| Property   | Value                                              |
| ---------- | -------------------------------------------------- |
| Model      | YOLOv11s (small variant)                           |
| Format     | TensorRT `.engine` (FP16)                          |
| Input      | 640×640 RGB                                        |
| Output     | Bounding boxes + class + confidence                |
| Target FPS | 30+ FPS                                            |
| Classes    | `person`, `obstacle`, `door`, `wall`, `free_space` |
| VRAM       | ~80-120MB                                          |

**Export Pipeline:**

```bash
# On Jetson (JetPack 6.x)
from ultralytics import YOLO
model = YOLO("yolo11s.pt")
model.export(format="engine", half=True, device=0, imgsz=640)
# Output: yolo11s.engine (~30MB)
```

**Detection Post-Processing:**

```python
class DetectionResult:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    confidence: float
    track_id: int                     # from ByteTrack/BoTSORT

class FrameDetections:
    persons: List[DetectionResult]
    obstacles: List[DetectionResult]
    free_zones: List[Tuple[float, float]]  # angular sectors
    timestamp: float
```

**Tracker**: Tích hợp **ByteTrack** để gán `track_id` ổn định cho mỗi person qua frames → cần thiết cho follow mode và intent prediction liên tục.

### 4.2 ROI Extraction

```python
class ROIExtractor:
    """Trích xuất cropped regions cho CNN Intent Prediction."""

    def extract(self, frame: np.ndarray, detections: FrameDetections) -> List[PersonROI]:
        rois = []
        for person in detections.persons:
            x1, y1, x2, y2 = person.bbox

            # Expand bbox 10% for context
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            crop = frame[
                max(0, y1-pad_y) : min(frame.shape[0], y2+pad_y),
                max(0, x1-pad_x) : min(frame.shape[1], x2+pad_x)
            ]

            # Resize to CNN input size
            crop_resized = cv2.resize(crop, (CNN_INPUT_W, CNN_INPUT_H))

            rois.append(PersonROI(
                image=crop_resized,
                bbox=person.bbox,
                track_id=person.track_id,
                relative_position=self._calc_relative_pos(person.bbox, frame.shape)
            ))
        return rois
```

---

## 5. Custom CNN — Human Intent Prediction

### 5.1 Architecture Design

> **IMPORTANT**
> **Đây là module BẮT BUỘC** — Core differentiator của hệ thống context-aware.

**Input**: Cropped ROI image (person region) — Resize to `128×256×3`
**Output**: Intent class probabilities + motion direction vector

```
┌─────────────────────────────────────────────────────┐
│              Custom CNN Architecture                 │
│                                                      │
│  Input: 128×256×3 (Cropped Person ROI)              │
│    │                                                 │
│    ▼                                                 │
│  ┌──────────────────────┐                           │
│  │ Backbone (Lightweight)│                           │
│  │ MobileNetV3-Small     │  ◄── Transfer Learning   │
│  │ (pretrained, frozen   │       from ImageNet       │
│  │  first 3 blocks)      │                           │
│  └──────────┬───────────┘                           │
│             │ Feature Map: 7×14×576                  │
│             ▼                                        │
│  ┌──────────────────────┐                           │
│  │ Temporal Aggregator   │                           │
│  │ (Conv1D over last    │  ◄── 5-frame history      │
│  │  5 frames per track) │       per track_id         │
│  └──────────┬───────────┘                           │
│             │                                        │
│             ▼                                        │
│  ┌──────────────────────┐    ┌───────────────────┐  │
│  │   Intent Head        │    │  Direction Head    │  │
│  │   FC(256)→FC(6)      │    │  FC(256)→FC(2)    │  │
│  │   Softmax            │    │  Tanh (dx, dy)    │  │
│  └──────────┬───────────┘    └────────┬──────────┘  │
│             │                         │              │
│             ▼                         ▼              │
│     Intent Class                Motion Vector        │
│  [6 probabilities]           [dx, dy] normalized     │
└─────────────────────────────────────────────────────┘
```

### 5.2 Intent Classes

| Class ID | Intent        | Description                 | Robot Response            |
| -------- | ------------- | --------------------------- | ------------------------- |
| 0        | `STATIONARY`  | Người đứng yên              | Giảm tốc, giữ khoảng cách |
| 1        | `APPROACHING` | Người đi tới robot          | Giảm tốc + chuẩn bị tránh |
| 2        | `DEPARTING`   | Người đi xa khỏi robot      | Duy trì / tăng tốc        |
| 3        | `CROSSING`    | Người cắt ngang đường robot | Dừng / chờ                |
| 4        | `FOLLOWING`   | Người đi theo robot         | Giữ tốc ổn định           |
| 5        | `ERRATIC`     | Chuyển động bất thường      | Dừng + cảnh báo           |

### 5.3 CNN Specifications

| Property        | Value                                       |
| --------------- | ------------------------------------------- |
| Backbone        | MobileNetV3-Small (pretrained ImageNet)     |
| Input Size      | 128×256×3                                   |
| Temporal Window | 5 frames per track_id                       |
| Parameters      | ~2.5M (backbone) + ~500K (heads) ≈ 3M total |
| TensorRT Format | FP16 `.engine`                              |
| VRAM            | ~30-60MB                                    |
| Inference Time  | ~2-4ms per person                           |
| Max Persons     | 5 concurrent (batch inference)              |

### 5.4 Training Data Strategy

```
┌─────────────────────────────────────────────────┐
│          CNN Training Pipeline (Laptop)          │
│                                                   │
│  1. Data Collection (Jetson → Laptop via gRPC)   │
│     • Cropped ROIs + track_ids + timestamps       │
│     • ~10,000 labeled sequences minimum           │
│                                                   │
│  2. Labeling Strategy                             │
│     ├─ Semi-auto: Gemma 4 VLM pre-labels ROIs    │
│     │   "What is this person's movement intent?"  │
│     └─ Human review: Verify/correct VLM labels    │
│                                                   │
│  3. Training                                      │
│     • PyTorch + MobileNetV3 transfer learning     │
│     • CrossEntropyLoss (intent) + MSELoss (dir)   │
│     • Augmentation: ColorJitter, RandomCrop, Flip │
│                                                   │
│  4. Export                                         │
│     • PyTorch → ONNX → TensorRT .engine (FP16)   │
│     • Deploy to Jetson via gRPC                    │
└─────────────────────────────────────────────────┘
```

---

## 6. RL Policy — Context-Aware Navigation

### 6.1 Problem Formulation (MDP)

> **NOTE**
> RL Policy quyết định **chiến lược navigation** (mode + tốc độ + hướng), không phải low-level motor control. Motor control do Nav2 trên RasPi xử lý.

#### State Space (Observation) — VLM-Aligned

**Chỉ dùng camera từ Jetson** — không phụ thuộc LiDAR từ RasPi.

> **GAP #4 Resolution**: State space bao gồm VLM semantic embedding để bridge gap giữa visual reasoning (VLM) và numeric policy (RL). Xem [GAP #4](#gap-4--vlm-rl-state-alignment).

```python
@dataclass
class ObservationSpace:
    # ── Scene Encoding (from YOLO detections) ──
    num_persons: int                      # Số người phát hiện (0-10)
    nearest_person_distance: float        # Khoảng cách person gần nhất (m)
    nearest_person_angle: float           # Góc person gần nhất (rad)
    nearest_obstacle_distance: float      # Khoảng cách obstacle gần nhất
    nearest_obstacle_angle: float         # Góc obstacle gần nhất
    free_space_ratio: float               # Tỷ lệ vùng thông thoáng (0-1)

    # ── Spatial Grid (simple occupancy from detections) ──
    occupancy_grid: np.ndarray            # 8×8 grid (64 cells) — 0=free, 1=occupied

    # ── Human Intent Features (from CNN, top-3 persons) ──
    person_intents: np.ndarray            # (3, 8) — [6 intent probs + dx, dy] × 3 persons

    # ── Robot State (internal) ──
    current_velocity: np.ndarray          # [vx, vy, vθ]
    previous_action: np.ndarray           # Last action taken

    # ── VLM Semantic Embedding (GAP #4) ──
    # Compressed scene understanding from VLM encoder.
    # Updated async (2-5s delay) — cached between updates.
    # Weight decays when embedding is stale (see staleness_weight).
    vlm_embedding: np.ndarray             # (64,) — projected from VLM hidden state
    vlm_staleness_weight: float           # [0, 1] — decays with age of embedding

    # Total dimensions: 5 + 1 + 64 + 24 + 3 + 5 + 64 + 1 = ~167 floats
```

**VLM Embedding Pipeline (GAP #4) — with staleness handling**:

```python
class VLMStateAligner:
    """Projects VLM hidden states into RL-compatible embedding space.

    Bridges the semantic gap between visual reasoning (VLM sees images)
    and policy optimization (RL operates on float vectors).

    Handles latency mismatch: VLM runs at ~100-500ms while control loop
    runs at ~16ms (60Hz). Embedding is cached and its weight decays
    with staleness to prevent stale features from misleading the policy.
    """

    STALENESS_HALF_LIFE_S = 2.0  # weight drops to 0.5 after 2 seconds

    def __init__(self, vlm_hidden_dim: int = 2048, projection_dim: int = 64):
        self.projector = nn.Sequential(
            nn.Linear(vlm_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self._cache = np.zeros(projection_dim)
        self._last_update_time = 0.0

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Extract VLM embedding from image (runs async on Server)."""
        hidden = self.vlm.encode_image(image)
        pooled = hidden[:, 0, :]
        projected = self.projector(pooled)
        self._cache = projected.detach().numpy()
        self._last_update_time = time.monotonic()
        return self._cache

    @property
    def staleness_weight(self) -> float:
        """Exponential decay: w = 2^(-age / half_life).

        At 0s: w=1.0, at 2s: w=0.5, at 4s: w=0.25, at 10s: w≈0.03
        This ensures stale embeddings are progressively ignored.
        """
        age = time.monotonic() - self._last_update_time
        return 2.0 ** (-age / self.STALENESS_HALF_LIFE_S)

    @property
    def weighted_embedding(self) -> np.ndarray:
        """Returns embedding scaled by staleness weight."""
        return self._cache * self.staleness_weight
```

#### Action Space (Continuous)

```python
@dataclass
class ActionSpace:
    # Navigation Mode (discrete, embedded as logits → argmax)
    mode_logits: np.ndarray     # (5,) → [CRUISE, CAUTIOUS, AVOID, FOLLOW, STOP]

    # Velocity Control (continuous)
    velocity_scale: float       # [0.0, 1.0] — Tỷ lệ tốc độ tối đa
    heading_offset: float       # [-π/4, π/4] — Điều chỉnh hướng (rad)

    # Follow Target (only active in FOLLOW mode)
    follow_target_idx: int      # Index of target person (0-2)
```

#### Navigation Modes

| Mode       | Trigger Condition                           | Robot Behavior                 |
| ---------- | ------------------------------------------- | ------------------------------ |
| `CRUISE`   | `free_space_ratio > 0.8`, no persons nearby | Tốc độ tối đa, đi thẳng        |
| `CAUTIOUS` | Persons detected nhưng không blocking       | Giảm 40% tốc, quan sát         |
| `AVOID`    | Person/obstacle trên đường đi               | Giảm 70% tốc, điều hướng tránh |
| `FOLLOW`   | Nhận lệnh follow hoặc detect gesture        | Bám theo target person         |
| `STOP`     | Nguy hiểm / người quá gần / erratic         | Dừng hoàn toàn                 |

#### Reward Function — Structured VLM-Guided with Temporal Intent (GAP #1 + #2)

> **GAP #1 Resolution**: We extend prior VLM-guided RL (which collapses VLM output into a single scalar) by extracting **structured sub-scores** from VLM reasoning. These sub-scores are **intermediate representations** — they are combined via learned weights into a single scalar `R_t ∈ ℝ` as required by standard RL theory (Sutton & Barto). This preserves the VLM's reasoning structure while remaining compatible with SAC optimization.
>
> **GAP #2 Resolution**: We extend the reward signal with a **temporal intent penalty term** `R_intent` that evaluates the robot's anticipation accuracy over a lookahead window `k=5` frames. This term penalizes late reactions and rewards proactive adjustments.

**VLM Structured Output (intermediate decomposition, NOT the final reward)**:

> **RL Theory Note**: Standard RL requires `R_t ∈ ℝ` (scalar). The structured VLM output below is an **intermediate signal** that is linearly combined into a single scalar reward. This is NOT multi-objective RL (MORL) — we do not compute a Pareto front. The decomposition only serves to make the VLM evaluation more interpretable and the weighting strategy more transparent.

```python
@dataclass
class VLMRewardSignal:
    """Structured intermediate signal from VLM evaluation (GAP #1).

    VLM evaluates on 3 axes independently → combined into scalar R_vlm.
    R_vlm = w_s * safety + w_soc * social + w_e * efficiency
    This is a standard scalarization approach, NOT MORL.
    """
    safety_score: float          # [-1, 1] Was the decision safe for humans?
    social_score: float          # [-1, 1] Was it socially appropriate?
    efficiency_score: float      # [-1, 1] Was it efficient toward goal?
    reasoning: str               # VLM's natural language explanation
    reasoning_embedding: np.ndarray  # (64,) encoded reasoning vector
    confidence: float            # VLM self-assessed confidence [0, 1]
```

**VLM Prompt (structured JSON output)**:

```
You are evaluating a robot's navigation decision.

[IMAGE: camera_frame]

Context:
- Detected: {num_persons} persons, {num_obstacles} obstacles
- Nearest person: {distance}m, intent: {intent_name}
- Intent trajectory (last 5 frames): {intent_sequence}
- Free space: {ratio}%

Robot action:
- Mode: {mode_name}, Speed: {velocity}%, Direction: {heading}deg

Evaluate this decision on THREE independent axes.
Respond with JSON:
{
  "safety_score": float,       // [-1, 1] physical safety
  "social_score": float,       // [-1, 1] social appropriateness
  "efficiency_score": float,   // [-1, 1] goal efficiency
  "reasoning": "string",       // explanation of your assessment
  "confidence": float          // [0, 1] how certain you are
}
```

**Temporal Intent Reward Component (GAP #2)**:

**Formal definition**:

```
R_intent(t) = Σ_{i=0}^{k-1} γ^i · risk(intent_{t+i})

where:
  k = 5             (lookahead window, in frames at ~30 FPS ≈ 167ms)
  γ = 0.9           (temporal discount — recent intents matter more)
  risk(·) ∈ [-1, 1] (intent risk score, see table below)
```

**Intent risk mapping** (non-differentiable — computed as penalty, not backpropagated):

| Intent | risk(·) | Rationale |
|---|---|---|
| STATIONARY | 0.0 | No risk |
| APPROACHING | -0.3 | Moderate risk — robot should slow |
| DEPARTING | +0.2 | Positive — person moving away |
| CROSSING | -0.6 | High risk — robot should stop/wait |
| FOLLOWING | 0.0 | Neutral |
| ERRATIC | -0.8 | Highest risk — unpredictable |

```python
INTENT_RISK = {0: 0.0, 1: -0.3, 2: 0.2, 3: -0.6, 4: 0.0, 5: -0.8}
TEMPORAL_GAMMA = 0.9
LOOKAHEAD_K = 5  # frames (~167ms at 30 FPS)

def compute_temporal_intent_reward(
    intent_trajectory: List[IntentPrediction],  # intent[t:t+k]
    action_taken: ActionSpace,
) -> float:
    """Discounted temporal intent risk (GAP #2).

    R_intent = Σ γ^i · risk(intent_{t+i})

    This is NOT differentiable — it is a shaped reward signal,
    not a loss function. RL (SAC) treats it as any other scalar reward.

    Additionally applies an anticipation penalty: if the trajectory
    shows an upcoming high-risk intent but the robot did not
    preemptively decelerate, apply extra penalty.
    """
    if len(intent_trajectory) < 2:
        return 0.0

    k = min(LOOKAHEAD_K, len(intent_trajectory))

    # 1. Discounted risk sum
    risk_sum = sum(
        (TEMPORAL_GAMMA ** i) * INTENT_RISK.get(p.intent_class, 0.0)
        for i, p in enumerate(intent_trajectory[:k])
    )

    # 2. Late-reaction penalty: intent changed to high-risk but robot
    #    did not reduce speed in the first half of the window
    intent_classes = [p.intent_class for p in intent_trajectory[:k]]
    high_risk_in_window = any(c in (3, 5) for c in intent_classes)  # CROSSING or ERRATIC
    mode_idx = action_taken.mode_logits.argmax()
    robot_did_not_slow = mode_idx in (0, 1)  # CRUISE or CAUTIOUS
    if high_risk_in_window and robot_did_not_slow:
        risk_sum -= 0.4  # anticipation failure penalty

    return risk_sum
```

**Composite Reward Function (scalar output — standard RL compatible)**:

```
Formal definition:

  R_t = w_vlm · R_vlm(t) + w_safety · R_safety(t) + w_intent · R_intent(t) + w_eff · R_eff(t)

  where R_t ∈ ℝ  (scalar — standard MDP reward)

  R_vlm    = scalarize(VLM structured output)  — GAP #1
  R_intent = Σ γ^i · risk(intent_{t+i})        — GAP #2
  R_safety = hard-coded penalties               — engineered
  R_eff    = smoothness + progress              — heuristic
```

```python
# Scalarization weights (tunable hyperparameters)
W_VLM = 0.35
W_SAFETY = 0.25
W_INTENT = 0.20
W_EFF = 0.20

# VLM sub-score weights (fixed — defines what "good navigation" means)
W_VLM_SAFETY = 0.4
W_VLM_SOCIAL = 0.3
W_VLM_EFFICIENCY = 0.3

def compute_reward(
    transition,
    vlm_signal: VLMRewardSignal,
    intent_trajectory: List[IntentPrediction],
) -> float:
    """Compute scalar reward R_t ∈ ℝ from structured components.

    This is standard scalarized reward shaping, NOT multi-objective RL.
    VLM sub-scores are intermediate — final reward is always scalar.

    GAP #1: VLM provides structured evaluation (safety/social/efficiency)
            → scalarized via fixed weights.
    GAP #2: Temporal intent trajectory → discounted risk penalty.
    """
    # === R_vlm: Scalarized VLM component (GAP #1) ===
    r_vlm = (
        W_VLM_SAFETY * vlm_signal.safety_score
        + W_VLM_SOCIAL * vlm_signal.social_score
        + W_VLM_EFFICIENCY * vlm_signal.efficiency_score
    )
    r_vlm *= vlm_signal.confidence  # down-weight uncertain evaluations

    # === R_safety: Hard-coded constraints (NOT learned, NOT from VLM) ===
    r_safety = 0.0
    if transition.collision:
        r_safety -= 10.0  # catastrophic — dominates all other terms
    if transition.min_person_distance < SAFETY_DISTANCE:
        r_safety -= 2.0 * (SAFETY_DISTANCE - transition.min_person_distance)
    if transition.mode == STOP and transition.danger_detected:
        r_safety += 1.0

    # === R_intent: Temporal intent risk (GAP #2) ===
    r_intent = compute_temporal_intent_reward(
        intent_trajectory, transition.action_taken
    )

    # === R_eff: Efficiency heuristic ===
    r_eff = 0.0
    r_eff -= 0.1 * np.linalg.norm(transition.accel_delta)  # jerk penalty
    if transition.goal_progress > 0:
        r_eff += 0.5 * transition.goal_progress
    if transition.free_space_ratio > 0.8 and transition.velocity_scale > 0.7:
        r_eff += 0.3

    # === Final scalar reward R_t ∈ ℝ ===
    R_t = W_VLM * r_vlm + W_SAFETY * r_safety + W_INTENT * r_intent + W_EFF * r_eff
    return R_t
```

**Reward Weight Analysis**:

| Component | Symbol | Weight | Source | Differentiable? |
|---|---|---|---|---|
| VLM Structured | `R_vlm` | 0.35 | Gemma 4 (3 sub-scores → scalarized) | No (reward shaping) |
| Safety Hard | `R_safety` | 0.25 | Hard-coded rules | No (engineered) |
| Temporal Intent | `R_intent` | 0.20 | CNN trajectory, k=5, γ=0.9 | No (shaped penalty) |
| Efficiency | `R_eff` | 0.20 | Heuristic | No (shaped bonus) |

> **Note on differentiability**: None of the reward components need to be differentiable. SAC uses rewards as scalar signals stored in the replay buffer — gradients flow through the actor/critic networks, not through the reward function.

### 6.2 Algorithm: SAC (Soft Actor-Critic)

| Property      | Value                           | Rationale                                    |
| ------------- | ------------------------------- | -------------------------------------------- |
| Algorithm     | SAC (Soft Actor-Critic)         | Sample efficient, continuous actions, stable |
| Framework     | Stable Baselines3 (PyTorch)     | Production-ready, well-documented            |
| Network       | MLP [256, 256] (actor & critic) | Sufficient for ~102-dim observation          |
| Learning Rate | 3e-4                            | SAC default, proven stable                   |
| Batch Size    | 256                             | Balance speed/stability                      |
| Replay Buffer | 100K transitions                | Limited by laptop RAM                        |
| γ (discount)  | 0.99                            | Long-horizon navigation                      |
| α (entropy)   | Auto-tuned                      | SAC automatic temperature                    |
| Export Format | PyTorch → ONNX → TensorRT FP16  | Edge deployment optimized                    |
| VRAM (Jetson) | ~30-50MB                        | Tiny MLP, very lightweight                   |

### 6.3 Training Architecture (on Laptop)

```
┌──────────────────────────────────────────────────────────────────┐
│                VLM-GUIDED RL TRAINING LOOP                       │
│                    (Local Laptop)                                │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │  1. EXPERIENCE COLLECTION (from Jetson via gRPC)        │    │
│   │     ┌──────────────────────────────────────────┐        │    │
│   │     │  ExperienceFrame {                       │        │    │
│   │     │    camera_frame: JPEG (compressed)       │        │    │
│   │     │    detections: List[Detection]           │        │    │
│   │     │    human_intents: List[IntentVector]     │        │    │
│   │     │    observation: ObservationSpace         │        │    │
│   │     │    action_taken: ActionSpace             │        │    │
│   │     │    robot_state: RobotState               │        │    │
│   │     │    timestamp: float                      │        │    │
│   │     │  }                                       │        │    │
│   │     └──────────────────────────────────────────┘        │    │
│   └─────────────────────────┬───────────────────────────────┘    │
│                             │                                    │
│   ┌─────────────────────────▼────────────────────────────────┐   │
│   │  2. VLM REWARD SHAPING (Gemma 4 via Ollama)              │   │
│   │                                                          │   │
│   │  For each (frame, action, context) tuple:                │   │
│   │                                                          │   │
│   │  Prompt Template:                                        │   │
│   │  ┌────────────────────────────────────────────────────┐  │   │
│   │  │ [IMAGE: camera_frame]                              │  │   │
│   │  │                                                    │  │   │
│   │  │ You are evaluating a robot's navigation decision.  │  │   │
│   │  │                                                    │  │   │
│   │  │ Context:                                           │  │   │
│   │  │ - Detected: {num_persons} persons, {num_obstacles} │  │   │
│   │  │ - Nearest person: {distance}m at {angle}deg        │  │   │
│   │  │ - Human intents: {intent_descriptions}             │  │   │
│   │  │ - Free space: {ratio}%                             │  │   │
│   │  │                                                    │  │   │
│   │  │ Robot action:                                      │  │   │
│   │  │ - Mode: {mode_name}                                │  │   │
│   │  │ - Speed: {velocity}% of max                        │  │   │
│   │  │ - Direction adjustment: {heading}deg               │  │   │
│   │  │                                                    │  │   │
│   │  │ Rate this decision from -1.0 to 1.0:               │  │   │
│   │  │ - Safety, Efficiency, Social appropriateness       │  │   │
│   │  │ Respond with JSON: {"score": float, "reason": str} │  │   │
│   │  └────────────────────────────────────────────────────┘  │   │
│   │                                                          │   │
│   │  → VLM Score → Reward Function → Total Reward            │   │
│   └─────────────────────────┬────────────────────────────────┘   │
│                             │                                    │
│   ┌─────────────────────────▼───────────────────────────────┐    │
│   │  3. SAC TRAINING                                        │    │
│   │     • Store (s, a, r, s') in Replay Buffer              │    │
│   │     • Train actor + critic networks                     │    │
│   │     • Auto-tune entropy temperature                     │    │
│   │     • Log metrics to TensorBoard                        │    │
│   └─────────────────────────┬───────────────────────────────┘    │
│                             │                                    │
│   ┌─────────────────────────▼───────────────────────────────┐    │
│   │  4. MODEL EXPORT & DEPLOY                               │    │
│   │     • PyTorch → ONNX (opset 17)                         │    │
│   │     • ONNX → TensorRT .engine (FP16, on Jetson)         │    │
│   │     • Transfer via gRPC → Hot-reload on Jetson          │    │
│   └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. Communication Architecture

### 7.1 Protocol Strategy

| Path            | Protocol        | Purpose                         | Latency Target |
| --------------- | --------------- | ------------------------------- | -------------- |
| Jetson → RasPi  | **ZMQ PUB/SUB** | Navigation commands (real-time) | < 5ms          |
| RasPi → Jetson  | **ZMQ PUB/SUB** | Robot state feedback            | < 5ms          |
| Jetson ↔ RasPi  | **ZMQ REQ/REP** | Config, mode changes            | < 50ms         |
| Jetson → Laptop | **gRPC stream** | Experience data upload          | Best-effort    |
| Laptop → Jetson | **gRPC unary**  | Model deployment, config        | < 1s           |
| Laptop → Jetson | **gRPC unary**  | VLM query response              | Async          |

### 7.2 ZMQ Channels (Jetson ↔ RasPi)

```
┌─────────────────┐                    ┌─────────────────┐
│     JETSON       │                    │    RASPBERRY PI  │
│                  │                    │                  │
│  PUB :5555 ─────────────────────────► SUB :5555         │
│  "ai/nav_cmd"   │  NavigationCmd     │  → Nav2 bridge   │
│                  │  @30Hz             │                  │
│  PUB :5556 ─────────────────────────► SUB :5556         │
│  "ai/detections" │  DetectionList    │  → Visualization │
│                  │  @10Hz             │                  │
│                  │                    │                  │
│  SUB :5560 ◄───────────────────────── PUB :5560         │
│  "robot/state"  │  RobotState        │  odom+battery    │
│                  │  @20Hz             │  @20Hz           │
│                  │                    │                  │
│  REP :5570 ◄───────────────────────── REQ :5570         │
│  "ai/config"    │  Config/Mode       │  mode changes    │
│                  │  on-demand         │                  │
└─────────────────┘                    └─────────────────┘
```

### 7.3 Message Schemas (Protobuf)

```protobuf
// messages.proto — Shared between Jetson and RasPi

syntax = "proto3";
package context_aware;

// === Jetson → RasPi ===

enum NavigationMode {
  CRUISE = 0;
  CAUTIOUS = 1;
  AVOID = 2;
  FOLLOW = 3;
  STOP = 4;
}

message NavigationCommand {
  NavigationMode mode = 1;
  float velocity_scale = 2;      // 0.0 - 1.0
  float heading_offset = 3;      // radians
  int32 follow_target_id = 4;    // -1 if not following
  double timestamp = 5;
  float confidence = 6;          // RL policy confidence
}

message Detection {
  int32 track_id = 1;
  float x1 = 2;
  float y1 = 3;
  float x2 = 4;
  float y2 = 5;
  string class_name = 6;
  float confidence = 7;
  int32 intent_class = 8;        // from CNN
  float intent_confidence = 9;
}

message DetectionList {
  repeated Detection detections = 1;
  double timestamp = 2;
  int32 frame_id = 3;
}

// === RasPi → Jetson ===

message RobotState {
  float vx = 1;                  // current velocity x (m/s)
  float vy = 2;                  // current velocity y (m/s)
  float vtheta = 3;              // current angular velocity (rad/s)
  float pos_x = 4;
  float pos_y = 5;
  float pos_theta = 6;
  float battery_percent = 7;
  string nav2_status = 8;       // "idle", "navigating", "stuck"
  double timestamp = 9;
}
```

### 7.4 gRPC Services (Jetson ↔ Laptop)

```protobuf
// training_service.proto — Jetson ↔ Laptop

syntax = "proto3";
package training;

// === Experience Streaming (Jetson → Laptop) ===

message ExperienceFrame {
  bytes camera_frame_jpeg = 1;    // Compressed JPEG
  repeated Detection detections = 2;
  repeated float observation = 3;  // Flattened observation vector
  repeated float action = 4;      // Action taken
  float velocity_scale = 5;
  double timestamp = 6;
}

service ExperienceCollector {
  rpc StreamExperience (stream ExperienceFrame) returns (StreamAck);
  rpc GetExperienceBatch (BatchRequest) returns (ExperienceBatch);
}

// === Model Deployment (Laptop → Jetson) ===

message ModelArtifact {
  string model_name = 1;          // "rl_policy" or "intent_cnn"
  bytes onnx_data = 2;            // ONNX model bytes
  string version = 3;
  map<string, string> metadata = 4;
}

message DeployResult {
  bool success = 1;
  string message = 2;
  string active_version = 3;
}

service ModelDeployer {
  rpc DeployModel (ModelArtifact) returns (DeployResult);
  rpc GetActiveModels (Empty) returns (ModelList);
  rpc RollbackModel (RollbackRequest) returns (DeployResult);
}

// === VLM Query (Jetson → Laptop, async) ===

message VLMQuery {
  bytes image_jpeg = 1;
  string context_json = 2;       // JSON with detections, actions
  string prompt_template = 3;
}

message VLMResponse {
  // Multi-dimensional reward signal (GAP #1)
  float safety_score = 1;        // [-1, 1] physical safety
  float social_score = 2;        // [-1, 1] social appropriateness
  float efficiency_score = 3;    // [-1, 1] goal efficiency
  string reasoning = 4;          // natural language explanation
  repeated float reasoning_embedding = 5;  // encoded reasoning vector
  float confidence = 6;          // VLM self-assessed confidence [0, 1]
  double latency_ms = 7;
}

service VLMService {
  rpc EvaluateDecision (VLMQuery) returns (VLMResponse);
  rpc BatchEvaluate (stream VLMQuery) returns (stream VLMResponse);
  rpc EncodeScene (VLMQuery) returns (SceneEmbedding);  // GAP #4: state alignment
}

message SceneEmbedding {
  repeated float embedding = 1;  // projected VLM hidden state for RL state
  double timestamp = 2;
}
```

### 7.5 Serialization Strategy

| Channel        | Serialization         | Rationale                                   |
| -------------- | --------------------- | ------------------------------------------- |
| ZMQ real-time  | **Protobuf** (binary) | Smallest payload, fastest parse             |
| gRPC all       | **Protobuf** (native) | Built-in to gRPC                            |
| Image transfer | **JPEG** (quality 85) | 10x smaller than raw, fast encode on Jetson |

---

## 8. Local Laptop — VLM-Guided Training

### 8.1 Gemma 4 VLM Configuration

```yaml
# Ollama configuration on Laptop
model: gemma4:e4b # 9.6GB — or e2b (7.2GB) if laptop is weaker
quantization: Q4_K_M # Default from Ollama
context_window: 8192 # Sufficient for evaluation prompts
temperature: 0.3 # Low for consistent scoring
num_gpu: 99 # Offload all layers to GPU


# Thinking mode: DISABLED for scoring
# (Enable for complex scene analysis if needed)
```

### 8.2 VLM Evaluation Pipeline

```
┌─────────────────────────────────────────────────┐
│           VLM Evaluation Modes                   │
│                                                   │
│  Mode 1: OFFLINE BATCH (Primary — Training)      │
│  ─────────────────────────────────────────────   │
│  • Collect experience buffer from Jetson          │
│  • Batch-evaluate frames with Gemma 4             │
│  • ~2-5 seconds per evaluation                    │
│  • Generate reward labels                         │
│  • Feed into SAC replay buffer                    │
│  • Throughput: ~500-1000 evaluations/hour         │
│                                                   │
│  Mode 2: ONLINE ASYNC (Secondary — Fine-tuning)  │
│  ─────────────────────────────────────────────   │
│  • Jetson streams live experience via gRPC        │
│  • Laptop evaluates async (non-blocking)          │
│  • Scores arrive 2-5s delayed                     │
│  • Used for online reward augmentation            │
│  • Does NOT block Jetson inference loop           │
│                                                   │
│  Mode 3: ACTIVE LEARNING (Periodic)              │
│  ─────────────────────────────────────────────   │
│  • VLM identifies uncertain/novel scenarios       │
│  • Flags frames for human review                  │
│  • Improves CNN training dataset                  │
│  • Runs daily/weekly                              │
└─────────────────────────────────────────────────┘
```

### 8.3 Training Schedule

```
Phase 1: CNN Pre-training (Week 1-2)
├─ Collect ROI dataset (~5000+ labeled images)
├─ VLM-assisted labeling with human verification
├─ Train MobileNetV3 + intent heads
└─ Export to TensorRT, deploy to Jetson

Phase 2: RL Warm-up (Week 3-4)
├─ Collect experience with random/heuristic policy
├─ VLM evaluates all collected experience
├─ Train SAC with VLM-shaped rewards
└─ Deploy v1 policy to Jetson

Phase 3: RL Iteration (Week 5+)
├─ Collect experience with current RL policy
├─ VLM evaluates new experience
├─ Continue SAC training (off-policy → efficient)
├─ Deploy improved policy
└─ Repeat cycle until convergence
```

---

## 9. Raspberry Pi 4 — Integration Interface

> **NOTE**
> RasPi repo đã hoàn thiện. Phần này mô tả **interface** mà Jetson giao tiếp, không code thêm trên RasPi.

### 9.1 ZMQ-ROS2 Bridge (trên RasPi)

RasPi cần một **bridge node** để chuyển đổi giữa ZMQ messages và ROS2 topics:

```
┌──────────────────────────────────────────────────────┐
│              RASPBERRY PI 4 (ROS2 Side)              │
│                                                       │
│  ZMQ SUB (:5555) ──► /ai/nav_cmd (geometry_msgs/Twist│
│                       + custom mode)                   │
│                                                       │
│  ZMQ SUB (:5556) ──► /ai/detections (visualization)  │
│                                                       │
│  /odom ──────────►  ZMQ PUB (:5560)                  │
│  /battery_state ─►  (RobotState message)              │
│  /nav2_status ───►                                    │
│                                                       │
│  Navigation Logic:                                     │
│  ┌─────────────────────────────────────────┐          │
│  │  if nav_cmd.mode == CRUISE:             │          │
│  │    apply cmd_vel at full scale          │          │
│  │  elif nav_cmd.mode == FOLLOW:           │          │
│  │    override local planner with AI cmd   │          │
│  │  elif nav_cmd.mode == STOP:             │          │
│  │    publish zero velocity immediately    │          │
│  │  else:                                  │          │
│  │    blend AI cmd with Nav2 local planner │          │
│  └─────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────┘
```

### 9.2 Command Interpretation

| AI Mode    | RasPi Behavior                              | Nav2 Integration                                |
| ---------- | ------------------------------------------- | ----------------------------------------------- |
| `CRUISE`   | cmd_vel.linear = max × velocity_scale       | Global planner active, local planner follows AI |
| `CAUTIOUS` | cmd_vel.linear = max × velocity_scale × 0.6 | Local planner active with conservative params   |
| `AVOID`    | Heading offset applied to cmd_vel           | Local planner DWB with AI-suggested direction   |
| `FOLLOW`   | Track target, ignore global goal            | Local planner disabled, direct velocity control |
| `STOP`     | Zero velocity, emergency                    | All planners paused                             |

---

## 10. Memory Budget Analysis

### 10.1 Jetson VRAM Budget (8GB Shared)

```
┌─────────────────────────────────────────────────────────────┐
│                 MEMORY ALLOCATION (8GB Total)                │
│                                                              │
│  ████████████████████████████████████████████████ 8192 MB    │
│  │                                                │          │
│  │  OS + System (headless)         ≈ 1800 MB     │          │
│  │  ░░░░░░░░░░░░                                  │          │
│  │                                                │          │
│  │  CUDA Runtime + Libraries       ≈  500 MB     │          │
│  │  ░░░░░                                         │          │
│  │                                                │          │
│  │  YOLOv11s TensorRT FP16        ≈  120 MB     │          │
│  │  █                                             │          │
│  │                                                │          │
│  │  Custom CNN TensorRT FP16      ≈   60 MB     │          │
│  │  █                                             │          │
│  │                                                │          │
│  │  RL Policy TensorRT FP16       ≈   50 MB     │          │
│  │  █                                             │          │
│  │                                                │          │
│  │  Camera Buffer (1080p × 3)     ≈  100 MB     │          │
│  │  █                                             │          │
│  │                                                │          │
│  │  ZMQ + gRPC + Buffers          ≈  100 MB     │          │
│  │  █                                             │          │
│  │                                                │          │
│  │  Processing Overhead           ≈  200 MB     │          │
│  │  ██                                            │          │
│  │                                                │          │
│  │  ═══════════════════════════════════════       │          │
│  │  TOTAL USED                    ≈ 2930 MB     │          │
│  │  FREE                          ≈ 5262 MB  ✅ │          │
│  │                                                │          │
│  │  Safety Margin                 ≈ 5.1 GB      │          │
│  │  (Swap on NVMe: 16GB backup)                  │          │
│  └────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

> **TIP**
> Không chạy Gemma 4 trên Jetson → **~5GB free RAM** — rất thoải mái. Có thể chạy tất cả 3 model (YOLO + CNN + RL) **đồng thời** mà không bị memory pressure.

### 10.2 Optimization Recommendations

| Config             | Value                          | Purpose                   |
| ------------------ | ------------------------------ | ------------------------- |
| Power Mode         | **25W** (`sudo nvpmodel -m 0`) | Maximum performance       |
| GUI                | **Disabled** (headless)        | Save ~800MB RAM           |
| Swap               | **16GB on NVMe**               | Safety net                |
| CUDA Lazy Loading  | `CUDA_MODULE_LOADING=LAZY`     | Reduce initial memory     |
| TensorRT Workspace | 256MB max                      | Limit engine build memory |
| Jetson Clocks      | `sudo jetson_clocks`           | Lock CPU/GPU at max freq  |

---

## 11. Deployment & Model Lifecycle

### 11.1 Model Versioning

```
models/
├── yolo/
│   ├── yolo11s_v1.engine        # TensorRT engine
│   └── yolo11s_v1.metadata.json # Input size, classes, metrics
├── cnn_intent/
│   ├── intent_v1.engine
│   ├── intent_v1.onnx           # Backup ONNX
│   └── intent_v1.metadata.json
├── rl_policy/
│   ├── policy_v3.engine         # Active version
│   ├── policy_v2.engine         # Previous (rollback)
│   └── policy_v3.metadata.json
└── active_models.json           # Currently loaded versions
```

### 11.2 Hot-Reload Protocol

```
Laptop exports new model via gRPC
    │
    ▼
Jetson receives ONNX bytes
    │
    ▼
Background thread: ONNX → TensorRT build
    │ (takes 1-3 minutes, does NOT block inference)
    │
    ▼
New .engine saved to models/ directory
    │
    ▼
Atomic swap: old model → new model
    │ (single frame gap, <33ms)
    │
    ├─ Success → Log version change, keep old as rollback
    └─ Failure → Auto-rollback to previous version
```

### 11.3 Docker Setup (Jetson)

```yaml
# docker-compose.yml (Jetson)
version: "3.8"
services:
  ai-server:
    image: context-aware-ai:latest
    runtime: nvidia
    environment:
      - CUDA_MODULE_LOADING=LAZY
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./models:/app/models
      - ./config:/app/config
      - ./logs:/app/logs
    ports:
      - "5555:5555" # ZMQ nav_cmd
      - "5556:5556" # ZMQ detections
      - "5560:5560" # ZMQ robot_state
      - "5570:5570" # ZMQ config
      - "50051:50051" # gRPC
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    restart: unless-stopped
    command: python -u main.py --config /app/config/production.yaml
```

---

## 12. Directory Structure

```
context-aware/                    # Jetson AI Server repo
├── docs/
│   ├── system-design.md          # This document
│   ├── adr/                      # Architecture Decision Records
│   │   ├── adr-001-no-ros2-on-jetson.md
│   │   ├── adr-002-zmq-grpc-hybrid.md
│   │   ├── adr-003-sac-over-ppo.md
│   │   └── adr-004-vlm-guided-reward.md
│   └── api/
│       └── protobuf-reference.md
│
├── src/
│   ├── main.py                   # Entry point, orchestrator
│   ├── config.py                 # Configuration management
│   │
│   ├── perception/               # Detection + Intent pipeline
│   │   ├── __init__.py
│   │   ├── camera.py             # Camera capture (CSI/USB)
│   │   ├── yolo_detector.py      # YOLOv11s TensorRT inference
│   │   ├── tracker.py            # ByteTrack multi-object tracking
│   │   ├── roi_extractor.py      # Crop person ROIs
│   │   └── intent_cnn.py         # Custom CNN intent prediction
│   │
│   ├── navigation/               # RL Policy + Context
│   │   ├── __init__.py
│   │   ├── context_builder.py    # Build observation from detections
│   │   ├── rl_policy.py          # RL policy TensorRT inference
│   │   ├── nav_command.py        # NavigationCommand generation
│   │   └── safety_monitor.py     # Hard safety constraints
│   │
│   ├── communication/            # ZMQ + gRPC
│   │   ├── __init__.py
│   │   ├── zmq_publisher.py      # Publish to RasPi
│   │   ├── zmq_subscriber.py     # Subscribe from RasPi
│   │   ├── grpc_server.py        # Model deployment service
│   │   ├── grpc_client.py        # Experience streaming to Laptop
│   │   └── proto/
│   │       ├── messages.proto
│   │       ├── training_service.proto
│   │       └── *_pb2.py          # Generated
│   │
│   ├── experience/               # Data collection for training
│   │   ├── __init__.py
│   │   ├── collector.py          # Collect experience frames
│   │   ├── buffer.py             # Local ring buffer
│   │   └── streamer.py           # Stream to laptop via gRPC
│   │
│   └── models/                   # Model management
│       ├── __init__.py
│       ├── loader.py             # Load TensorRT engines
│       ├── version_manager.py    # Model versioning + rollback
│       └── converter.py          # ONNX → TensorRT on-device
│
├── training/                     # Scripts to run on LAPTOP
│   ├── train_cnn.py              # CNN intent training
│   ├── train_rl.py               # SAC RL training
│   ├── vlm_evaluator.py          # Gemma 4 reward shaping
│   ├── export_model.py           # PyTorch → ONNX export
│   ├── deploy_model.py           # gRPC deploy to Jetson
│   └── configs/
│       ├── cnn_config.yaml
│       └── rl_config.yaml
│
├── proto/                        # Shared protobuf definitions
│   ├── messages.proto
│   └── training_service.proto
│
├── config/
│   ├── production.yaml           # Jetson production config
│   ├── development.yaml          # Development/testing config
│   └── models.yaml               # Model paths and versions
│
├── models/                       # TensorRT engines (gitignored)
│   ├── yolo/
│   ├── cnn_intent/
│   └── rl_policy/
│
├── scripts/
│   ├── setup_jetson.sh           # JetPack setup + dependencies
│   ├── benchmark.py              # FPS + latency benchmarks
│   └── health_check.py           # System monitoring
│
├── tests/
│   ├── test_perception.py
│   ├── test_navigation.py
│   ├── test_communication.py
│   └── test_integration.py
│
├── Dockerfile                    # Jetson container
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 13. Architecture Decision Records

### ADR-001: Không sử dụng ROS2 trên Jetson

|                |                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------- |
| **Status**     | Accepted                                                                                                      |
| **Context**    | Jetson chạy AI inference thuần túy. ROS2 DDS overhead không cần thiết, tốn ~200-400MB RAM và tăng complexity. |
| **Decision**   | Jetson chạy pure Python/C++ với ZMQ+gRPC. RasPi giữ ROS2 cho SLAM+Nav2.                                       |
| **Trade-offs** | Mất ROS2 ecosystem (rviz, rosbag) trên Jetson, nhưng tiết kiệm RAM và giảm dependency.                        |
| **Mitigation** | ZMQ-ROS2 bridge trên RasPi để chuyển đổi messages.                                                            |

### ADR-002: ZMQ + gRPC Hybrid Communication

|                |                                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| **Status**     | Accepted                                                                                                     |
| **Context**    | Cần real-time (<5ms) cho navigation commands VÀ reliable structured cho model deployment.                    |
| **Decision**   | ZMQ PUB/SUB cho real-time data, gRPC cho structured operations.                                              |
| **Trade-offs** | Hai protocol = phức tạp hơn, nhưng mỗi protocol optimal cho use case riêng.                                  |
| **Rationale**  | ZMQ không có service definition (tự serialize). gRPC quá nặng cho 30Hz data. Hybrid tận dụng ưu điểm cả hai. |

### ADR-003: SAC thay vì PPO cho RL

|                |                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Status**     | Accepted                                                                                                                 |
| **Context**    | Action space liên tục (velocity_scale, heading_offset). Cần sample efficiency vì data thu thập từ robot thực tế tốn kém. |
| **Decision**   | SAC (Soft Actor-Critic) — off-policy, sample efficient, automatic entropy tuning.                                        |
| **Trade-offs** | PPO ổn định hơn nhưng cần nhiều data hơn 5-10x. SAC phức tạp hơn nhưng học nhanh hơn từ replay buffer.                   |
| **Revisit**    | Nếu SAC training không ổn định → chuyển sang PPO.                                                                        |

### ADR-004: VLM-Guided Reward Shaping

|                |                                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**     | Accepted                                                                                                                                    |
| **Context**    | Thiết kế reward function thủ công cho social navigation rất khó và brittle. Gemma 4 VLM có khả năng đánh giá context phong phú từ hình ảnh. |
| **Decision**   | Gemma 4 chạy trên Laptop đánh giá offline/async. Score của VLM là thành phần chính (40%) trong reward function.                             |
| **Trade-offs** | Latency cao (2-5s/evaluation), không thể dùng real-time. Training chậm hơn pure RL. VLM có thể hallucinate.                                 |
| **Mitigation** | Offline batch evaluation. Safety penalties là hard-coded (không phụ thuộc VLM). Human spot-check VLM scores định kỳ.                        |

### ADR-005: Cropped ROI + MobileNetV3 cho Intent Prediction

|                |                                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**     | Accepted                                                                                                                                    |
| **Context**    | Cần nhận diện ý định di chuyển của người từ hình ảnh. Pose estimation (skeleton) chính xác hơn nhưng nặng hơn.                              |
| **Decision**   | Dùng cropped ROI image → MobileNetV3-Small backbone. Không dùng pose estimation riêng.                                                      |
| **Trade-offs** | Mất thông tin skeleton chi tiết, nhưng tiết kiệm ~50MB VRAM và ~5ms latency. ROI vẫn chứa đủ visual features về body orientation và motion. |
| **Revisit**    | Nếu accuracy thấp → thêm YOLOv11s-pose và concatenate skeleton features.                                                                    |

---

## 14. Risk Analysis & Mitigations

### Technical Risks

| Risk                                    | Probability | Impact   | Mitigation                                                               |
| --------------------------------------- | ----------- | -------- | ------------------------------------------------------------------------ |
| VLM reward inconsistent (hallucination) | Medium      | High     | Hard safety penalties override VLM. Batch human review.                  |
| CNN intent accuracy <80%                | Medium      | Medium   | Augment with temporal features. Fall back to distance-based heuristic.   |
| ZMQ message loss (WiFi)                 | Low         | High     | Heartbeat monitoring. Auto-STOP if RasPi unreachable >500ms.             |
| TensorRT build fails on new ONNX        | Low         | Medium   | Keep previous .engine as rollback. Validate ONNX on laptop first.        |
| RL policy diverges                      | Medium      | High     | Clip actions to safe range. Safety monitor overrides dangerous commands. |
| 8GB RAM insufficient                    | Very Low    | Critical | Analyzed: ~3GB used, 5GB free. Swap 16GB backup.                         |

### Safety Architecture

```
┌─────────────────────────────────────────────────┐
│           SAFETY MONITOR (Always Active)         │
│                                                   │
│  Layer 1: RL Policy Output                       │
│    └─ Clip velocity_scale to [0, 1]              │
│    └─ Clip heading_offset to [-π/4, π/4]         │
│                                                   │
│  Layer 2: Safety Monitor (on Jetson)             │
│    └─ IF nearest_person < 0.5m → FORCE STOP      │
│    └─ IF no detection for 3s → CAUTIOUS mode      │
│    └─ IF RL confidence < 0.3 → Use heuristic      │
│                                                   │
│  Layer 3: RasPi Watchdog                          │
│    └─ IF no AI command for 1s → STOP              │
│    └─ IF Nav2 reports stuck → Override to STOP    │
│    └─ Hardware E-STOP → Kill motors immediately   │
│                                                   │
│  Priority: Layer 3 > Layer 2 > Layer 1            │
└─────────────────────────────────────────────────┘
```

---

## Appendix A: Tech Stack Summary

| Component            | Technology                           | Version               |
| -------------------- | ------------------------------------ | --------------------- |
| **Jetson OS**        | Ubuntu 22.04 (JetPack 6.x L4T)       | Latest                |
| **AI Framework**     | PyTorch → TensorRT                   | PyTorch 2.x, TRT 10.x |
| **Object Detection** | Ultralytics YOLOv11s                 | Latest                |
| **CNN Backbone**     | MobileNetV3-Small (torchvision)      | Latest                |
| **RL Framework**     | Stable Baselines3 (training only)    | Latest                |
| **VLM**              | Gemma 4 E4B via Ollama (laptop only) | Latest                |
| **Messaging**        | ZeroMQ (pyzmq)                       | 25.x                  |
| **RPC**              | gRPC + Protobuf                      | 1.60+                 |
| **Serialization**    | Protobuf                             | 4.x                   |
| **Tracking**         | ByteTrack                            | Latest                |
| **Container**        | Docker + NVIDIA Container Runtime    | Latest                |
| **Monitoring**       | tegrastats, jtop                     | -                     |

## Appendix B: Network Configuration

```yaml
# Default network config
jetson:
  ip: 192.168.1.10
  zmq_pub_nav: 5555
  zmq_pub_det: 5556
  zmq_sub_state: 5560
  zmq_rep_config: 5570
  grpc_port: 50051

raspi:
  ip: 192.168.1.20
  zmq_pub_state: 5560
  zmq_sub_nav: 5555
  zmq_sub_det: 5556
  zmq_req_config: 5570

laptop:
  ip: 192.168.1.100
  grpc_port: 50052 # VLM + Training services
  ollama_port: 11434 # Ollama API
```

---

## 15. Bottleneck Analysis & Mitigation

> **Key Insight**: Vấn đề khó sửa nhất KHÔNG phải model — mà là **data quality**, **state representation**, và **system interfaces**.

### 15.1 Bottleneck Classification Framework

| Category | Component Type                       | Fixability | Cost of Change | Priority |
| -------- | ------------------------------------ | ---------- | -------------- | -------- |
| **A**    | Model-level (CNN, RL weights)        | Easy       | Low            | Medium   |
| **B**    | Representation-level (state, reward) | Moderate   | Medium         | High     |
| **C**    | System-level (data, architecture)    | Hard       | Very High      | Critical |

### 15.2 Category A — Model-Level (LOW RISK, Iterate Later)

#### A1. CNN Intent Prediction

| Aspect         | Detail                                                                                                                      |
| -------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Issues**     | Accuracy <80%, weak temporal modeling, sensitivity to viewpoint/lighting                                                    |
| **Impact**     | Local misclassification → RL receives noisy intent signals                                                                  |
| **Mitigation** | Improve dataset diversity, replace backbone (MobileNet → EfficientNet), add temporal module (Conv1D/GRU), data augmentation |
| **Conclusion** | ✅ Safe to iterate after deployment                                                                                         |

#### A2. RL Policy Optimization

| Aspect         | Detail                                                                          |
| -------------- | ------------------------------------------------------------------------------- |
| **Issues**     | Suboptimal policy convergence, training instability                             |
| **Impact**     | Inefficient navigation, non-smooth trajectories                                 |
| **Mitigation** | Hyperparameter tuning, algorithm switch (SAC ↔ PPO), replay buffer improvements |
| **Conclusion** | ✅ RL is inherently iterative → safe to refine later                            |

#### A3. VLM Prompt Engineering

| Aspect         | Detail                                                      |
| -------------- | ----------------------------------------------------------- |
| **Issues**     | Inconsistent scoring, prompt sensitivity                    |
| **Mitigation** | Prompt refinement, temperature tuning, output normalization |
| **Conclusion** | ✅ Low-cost improvement → safe to delay                     |

### 15.3 Category B — Representation-Level (MEDIUM RISK)

#### B1. State Representation (⚠️ CRITICAL — Design Early)

**Current limitation**: Only snapshot-based observation, no temporal encoding.

| Aspect          | Detail                                                            |
| --------------- | ----------------------------------------------------------------- |
| **Impact**      | Reactive behavior only, no anticipation capability                |
| **Risk**        | Requires **full RL retraining** if changed later                  |
| **Fix**         | Stack last k observations (k=3–5) OR lightweight temporal encoder |
| **Requirement** | State must be: extensible, versioned, backward-compatible         |

> ⚠️ **WARNING**
> State representation phải được thiết kế **temporal-ready từ ngày đầu**, dù Phase 1 chỉ dùng snapshot. Thay đổi state space sau = retrain toàn bộ RL.

**Implementation — Temporal-Ready State**:

```python
@dataclass
class ObservationSpace:
    # ... existing fields ...

    # Temporal buffer (Phase 1: k=1, Phase 2: k=3-5)
    temporal_stack_size: int = 1  # Configurable, version-tracked
    observation_history: deque    # Ring buffer of past observations

    def get_stacked_observation(self) -> np.ndarray:
        """Returns stacked observations. k=1 = snapshot (backward-compatible)."""
        if self.temporal_stack_size == 1:
            return self.current_observation
        return np.concatenate(list(self.observation_history))
```

#### B2. Reward Function Stability

| Aspect         | Detail                                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Issues**     | VLM inconsistency, prompt bias, non-stationary reward                                                                                 |
| **Impact**     | RL learns unstable or incorrect behavior                                                                                              |
| **Mitigation** | Score normalization, confidence filtering, ensemble evaluation (multi-query averaging), convert to discrete labels (GOOD/BAD/NEUTRAL) |
| **Conclusion** | ⚠️ Can start simple but must stabilize before large-scale training                                                                    |

**Reward Stabilization Protocol**:

```python
def stabilize_vlm_reward(raw_scores: List[float]) -> float:
    """Multi-query averaging + outlier rejection."""
    # Query VLM 3 times with slight prompt variations
    if len(raw_scores) < 2:
        return raw_scores[0]

    # Reject outliers (>2σ from mean)
    mean, std = np.mean(raw_scores), np.std(raw_scores)
    filtered = [s for s in raw_scores if abs(s - mean) < 2 * std]

    # Discretize for stability: GOOD(+0.7) / NEUTRAL(0) / BAD(-0.7)
    avg = np.mean(filtered)
    if avg > 0.3: return 0.7    # GOOD
    if avg < -0.3: return -0.7  # BAD
    return 0.0                  # NEUTRAL
```

#### B3. Human Interaction Modeling (→ GAP #3)

| Aspect         | Detail                                                                                                            |
| -------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Issues**     | Independent per-person intent, no crowd interaction modeling, no trajectory conditioning on robot actions          |
| **Impact**     | Poor performance in crowded environments. Robot cannot anticipate how humans react to its own actions.            |
| **Mitigation** | Intent-conditioned trajectory prediction: predict human trajectory given robot action. See [GAP #3](#gap-3--intent-aware-human-interaction). |
| **Conclusion** | ⚠️ Phase 4 adds intent-conditioned trajectory prediction, Phase 5 evaluates in ablation study.                   |

### 15.4 Category C — System-Level (CRITICAL — Must Be Correct From Day 1)

#### C1. Data Logging Pipeline (🔴 MOST CRITICAL)

**Required data per frame** — ALL must be logged consistently:

| Field                | Type          | Purpose                        |
| -------------------- | ------------- | ------------------------------ |
| `raw_image`          | JPEG bytes    | VLM evaluation, CNN retraining |
| `detections`         | Protobuf list | Debugging, ablation study      |
| `intent_predictions` | Float array   | CNN performance tracking       |
| `observation_vector` | Float array   | RL replay buffer               |
| `action_taken`       | Float array   | RL replay buffer               |
| `robot_state`        | Protobuf      | Context reconstruction         |
| `timestamp`          | Float64       | Temporal alignment             |

> 🔴 **CAUTION**
> Nếu data logging không đúng từ đầu:
>
> - ❌ Không thể retrain RL
> - ❌ Không thể debug failures
> - ❌ Không thể thực hiện ablation study
> - ❌ Không thể publish research paper

**Logging Requirements**:

- Consistent format (HDF5 or structured directory)
- Lossless (or controlled JPEG quality ≥ 85)
- Timestamp-aligned across all fields
- Ring buffer on Jetson (last 10K frames) + batch upload to Laptop

#### C2. System Architecture Modularity

| Aspect         | Detail                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| **Issues**     | Tight coupling between modules, non-modular pipeline                                                    |
| **Impact**     | Hard to upgrade components, high refactor cost                                                          |
| **Mitigation** | Strict module boundaries (perception → state builder → RL → communication), clear interfaces (Protobuf) |
| **Conclusion** | 🔴 Architecture errors are extremely costly → must be correct early                                     |

**Module Contract Rules**:

```
✅ Modules communicate ONLY through defined interfaces
✅ Each module can be replaced independently
✅ Protobuf defines all inter-module data contracts
❌ No module directly accesses another module's internals
❌ No shared mutable state between modules
```

#### C3. Safety Layer

| Aspect         | Detail                                                                                     |
| -------------- | ------------------------------------------------------------------------------------------ |
| **Required**   | Hard stop when distance < threshold, fallback when RL confidence low, watchdog timeout     |
| **Impact**     | Unsafe behavior → real-world deployment risk                                               |
| **Conclusion** | 🔴 Cannot be learned → must be engineered. See [Safety Architecture](#safety-architecture) |

### 15.5 Priority Summary

```
🔴 MUST FIX EARLY (Before any training)
├── 1. Data logging pipeline (C1)
├── 2. State representation — temporal-ready (B1)
└── 3. Safety layer (C3)

🟡 SHOULD IMPROVE (Before final system)
├── 1. Reward stabilization (B2)
└── 2. Interaction modeling (B3)

🟢 CAN ITERATE LATER (After deployment)
├── 1. CNN architecture (A1)
├── 2. RL tuning (A2)
└── 3. VLM prompt (A3)
```

---

## 16. Research Gap Analysis & Novel Contributions

> **Thesis**: *"We propose a temporally-aware, VLM-guided reward shaping framework with state alignment, validated on a real-world edge-cloud robotic system."*
>
> This section analyzes gaps in the state-of-the-art and positions our contributions precisely. We aim for **focused, defensible novelty** rather than breadth.

### GAP #1 — VLM → Reward: Structured Scalarization

**Problem trong literature hiện tại:**

```
Current approach (all prior work):

  Image + Context ──► VLM ──► score: float

  → Mất: reasoning, explanation, structure
  → VLM output bị nén thành 1 scalar
  → RL không biết TẠI SAO reward cao/thấp
```

**Gap**: Prior VLM-guided RL systems collapse VLM output into a single opaque scalar, losing the reasoning structure that makes VLM evaluation valuable.

**Our solution — Structured Scalarization:**

```
Our approach:

  Image + Context ──► VLM ──► {
                                safety_score,
                                social_score,
                                efficiency_score
                              }
                     ──► R_vlm = w_s·safety + w_soc·social + w_e·efficiency
                         (scalar ∈ ℝ — standard RL compatible)

  → Structured intermediate, scalar final
  → Sub-weights adjustable per training phase
  → Interpretable: can diagnose which axis is weak
```

> **RL Theory Compliance**: Final reward `R_t ∈ ℝ` is always scalar per standard MDP formulation (Sutton & Barto, 2018). The structured VLM output is an **intermediate decomposition** — we use fixed-weight scalarization, NOT multi-objective RL (no Pareto front, no MORL optimization).

**Contribution**: We extend prior VLM-guided RL by preserving the reasoning decomposition of VLM evaluations through structured scalarization, enabling interpretable reward diagnosis and phase-specific weight tuning.

**Implementation**: See [Reward Function](#reward-function--structured-vlm-guided-with-temporal-intent-gap-1--2) in Section 6.

---

### GAP #2 — Temporal Intent in Reward Shaping

**Problem:**

```
Current systems:

  CNN intent (single frame) ──► reward at time t

  → Reward chỉ dùng intent tại thời điểm t
  → Không đánh giá khả năng ANTICIPATE
  → Robot chỉ reactive, không proactive
```

**Gap**: Intent prediction exists in literature but **does not influence reward temporally**. No existing system uses intent trajectory for reward shaping.

**Our solution — Discounted Temporal Intent Risk:**

```
Formal definition:

  R_intent(t) = Σ_{i=0}^{k-1} γ^i · risk(intent_{t+i})

  Parameters (fixed, not learned):
    k = 5 frames      (~167ms lookahead at 30 FPS)
    γ = 0.9            (temporal discount)
    risk(·) ∈ [-1, 1]  (per-class risk mapping, see Section 6)

  Properties:
    • Non-differentiable (shaped reward, not a loss function)
    • Computed from CNN intent predictions (already available)
    • Penalizes late reaction to high-risk intents (CROSSING, ERRATIC)
    • SAC treats it as any other scalar reward component

  Example:
    intent trajectory: [STAT, STAT, APPR, CROSS, CROSS]
    risk trajectory:   [0.0,  0.0, -0.3, -0.6,  -0.6]
    R_intent = 0.0 + 0.9·0.0 + 0.81·(-0.3) + 0.73·(-0.6) + 0.66·(-0.6)
             = -1.08  → significant penalty for approaching crossing
```

**Contribution**: We extend VLM-guided reward shaping with a temporal intent penalty term that rewards proactive behavior (slowing down *before* a crossing occurs) rather than reactive behavior.

**Implementation**: See `compute_temporal_intent_reward()` in Section 6.

---

### GAP #3 — Intent-Aware Human Interaction

**Problem:**

```
All current approaches:

  Human = passive object with intent label
  Robot acts ──► Human does NOT react to robot

  Reality:
  Robot acts ──► Human REACTS ──► Robot trajectory affected
```

**Gap**: Robot không model **phản ứng của con người** đối với hành động của robot. Human modeling hoàn toàn open-loop.

**Our solution — Intent-Conditioned Trajectory Prediction (scoped)**:

> **Scope decision**: Full game-theoretic modeling (Nash equilibrium, dual-policy optimization) is a separate research problem requiring extensive data and computation. We intentionally scope GAP #3 to a tractable extension:

```
Our approach (Phase 4):

  ┌────────────────────────────────────────────────┐
  │  Intent-Conditioned Trajectory Prediction       │
  │                                                 │
  │  Input:                                         │
  │    • person_intent[t:t-5]  (intent history)     │
  │    • robot_action[t]       (current action)     │
  │    • relative_position     (person vs robot)    │
  │                                                 │
  │  Output:                                        │
  │    • predicted_trajectory[t+1:t+5]  (5 frames)  │
  │    • collision_probability  (scalar)             │
  │                                                 │
  │  Method: Lightweight MLP (3 layers)             │
  │  Training: from Phase 1-3 deployment data       │
  │  Integration: collision_prob → R_safety bonus    │
  └────────────────────────────────────────────────┘

What we do NOT attempt:
  ✗ Game-theoretic dual-policy optimization
  ✗ Nash equilibrium computation
  ✗ Full closed-loop interactive RL
  (These are separate research contributions)
```

**Contribution**: We extend intent prediction with action-conditioned trajectory forecasting — predicting how a person's trajectory changes given the robot's current action. This is a lightweight, data-driven alternative to full game-theoretic modeling.

**Data requirement**: Interaction dataset collected from Phase 1-3 deployment (robot actions + tracked human positions).

---

### GAP #4 — VLM-RL State Alignment

**Problem:**

```
Current architecture:

  VLM nhìn IMAGE ──► text/score    (runs at ~100-500ms)
  RL  dùng VECTOR ──► action       (runs at ~16ms)

  → Semantic gap: VLM "hiểu" scene nhưng RL không thấy
  → Latency mismatch: VLM 100x slower than control loop
  → Distribution shift: embedding stale, state changes fast
```

**Gap**: Chưa có mapping từ VLM hidden representation → RL state space, and no handling of the latency mismatch between VLM inference and control loop.

**Our solution — VLM State Projection with Staleness Decay:**

```python
# Bridge: VLM visual space → RL numeric space
vlm_embedding = VLMStateAligner.encode(image)   # [64-dim]
w = aligner.staleness_weight                     # exponential decay
state = np.concatenate([
    scene_features,              # [6]  from YOLO
    occupancy_grid,              # [64] from detections
    person_intents,              # [24] from CNN
    robot_state,                 # [8]  vx, vy, vθ, prev_action
    vlm_embedding * w,           # [64] weighted by staleness
    [w],                         # [1]  staleness weight itself
])                               # Total: ~167 floats
```

**Key design decisions:**

| Decision | Choice | Rationale |
|---|---|---|
| Projection dim | 64 | Balance information vs RL input size |
| Update frequency | Async (2-5s) | VLM too slow for real-time |
| Staleness handling | Exponential decay (half-life=2s) | Stale embeddings progressively ignored |
| Training | End-to-end with RL | Projector learns what VLM features matter |
| Fallback | Zero vector + w=0 if VLM unavailable | RL must work without VLM in production |

**Contribution**: We extend VLM-guided RL with a state alignment mechanism that projects VLM hidden representations into the RL observation space, with explicit staleness-aware weighting to handle the latency mismatch between VLM inference (~500ms) and the control loop (~16ms).

**Implementation**: See `VLMStateAligner` class in [State Space](#state-space-observation--vlm-aligned).

---

### GAP #5 — Real-World Validation Pipeline

**Problem trong literature:**

```
Most research papers:

  Train in simulation ──► Report metrics ──► Done

  Missing:
  ✗ Real hardware deployment pipeline
  ✗ Continuous learning loop
  ✗ Production safety guarantees
```

**Gap**: Literature thiếu **real-world validation** of VLM-guided social navigation trên phần cứng thực.

> **Positioning**: This is an **engineering contribution** (real-world validation), NOT an algorithmic novelty. It strengthens the paper by demonstrating that GAP #1-#4 solutions work on commodity hardware, not just in simulation.

**Our solution — Edge-Cloud Continuous Learning Pipeline:**

```
┌─────────────────────────────────────────────────────────────────┐
│              CONTINUOUS LEARNING LOOP (GAP #5)                   │
│                                                                  │
│   JETSON (Edge)                SERVER (Training)                 │
│   ─────────────                ──────────────────                 │
│                                                                  │
│   1. Run inference     ──►  2. Collect experience data           │
│      (YOLO+CNN+RL)             (ROI batches via rsync)           │
│         │                         │                              │
│         │                         ▼                              │
│         │                  3. VLM evaluates decisions             │
│         │                     (structured reward signal)          │
│         │                         │                              │
│         │                         ▼                              │
│         │                  4. Train RL + fine-tune CNN            │
│         │                     (SAC + continual learning)          │
│         │                         │                              │
│         │                         ▼                              │
│   6. Auto-update       ◄── 5. Export models                      │
│      (git pull + restart)      (.pt → deploy)                    │
│         │                                                        │
│         ▼                                                        │
│   7. Improved policy   ──►  Loop back to step 1                  │
│      running on Jetson                                           │
└─────────────────────────────────────────────────────────────────┘
```

**Contribution**: We provide a real-world validation of VLM-guided RL in an edge-cloud robotic system, demonstrating that the proposed reward shaping framework (GAP #1-#4) transfers from theory to commodity hardware (Jetson Orin Nano 8GB + consumer GPU server).

**Already implemented**: Docker infrastructure, auto-updater, ROI collection pipeline, training server, Edge API.

---

### Research Contribution Summary

| GAP | Problem | Our Solution | Type | Phase |
|-----|---------|-------------|------|-------|
| #1 | VLM→opaque scalar | Structured scalarization (safety/social/efficiency) | Algorithmic | Phase 3 |
| #2 | Temporal intent unused | R_intent = Σ γ^i · risk(intent_{t+i}), k=5, γ=0.9 | Algorithmic | Phase 3 |
| #3 | Open-loop human model | Intent-conditioned trajectory prediction | Algorithmic | Phase 4 |
| #4 | VLM-RL semantic gap | State projection + staleness decay | Algorithmic | Phase 3 |
| #5 | Simulation-only validation | Edge-cloud pipeline on real hardware | Engineering | Phase 1-3 |

> **Publication angle**: *"Temporally-Aware VLM-Guided Reward Shaping with State Alignment for Context-Aware Social Navigation"* — core algorithmic contributions are GAP #1, #2, #4. GAP #3 is a tractable extension. GAP #5 provides real-world validation that elevates the work above simulation-only papers.

## 17. Development Strategy

> **Strategy**: Start simple → Collect data → Iterate → Refine.
> Never compromise on: **data logging**, **safety**, **architectural modularity**.
> Each phase now explicitly addresses identified GAPs.

### Phase 1 — Foundation + Deployment Pipeline (Week 1-3) — GAP #5

```
┌──────────────────────────────────────────────────────────┐
│  PHASE 1: Foundation                                     │
│  GAP addressed: #5 (System-level deployment pipeline)    │
│                                                          │
│  ✅ YOLO + ByteTrack (detection + tracking)              │
│  ✅ Simple CNN (no temporal, snapshot only)               │
│  ✅ Heuristic navigation policy (rule-based, no RL)      │
│  ✅ Full data logging pipeline (C1 — CRITICAL)           │
│  ✅ ZMQ communication to RasPi                           │
│  ✅ Safety layer (C3 — CRITICAL)                         │
│  ✅ State representation with temporal-ready design (B1)  │
│  ✅ Docker isolation (Jetson + Server) — GAP #5          │
│  ✅ Auto-updater daemon (git poll → rebuild) — GAP #5    │
│  ✅ Edge API with /logs endpoint — GAP #5                │
│  ✅ ROI collection + rsync pipeline — GAP #5             │
│                                                          │
│  Deliverable: Robot navigates with rules, logs all data, │
│  full CI/CD pipeline operational                         │
└──────────────────────────────────────────────────────────┘
```

### Phase 2 — Perception + Temporal Intent (Week 4-5) — GAP #2 Prep

```
┌──────────────────────────────────────────────────────────┐
│  PHASE 2: Perception Upgrade                             │
│  GAP addressed: #2 prep (temporal intent infrastructure) │
│                                                          │
│  ✅ Add temporal state stacking (k=3-5)                  │
│  ✅ Improve CNN with temporal aggregator (Conv1D/GRU)    │
│  ✅ Intent trajectory buffer per track_id — GAP #2 prep │
│  ✅ gRPC setup for Server communication                  │
│  ✅ Experience streaming to Server                       │
│  ✅ Continual learning pipeline (EWC + Replay Buffer)    │
│                                                          │
│  Deliverable: Temporal perception, data pipeline to      │
│  Server, intent trajectory logging enabled               │
└──────────────────────────────────────────────────────────┘
```

### Phase 3 — VLM-Guided RL + Core Contributions (Week 6-8) — GAP #1, #2, #4

```
┌──────────────────────────────────────────────────────────┐
│  PHASE 3: RL + VLM Integration                           │
│  GAPs addressed: #1, #2, #4 (core algorithmic)           │
│                                                          │
│  ✅ Structured VLM reward scalarization (GAP #1)         │
│     → VLM outputs {safety, social, efficiency} scores    │
│     → Scalarized via fixed weights into R_vlm ∈ ℝ       │
│  ✅ Temporal intent reward R_intent (GAP #2)             │
│     → Σ γ^i · risk(intent_{t+i}), k=5, γ=0.9           │
│     → Penalize late reaction to CROSSING/ERRATIC         │
│  ✅ VLM-RL state alignment (GAP #4)                      │
│     → VLMStateAligner: image → 64-dim with staleness     │
│     → w = 2^(-age/2s) exponential decay                  │
│     → End-to-end projector training with SAC             │
│  ✅ Reward stabilization protocol (B2)                   │
│  ✅ Train SAC with composite scalar reward               │
│  ✅ Deploy first RL policy to Jetson                     │
│  ✅ Model hot-reload via gRPC                            │
│                                                          │
│  Deliverable: Robot navigates with VLM-guided RL policy  │
│  using structured reward + temporal intent + VLM state   │
└──────────────────────────────────────────────────────────┘
```

### Phase 4 — Intent-Aware Interaction (Week 9-12) — GAP #3

```
┌──────────────────────────────────────────────────────────┐
│  PHASE 4: Interaction Modeling                            │
│  GAP addressed: #3 (intent-conditioned prediction)       │
│                                                          │
│  ✅ Intent-conditioned trajectory prediction — GAP #3    │
│     → Predict human trajectory given robot action         │
│     → Lightweight MLP (3 layers), trained from Ph1-3 data│
│  ✅ Collision probability → R_safety bonus               │
│     → Higher predicted collision prob → larger penalty    │
│  ✅ Policy refinement (iterative SAC training)           │
│  ✅ CNN backbone upgrade if needed (A1)                  │
│  ✅ Performance benchmarking + optimization              │
│                                                          │
│  Deliverable: Robot uses predicted human trajectories     │
│  to plan safer navigation in crowds                      │
└──────────────────────────────────────────────────────────┘
```

### Phase 5 — Research & Publication (Week 13+)

```
┌──────────────────────────────────────────────────────────┐
│  PHASE 5: Research Output                                │
│                                                          │
│  ✅ Ablation study                                       │
│     → GAP #1: opaque scalar vs structured scalarization  │
│     → GAP #2: with/without temporal intent in reward     │
│     → GAP #3: with/without intent-conditioned prediction │
│     → GAP #4: with/without VLM state embedding           │
│  ✅ Benchmark vs baselines (DWB, TEB, pure RL, SARL)     │
│  ✅ Real-world deployment metrics (GAP #5 validation)    │
│  ✅ Document results & write paper                       │
│  ✅ Open-source full pipeline                            │
│                                                          │
│  Deliverable: Research paper with focused contributions  │
│  + demonstration video + open-source release             │
└──────────────────────────────────────────────────────────┘
```

### Development Phase Dependencies (Updated)

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
 GAP #5      GAP #2p     GAP #1       GAP #3     Ablation
 (infra)     (temporal)  GAP #2       (traj)     Benchmark
                         GAP #4                  Publish
  │            │            │            │            │
  │            │            │            │            └─ All GAPs evaluated
  │            │            │            └─ Depends on: stable RL (Ph3)
  │            │            └─ Depends on: temporal data (Ph2)
  │            └─ Depends on: data pipeline (Ph1)
  └─ FOUNDATION: deployment + safety + data logging
```

---

> **Next Steps**: Review this design → Approve → Continue **Phase 2** (temporal perception + intent trajectory) and prepare **Phase 3** (core algorithmic contributions: GAP #1, #2, #4).

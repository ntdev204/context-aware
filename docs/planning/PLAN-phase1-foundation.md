# PLAN — Phase 1: Foundation (Minimal Viable System)

> **Project**: Context-Aware Navigation for Mecanum Robot
> **Phase**: 1 of 5
> **Timeline**: Week 1–3
> **Status**: 📋 Planning
> **Dependencies**: None (this is the foundation)
> **Ref**: [system-design.md](./system-design.md) — Section 16, Phase 1

---

## Mục tiêu

Xây dựng hệ thống nền tảng hoàn chỉnh trên Jetson: detection + tracking + CNN intent (snapshot) + heuristic navigation + data logging + safety layer + ZMQ communication tới RasPi.

> ⚠️ **WARNING**
> Phase này KHÔNG có RL. Robot điều hướng bằng **rule-based heuristic**. Mục đích chính: thu data đúng + an toàn + modular.

---

## Deliverables

| #   | Deliverable           | Mô tả                                    | Bottleneck |
| --- | --------------------- | ---------------------------------------- | ---------- |
| D1  | Perception Pipeline   | YOLO + ByteTrack + ROI Extraction        | —          |
| D2  | CNN Intent Prediction | MobileNetV3-Small, snapshot mode (k=1)   | —          |
| D3  | Context State Builder | Temporal-ready observation (k=1 ban đầu) | B1         |
| D4  | Heuristic Navigation  | Rule-based policy thay cho RL            | —          |
| D5  | Data Logging Pipeline | Full-frame logging, lossless, aligned    | C1 🔴      |
| D6  | Safety Layer          | 3-layer safety monitor                   | C3 🔴      |
| D7  | ZMQ Communication     | Jetson ↔ RasPi real-time messaging       | C2         |
| D8  | Project Structure     | Modular codebase, Protobuf, config       | C2         |

---

## Task Breakdown

### 1. Project Setup & Structure

- [ ] Khởi tạo project structure theo [system-design.md Section 12](./system-design.md#12-directory-structure)
- [ ] Setup `pyproject.toml` + `requirements.txt` với dependencies:
  - `ultralytics`, `opencv-python`, `pyzmq`, `protobuf`, `grpcio`, `numpy`
- [ ] Tạo `src/config.py` — YAML-based config management
  - Config paths: `config/development.yaml`, `config/production.yaml`
- [ ] Tạo `src/main.py` — Entry point, pipeline orchestrator
- [ ] Setup logging (Python `logging` module, structured format)

**Files**:

```
src/main.py
src/config.py
config/development.yaml
config/production.yaml
pyproject.toml
requirements.txt
```

---

### 2. Protobuf Definitions

- [ ] Viết `proto/messages.proto` — NavigationCommand, Detection, DetectionList, RobotState
- [ ] Viết `proto/training_service.proto` — ExperienceFrame, ModelArtifact (chuẩn bị cho Phase 2)
- [ ] Generate Python code: `*_pb2.py`, `*_pb2_grpc.py`
- [ ] Tạo script `scripts/generate_proto.sh` để auto-generate

**Ref**: system-design.md Section 7.3, 7.4

**Files**:

```
proto/messages.proto
proto/training_service.proto
src/communication/proto/  (generated)
scripts/generate_proto.sh
```

---

### 3. Perception Pipeline (D1)

#### 3.1 Camera Capture

- [ ] `src/perception/camera.py`
  - CSI / USB camera support via OpenCV
  - Target resolution: 1080p → resize to 640×640 cho YOLO
  - Frame rate control: 30 FPS cap
  - Thread-safe frame buffer (double buffering)

#### 3.2 YOLO Detector

- [ ] `src/perception/yolo_detector.py`
  - Load YOLOv11s `.engine` (TensorRT FP16)
  - Fallback: load `.pt` cho development mode
  - Input: 640×640 RGB
  - Output: `List[DetectionResult]` (bbox, class_id, confidence)
  - Classes: `person`, `obstacle`, `door`, `wall`, `free_space`
  - Target: 30+ FPS trên Jetson

#### 3.3 Tracker

- [ ] `src/perception/tracker.py`
  - ByteTrack integration cho stable `track_id`
  - Config: max_age, min_hits, iou_threshold
  - Output: `List[TrackedDetection]` (bbox, class_id, confidence, track_id)

#### 3.4 ROI Extractor

- [ ] `src/perception/roi_extractor.py`
  - Crop person ROIs từ frame
  - Padding 10% context (bbox expansion)
  - Resize to CNN input size: 128×256×3
  - Output: `List[PersonROI]` (image, bbox, track_id, relative_position)

**Ref**: system-design.md Section 4

**Files**:

```
src/perception/__init__.py
src/perception/camera.py
src/perception/yolo_detector.py
src/perception/tracker.py
src/perception/roi_extractor.py
```

---

### 4. CNN Intent Prediction — Snapshot Mode (D2)

- [ ] `src/perception/intent_cnn.py`
  - Load MobileNetV3-Small + Intent Head + Direction Head
  - Phase 1: snapshot mode (1 frame per person, no temporal)
  - Input: 128×256×3 (single cropped ROI)
  - Output: `IntentPrediction` (6 intent probs + dx, dy direction vector)
  - Batch inference: tối đa 5 persons cùng lúc
  - TensorRT `.engine` (FP16) hoặc PyTorch fallback

**Intent Classes** (runtime 6 slots; 5 trainable + abstain):
| ID | Intent | Description |
|----|--------|-------------|
| 0 | STATIONARY | Người đứng yên |
| 1 | APPROACHING | Người đi tới robot |
| 2 | DEPARTING | Người đi xa robot |
| 3 | CROSSING | Người cắt ngang |
| 4 | ERRATIC | Chuyển động bất thường |
| 5 | UNCERTAIN | Abstain / cần review |

> **NOTE**: Phase 1 chưa có trained CNN weights → dùng pretrained MobileNetV3 + random heads (for pipeline testing). Training CNN thực sự ở Phase 2 sau khi có data.

**Ref**: system-design.md Section 5

**Files**:

```
src/perception/intent_cnn.py
```

---

### 5. Context State Builder — Temporal-Ready (D3) ⚠️

- [ ] `src/navigation/context_builder.py`
  - Build `ObservationSpace` từ detections + intents + robot state
  - **CRITICAL**: Thiết kế temporal-ready từ đầu (Bottleneck B1)
    - `temporal_stack_size = 1` (Phase 1, snapshot)
    - Nhưng kiến trúc hỗ trợ `k = 3-5` (Phase 2)
    - `observation_history: deque(maxlen=k)` — ring buffer
    - Versioned: `state_version: str` trong observation
  - Observation vector (~102 floats):
    - Scene encoding: 6 floats (num_persons, nearest distances/angles, free_space_ratio)
    - Occupancy grid: 64 floats (8×8 grid từ detections)
    - Human intents: 24 floats (top-3 persons × 8 features)
    - Robot state: 3 floats (vx, vy, vθ)
    - Previous action: 5 floats
  - `get_stacked_observation()` method — backward-compatible

**Ref**: system-design.md Section 6.1, Bottleneck B1

**Files**:

```
src/navigation/__init__.py
src/navigation/context_builder.py
```

---

### 6. Heuristic Navigation Policy (D4)

- [ ] `src/navigation/heuristic_policy.py` (**NEW** — không có trong directory structure gốc)
  - Rule-based navigation thay cho RL Policy
  - Input: `ObservationSpace`
  - Output: `NavigationCommand` (mode, velocity_scale, heading_offset)
  - Logic rules:

  ```
  if free_space_ratio > 0.8 AND no persons nearby:
      mode = CRUISE, velocity = 1.0
  elif persons detected BUT not blocking:
      mode = CAUTIOUS, velocity = 0.6
  elif nearest_person < SAFETY_DISTANCE:
      mode = STOP, velocity = 0.0
  elif person crossing path:
      mode = AVOID, velocity = 0.3, heading_offset = avoid_direction
  ```

- [ ] `src/navigation/nav_command.py`
  - `NavigationCommand` dataclass generation
  - Serialize to Protobuf cho ZMQ publish

**Files**:

```
src/navigation/heuristic_policy.py
src/navigation/nav_command.py
```

---

### 7. Data Logging Pipeline (D5) 🔴 CRITICAL

> 🔴 **CAUTION**
> Đây là bottleneck C1 — MOST CRITICAL. Sai ở đây = không thể train RL, không thể debug, không thể publish.

- [ ] `src/experience/collector.py`
  - Thu thập tất cả data mỗi frame:

  | Field                | Type                      | Source             |
  | -------------------- | ------------------------- | ------------------ |
  | `raw_image`          | JPEG bytes (quality ≥ 85) | Camera             |
  | `detections`         | Protobuf list             | YOLO + tracker     |
  | `intent_predictions` | Float array               | CNN                |
  | `observation_vector` | Float array (102 dims)    | Context builder    |
  | `action_taken`       | Float array (7 dims)      | Heuristic policy   |
  | `robot_state`        | Protobuf                  | ZMQ from RasPi     |
  | `timestamp`          | Float64 (monotonic)       | System clock       |
  | `frame_id`           | Int                       | Sequential counter |

- [ ] `src/experience/buffer.py`
  - Ring buffer on Jetson: last 10,000 frames
  - Thread-safe: producer (inference loop) / consumer (upload)
  - Memory-efficient: JPEG compressed images
  - Overflow strategy: drop oldest frames

- [ ] Storage format selection:
  - **Option A**: HDF5 files (structured, fast random access)
  - **Option B**: Directory-based (1 dir per episode, JSON + JPEG files)
  - **Recommendation**: HDF5 cho production, Directory cho debugging

- [ ] Timestamp alignment verification
  - All fields from same frame share exact timestamp
  - Monotonic clock (`time.monotonic()`) cho ordering
  - Wall clock (`time.time()`) cho human readability

**Ref**: system-design.md Bottleneck C1

**Files**:

```
src/experience/__init__.py
src/experience/collector.py
src/experience/buffer.py
```

---

### 8. Safety Layer (D6) 🔴 CRITICAL

> 🔴 **CAUTION**
> Safety CANNOT be learned — must be engineered. Bottleneck C3.

- [ ] `src/navigation/heuristic_policy.py`
  - **Layer 1: Output Clipping** (trong navigation policy)
    - `velocity_scale` clamp [0.0, 1.0]
    - `heading_offset` clamp [-π/4, π/4]
  - **Layer 2: Safety Monitor** (trên Jetson)
    - Hard stop nếu `nearest_person_distance < 0.5m`
    - Hard stop nếu `nearest_obstacle_distance < 0.3m`
    - Giảm 50% velocity nếu `nearest_person_distance < 1.0m`
    - Override mode to STOP nếu phát hiện ERRATIC intent
  - **Layer 3: Watchdog** (chuẩn bị cho RasPi side)
    - Heartbeat check: nếu không nhận command trong 500ms → STOP
    - Battery check: nếu battery < 10% → return home / STOP

- [ ] Safety monitor chạy TRƯỚC khi publish NavigationCommand qua ZMQ
  - Input: raw NavigationCommand + detections + observation
  - Output: safe NavigationCommand (có thể bị override)

**Ref**: system-design.md Section 14 (Safety Architecture)

**Files**:

```
src/navigation/heuristic_policy.py
```

---

### 9. ZMQ Communication — Jetson Side (D7)

- [ ] `src/communication/zmq_publisher.py`
  - PUB `:5555` — `ai/nav_cmd` (NavigationCommand @ 30Hz)
  - PUB `:5556` — `ai/detections` (DetectionList @ 10Hz)
  - Protobuf serialization
  - Non-blocking send

- [ ] `src/communication/zmq_subscriber.py`
  - SUB `:5560` — `robot/state` (RobotState @ 20Hz)
  - Protobuf deserialization
  - Timeout handling: nếu không nhận RobotState > 500ms → trigger safety

- [ ] ZMQ socket configuration:
  - `ZMQ_SNDHWM = 2` (drop old if slow consumer)
  - `ZMQ_RCVHWM = 2`
  - `ZMQ_LINGER = 0` (no hung sockets on shutdown)
  - `ZMQ_CONFLATE = 1` (chỉ giữ message mới nhất)

**Ref**: system-design.md Section 7.1, 7.2

**Files**:

```
src/communication/__init__.py
src/communication/zmq_publisher.py
src/communication/zmq_subscriber.py
```

---

### 10. Main Pipeline Orchestrator

- [ ] `src/main.py` — Inference loop chính:

  ```
  while running:
      1. Camera grab frame (~2ms)
      2. YOLO detection (~6ms)
      3. ByteTrack update (~1ms)
      4. ROI extraction (~1ms)
      5. CNN intent prediction (~3ms)
      6. Context state build (~1ms)
      7. Heuristic policy decision (~0.5ms)
      8. Safety monitor check (~0.5ms)
      9. ZMQ publish NavigationCommand (~1ms)
      10. Data logging (async, non-blocking)

      Total target: ~16ms per frame → 60 FPS headroom → 30 FPS target
  ```

- [ ] Threading model:
  - **Main thread**: inference loop (steps 1-9)
  - **Camera thread**: capture + double buffer
  - **Logging thread**: async data writing
  - **ZMQ subscriber thread**: receive RobotState

- [ ] Graceful shutdown: signal handler, cleanup resources

**Files**:

```
src/main.py
```

---

### 11. Testing

- [ ] `tests/test_perception.py`
  - Test YOLO detector với sample images
  - Test ROI extraction logic
  - Test CNN forward pass (random weights)
  - Test tracker ID consistency

- [ ] `tests/test_navigation.py`
  - Test context builder observation shape
  - Test heuristic policy rules
  - Test safety monitor overrides

- [ ] `tests/test_communication.py`
  - Test ZMQ PUB/SUB loopback
  - Test Protobuf serialization/deserialization

- [ ] `tests/test_data_logging.py` 🔴
  - Test data completeness (all fields present)
  - Test timestamp alignment
  - Test ring buffer overflow behavior
  - Test data replayability (load + verify)

**Files**:

```
tests/test_perception.py
tests/test_navigation.py
tests/test_communication.py
tests/test_data_logging.py
```

---

### 12. Documentation & Scripts

- [ ] `scripts/setup_jetson.sh` — JetPack dependencies
- [ ] `scripts/benchmark.py` — FPS + latency benchmarks
- [ ] `scripts/health_check.py` — Memory, GPU utilization
- [ ] `README.md` — Setup instructions, quickstart

---

## Verification Checklist

### Functional Tests

| #   | Test                | Criteria                          | Status |
| --- | ------------------- | --------------------------------- | ------ |
| V1  | YOLO inference      | Detects persons in test images    | ⬜     |
| V2  | ByteTrack           | Stable track_id across 100 frames | ⬜     |
| V3  | ROI extraction      | Correct crop size (128×256)       | ⬜     |
| V4  | CNN forward pass    | Output shape: (6,) + (2,)         | ⬜     |
| V5  | Context builder     | Observation vector shape: (102,)  | ⬜     |
| V6  | Heuristic policy    | Correct mode for each scenario    | ⬜     |
| V7  | Safety monitor      | Hard stop when person < 0.5m      | ⬜     |
| V8  | ZMQ publish         | RasPi receives NavigationCommand  | ⬜     |
| V9  | ZMQ subscribe       | Jetson receives RobotState        | ⬜     |
| V10 | Data logging        | All 7 fields present per frame    | ⬜     |
| V11 | Timestamp alignment | Max drift < 1ms across fields     | ⬜     |
| V12 | Ring buffer         | Handles 10K frames without leak   | ⬜     |

### Performance Benchmarks (Jetson)

| Metric                | Target           | Status |
| --------------------- | ---------------- | ------ |
| Pipeline FPS          | ≥ 30 FPS         | ⬜     |
| YOLO latency          | < 8ms            | ⬜     |
| CNN latency           | < 5ms per person | ⬜     |
| ZMQ latency           | < 5ms RTT        | ⬜     |
| Memory usage          | < 3.5 GB total   | ⬜     |
| Data write throughput | ≥ 30 frames/s    | ⬜     |

### Safety Tests 🔴

| #   | Scenario                  | Expected         | Status |
| --- | ------------------------- | ---------------- | ------ |
| S1  | Person at 0.3m            | STOP immediately | ⬜     |
| S2  | No RobotState for 600ms   | STOP (watchdog)  | ⬜     |
| S3  | ERRATIC intent detected   | STOP + alert     | ⬜     |
| S4  | Velocity overflow (>1.0)  | Clamp to 1.0     | ⬜     |
| S5  | Multiple persons crossing | STOP or AVOID    | ⬜     |

---

## Phase 1 → Phase 2 Handoff

Khi HOÀN THÀNH Phase 1, Phase 2 cần:

1. ✅ Data logging đang hoạt động → dùng data để train CNN tốt hơn
2. ✅ Temporal-ready state → enable k=3-5 stacking
3. ✅ ZMQ stable → thêm gRPC cho Laptop
4. ✅ Modular architecture → swap heuristic → RL policy
5. ✅ Safety layer tested → giữ nguyên khi thêm RL

---

## Risk Register (Phase 1)

| Risk                                 | Probability | Impact   | Mitigation                                              |
| ------------------------------------ | ----------- | -------- | ------------------------------------------------------- |
| YOLO .engine build fails trên Jetson | Low         | High     | Dùng .pt fallback, build lại với đúng JetPack version   |
| Camera driver conflict (CSI vs USB)  | Medium      | Medium   | Test cả hai, document working config                    |
| ZMQ message loss qua WiFi            | Medium      | High     | Test với Ethernet trước, WiFi sau. Heartbeat monitoring |
| Data logging bottleneck (disk I/O)   | Low         | Critical | Async write, JPEG compression, monitor disk throughput  |
| ByteTrack ID switches thường xuyên   | Medium      | Medium   | Tune IOU threshold, max_age parameters                  |

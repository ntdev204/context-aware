# PLAN — Phase 2: Perception + Temporal Intent (GAP #2 Prep)

> **Project**: Context-Aware Navigation for Mecanum Robot
> **Phase**: 2 of 5
> **Timeline**: Week 4–5
> **Status**: 📋 Planning
> **Dependencies**: Phase 1 hoàn thành (data logging, ZMQ, perception pipeline, Docker infra)
> **Ref**: [system-design.md](../architecture/system-design.md) — Section 6, 16 (GAP #2), 17
> **GAP addressed**: #2 prep (temporal intent infrastructure)

---

## Mục tiêu

Nâng cấp perception pipeline: thêm temporal state stacking (k=3-5), cải thiện CNN với temporal aggregator, **thiết lập intent trajectory buffer per track_id** (chuẩn bị cho GAP #2 temporal intent reward ở Phase 3), và thiết lập gRPC + experience streaming tới Server.

---

## Prerequisites (từ Phase 1)

| Requirement           | Cần từ Phase 1                                      | Status |
| --------------------- | --------------------------------------------------- | ------ |
| Data logging pipeline | Đang thu data đúng format, aligned timestamps       | ⬜     |
| Temporal-ready state  | `ObservationSpace` với `temporal_stack_size` config | ⬜     |
| YOLO + ByteTrack      | Stable detection + tracking                         | ⬜     |
| ZMQ communication     | Jetson ↔ RasPi hoạt động ổn định                    | ⬜     |
| Collected data        | ≥ 5,000 frames with detections                      | ⬜     |

---

## Deliverables

| #   | Deliverable                  | Mô tả                                             | Bottleneck |
| --- | ---------------------------- | ------------------------------------------------- | ---------- |
| D1  | Temporal State Stacking      | Observation stacking k=3-5                        | B1 ⚠️      |
| D2  | CNN Temporal Upgrade         | Conv1D/GRU temporal aggregator                    | A1         |
| D3  | CNN Training Pipeline        | Train CNN intent trên Server                      | A1         |
| D4  | **Intent Trajectory Buffer** | Per-track intent history buffer — **GAP #2 prep** | —          |
| D5  | gRPC Server (Jetson)         | Model deployment + experience streaming           | C2         |
| D6  | gRPC Client (Server)         | Receive experiences, deploy models                | C2         |
| D7  | Experience Streamer          | Async experience stream → Server                  | C1         |
| D8  | Continual Learning Pipeline  | EWC + Replay Buffer cho CNN fine-tuning           | —          |

---

## Task Breakdown

### 1. Temporal State Stacking (D1) ⚠️ CRITICAL

> ⚠️ **WARNING**
> Đây là thời điểm kích hoạt temporal-ready design từ Phase 1. Thay đổi `temporal_stack_size: 1 → 3-5`.

- [ ] Cập nhật `src/navigation/context_builder.py`:
  - `temporal_stack_size` = 3 (default), configurable qua YAML
  - `observation_history: deque(maxlen=temporal_stack_size)`
  - Mỗi frame mới → push vào deque
  - `get_stacked_observation()` returns concatenated array
  - When deque chưa đầy → pad với zeros (cold start)

- [ ] State versioning:
  - `state_version: "v2-temporal-k3"` (update từ `"v1-snapshot"`)
  - Metadata ghi rõ: `obs_dim = 102 * k`
  - Backward-compatible: k=1 vẫn hoạt động

- [ ] Update config:

  ```yaml
  # config/development.yaml
  context:
    temporal_stack_size: 3 # Phase 1: 1, Phase 2: 3-5
    state_version: "v2-temporal-k3"
  ```

- [ ] Validation:
  - Test observation shape: `(102 * k,)` = `(306,)` khi k=3
  - Test cold-start behavior (deque chưa đầy)
  - Test transition từ v1 → v2

**Files**:

```
src/navigation/context_builder.py  (MODIFY)
config/development.yaml            (MODIFY)
config/production.yaml             (MODIFY)
```

---

### 2. CNN Temporal Upgrade (D2)

- [ ] Cập nhật `src/perception/intent_cnn.py`:
  - **Temporal Aggregator**: Conv1D hoặc GRU over last 5 frames per track_id
  - Architecture:

  ```
  Input: 5 × 128×256×3 (5 consecutive ROI frames cho cùng track_id)
      │
      ▼
  MobileNetV3-Small backbone (shared weights, per-frame)
      │ → 5 × feature_vector (576-dim)
      │
      ▼
  Option A: Conv1D(576, 256, kernel=3) → ReLU → Pool
  Option B: GRU(input=576, hidden=256, num_layers=1)
      │
      ▼
  FC(256) → IntentHead(6) + DirectionHead(2)
  ```

  - **Temporal buffer per track_id** (shared with D4 intent trajectory):

    ```python
    class TrackTemporalBuffer:
        def __init__(self, max_tracks=10, temporal_window=5):
            self.buffers: Dict[int, deque] = {}
            # Mỗi track_id có riêng 1 deque(maxlen=5)
    ```

  - When track_id mới → pad previous frames với current frame (duplicate)
  - When track_id mất → cleanup buffer sau 30 frames

- [ ] Model export pipeline:
  - PyTorch model → ONNX (opset 17)
  - Batch input: `(batch_size, 5, 3, 128, 256)`
  - Dynamic batch: `batch_size` = 1-5

**Files**:

```
src/perception/intent_cnn.py       (MODIFY)
src/perception/temporal_buffer.py  (NEW)
```

---

### 3. Intent Trajectory Buffer — GAP #2 Prep (D4)

> **GAP #2**: Phase 3 sẽ dùng `R_intent = Σ γ^i · risk(intent_{t+i})` với k=5 frames.
> Phase 2 cần **log intent trajectory per track_id** để có data cho reward shaping.

- [ ] `src/perception/intent_trajectory.py` (NEW):

  ```python
  class IntentTrajectoryTracker:
      """Maintains intent history per tracked person.

      Used in Phase 2 for logging, in Phase 3 for temporal reward.
      """
      def __init__(self, window_size: int = 5, max_tracks: int = 10):
          self.trajectories: Dict[int, deque] = {}  # track_id → deque
          self.window_size = window_size

      def update(self, track_id: int, intent_class: int, confidence: float):
          if track_id not in self.trajectories:
              self.trajectories[track_id] = deque(maxlen=self.window_size)
          self.trajectories[track_id].append(IntentPrediction(
              intent_class=intent_class,
              confidence=confidence,
              timestamp=time.monotonic(),
          ))

      def get_trajectory(self, track_id: int) -> List[IntentPrediction]:
          return list(self.trajectories.get(track_id, []))

      def cleanup_stale(self, active_track_ids: Set[int]):
          stale = set(self.trajectories.keys()) - active_track_ids
          for tid in stale:
              del self.trajectories[tid]
  ```

- [ ] Integrate into main pipeline:
  - After CNN predicts intent → `tracker.update(track_id, intent_class, conf)`
  - Include `intent_trajectory` in experience frame for logging
  - Log to HDF5: `intent_trajectories` field per frame

**Files**:

```
src/perception/intent_trajectory.py   (NEW)
src/main.py                           (MODIFY — integrate tracker)
```

---

### 4. CNN Training Pipeline — Server Side (D3)

- [ ] `training/train_cnn.py`:
  - Dataset: ROIs thu từ Phase 1 data logging
  - Labeling workflow:
    1. **Auto-label** với VLM (Gemma 4): send ROI → ask intent → get label
    2. **Human verify**: review VLM labels, correct mistakes
    3. **Export**: labeled dataset → PyTorch Dataset
  - Training config:

    ```yaml
    # training/configs/cnn_config.yaml
    model:
      backbone: mobilenetv3_small
      temporal_window: 5
      num_intent_classes: 6
      pretrained: true
      freeze_backbone_blocks: 3 # Freeze first 3 blocks

    training:
      epochs: 50
      batch_size: 32
      learning_rate: 1e-3
      weight_decay: 1e-4
      scheduler: cosine_annealing

    loss:
      intent: cross_entropy # weight: 0.7
      direction: mse # weight: 0.3

    augmentation:
      color_jitter: true
      random_crop: true
      horizontal_flip: true
      random_occlusion: true # Simulate partial occlusion
    ```

  - Train / Val / Test split: 70% / 15% / 15%
  - Metric tracking: accuracy, F1-score per class, confusion matrix

- [ ] `training/vlm_labeler.py`:
  - Sử dụng Gemma 4 (Ollama) để auto-label ROI images
  - Prompt: "What is this person's movement intent? [STATIONARY/APPROACHING/DEPARTING/CROSSING/FOLLOWING/ERRATIC]"
  - Output: intent label + confidence
  - Batch processing: ~500-1000 images/hour

- [ ] `training/export_model.py`:
  - PyTorch → ONNX (opset 17)
  - Validate ONNX model with `onnxruntime`
  - (TensorRT build sẽ thực hiện trên Jetson)

**Ref**: system-design.md Section 5.4

**Files**:

```
training/train_cnn.py              (NEW)
training/vlm_labeler.py            (NEW)
training/export_model.py           (NEW/MODIFY)
training/configs/cnn_config.yaml   (NEW/MODIFY)
training/dataset/                  (NEW dir)
```

---

### 4. gRPC Server — Jetson Side (D4)

- [ ] `src/communication/grpc_server.py`:
  - Service: `ModelDeployer` (Laptop → Jetson)
    - `DeployModel(ModelArtifact)` → receive ONNX, build TensorRT, hot-reload
    - `GetActiveModels(Empty)` → return danh sách models đang chạy
    - `RollbackModel(RollbackRequest)` → rollback về version trước
  - Port: `50051`
  - Async handler: model build chạy background thread
  - Security: basic token authentication

- [ ] `src/models/converter.py`:
  - ONNX → TensorRT `.engine` conversion
  - FP16 quantization
  - TensorRT workspace: 256MB max
  - Timeout: 5 phút max cho build

- [ ] `src/models/version_manager.py`:
  - Track active model versions
  - Keep last 2 versions cho rollback
  - `active_models.json` update

- [ ] `src/models/loader.py`:
  - Load TensorRT `.engine` files
  - Atomic swap khi hot-reload (< 33ms gap)

**Ref**: system-design.md Section 7.4, 11.1, 11.2

**Files**:

```
src/communication/grpc_server.py   (NEW)
src/models/__init__.py             (NEW)
src/models/converter.py            (NEW)
src/models/version_manager.py      (NEW)
src/models/loader.py               (NEW)
```

---

### 6. Experience Streamer — Jetson → Server (D7)

- [ ] `src/experience/streamer.py`:
  - gRPC client: stream `ExperienceFrame` tới Laptop
  - Data per frame:
    - `camera_frame_jpeg`: JPEG bytes (quality 85)
    - `detections`: Protobuf list
    - `observation`: flattened float array
    - `action`: action taken
    - `timestamp`: precise timestamp
  - Async streaming: non-blocking, does NOT affect inference loop
  - Batch mode: collect N frames → send batch (reduce overhead)
  - Error handling: retry with exponential backoff, drop if disconnected

- [ ] `src/communication/grpc_client.py`:
  - gRPC client wrapper cho experience streaming
  - Connection management: auto-reconnect
  - Health check: ping Laptop trước khi stream

- [ ] Laptop-side receiver (chuẩn bị cho Phase 3):
  - `training/experience_receiver.py`:
    - gRPC server trên Laptop
    - Receive experiences → store to disk
    - Format: HDF5 hoặc Parquet files
    - Organize by session/timestamp

**Ref**: system-design.md Section 7.4

**Files**:

```
src/experience/streamer.py             (NEW)
src/communication/grpc_client.py       (NEW)
training/experience_receiver.py        (NEW)
```

---

### 6. Update Main Pipeline

- [ ] `src/main.py` — Thêm:
  - gRPC server thread (model deployment)
  - Experience streamer thread (async upload)
  - Temporal buffer integration
  - Config reload support

- [ ] Updated pipeline flow:
  ```
  while running:
      1. Camera grab frame
      2. YOLO detection
      3. ByteTrack update
      4. ROI extraction
      5. CNN intent (NOW with temporal buffer)
      6. Context state build (NOW with k=3-5 stacking)
      7. Heuristic policy decision (vẫn rule-based, RL ở Phase 3)
      8. Safety monitor check
      9. ZMQ publish NavigationCommand
      10. Data logging (async)
      11. Experience streaming to Laptop (async, NEW)
  ```

**Files**:

```
src/main.py  (MODIFY)
```

---

### 7. Testing

- [ ] `tests/test_temporal_state.py`:
  - Observation shape khi k=1, k=3, k=5
  - Cold-start padding behavior
  - State version compatibility

- [ ] `tests/test_cnn_temporal.py`:
  - Temporal buffer per track_id
  - Track creation/cleanup
  - Forward pass shape: (batch, 5, 3, 128, 256) → (batch, 6+2)

- [ ] `tests/test_grpc.py`:
  - Model deployment round-trip
  - Experience streaming throughput
  - Connection recovery

- [ ] `tests/test_export.py`:
  - PyTorch → ONNX export correctness
  - ONNX validation with onnxruntime

**Files**:

```
tests/test_temporal_state.py     (NEW)
tests/test_cnn_temporal.py       (NEW)
tests/test_grpc.py               (NEW)
tests/test_export.py             (NEW)
```

---

## Verification Checklist

### Functional Tests

| #   | Test              | Criteria                                              | Status |
| --- | ----------------- | ----------------------------------------------------- | ------ |
| V1  | Temporal stacking | Observation shape = (102\*k,)                         | ⬜     |
| V2  | Cold start        | Padded correctly when deque not full                  | ⬜     |
| V3  | CNN temporal      | Forward pass with 5-frame input works                 | ⬜     |
| V4  | Track buffer      | Tracks maintain independent buffers                   | ⬜     |
| V5  | VLM labeling      | Gemma 4 labels ROIs correctly (>80% agree with human) | ⬜     |
| V6  | CNN training      | Converges on test dataset, accuracy >70%              | ⬜     |
| V7  | ONNX export       | Valid ONNX, inference matches PyTorch                 | ⬜     |
| V8  | gRPC deploy       | Send ONNX → Jetson builds TensorRT                    | ⬜     |
| V9  | Hot-reload        | Model swap < 33ms gap                                 | ⬜     |
| V10 | Experience stream | Laptop receives complete frames                       | ⬜     |
| V11 | Stream throughput | ≥ 10 frames/s over WiFi                               | ⬜     |

### Performance (Jetson)

| Metric                        | Target           | Status |
| ----------------------------- | ---------------- | ------ |
| Pipeline FPS (with temporal)  | ≥ 25 FPS         | ⬜     |
| CNN latency (temporal)        | < 8ms per person | ⬜     |
| gRPC overhead                 | < 5% CPU         | ⬜     |
| Memory (with gRPC + temporal) | < 4.0 GB         | ⬜     |

---

## Phase 2 → Phase 3 Handoff

Khi HOÀN THÀNH Phase 2, Phase 3 cần:

1. ✅ Temporal state hoạt động → RL observation space đã final
2. ✅ CNN trained & deployed → intent predictions có ý nghĩa
3. ✅ **Intent trajectory buffer** → sẵn sàng cho GAP #2 temporal reward
4. ✅ gRPC server → nhận RL model từ Server
5. ✅ Experience streaming → data pipeline cho RL training
6. ✅ Collected experiences → sẵn sàng cho VLM evaluation + SAC training
7. ✅ Continual learning pipeline (EWC + Replay) → CNN fine-tuning automated

---

## Risk Register (Phase 2)

| Risk                                   | Probability | Impact | Mitigation                                               |
| -------------------------------------- | ----------- | ------ | -------------------------------------------------------- |
| CNN temporal model quá nặng cho Jetson | Low         | High   | Benchmark trước. Fallback: chỉ dùng Conv1D (nhẹ hơn GRU) |
| VLM auto-labeling quality thấp         | Medium      | Medium | Human verify 100% initially, giảm dần khi VLM consistent |
| gRPC streaming unstable qua WiFi       | Medium      | High   | Retry logic, batch mode, test với Ethernet trước         |
| TensorRT build timeout                 | Low         | Medium | Increase workspace, pre-build trên Jetson cùng JetPack   |
| Not enough data from Phase 1           | Medium      | High   | Chạy Phase 1 thêm 1 tuần nếu cần, hoặc augment data      |

# Temporal Intent CNN — Thiết Kế Kiến Trúc Mạng

> **Phạm vi:** Module `src/perception/intent_cnn.py` + `src/perception/roi_extractor.py`
> **Cập nhật:** 2026-05-01
> **Backend:** PyTorch · MobileNetV3-Small + TCN/Conv1D · FP16/CUDA (Jetson Orin)

> **Quyết định hiện tại:** hệ thống đã loại bỏ cơ chế bám người. Vì vậy `FOLLOW`/`FOLLOWING` không còn là navigation mode hoặc intent class. Các mẫu mơ hồ được đưa vào `UNCERTAIN` và hàng đợi human review.

---

## 1. Tổng Quan

Intent CNN là mạng nơ-ron tích chập (CNN) tùy biến được thiết kế để **phân loại ý định chuyển động của người** dựa trên ảnh crop ROI (Region of Interest) được trích xuất từ pipeline YOLO + ByteTrack.

Mạng giải quyết bài toán: _"Robot có thể biết người đang định làm gì không?"_ — không chỉ phát hiện tư thế mà còn **dự đoán hành vi tương lai gần** để robot phản ứng chủ động.

### Tại sao không dùng YOLO pose hoặc action recognition thuần túy?

| Tiêu chí           | YOLO Pose | Action Recognition (SlowFast) | Intent CNN (thiết kế này) |
| ------------------ | --------- | ----------------------------- | ------------------------- |
| Latency            | ~10ms     | ~150ms+                       | ~3ms (FP16)               |
| Phụ thuộc temporal | Không     | Cần video clip                | Không (single frame)      |
| Nhận dạng ý định   | Không     | Hành động đã xảy ra           | Ý định đang xảy ra        |
| Deploy Jetson      | Tốt       | Khó (VRAM)                    | Tốt (MobileNet)           |

---

## 2. Phân Loại Ý Định (Intent Classes)

Mạng runtime xuất vector 6 slot, nhưng chỉ có **5 nhãn trainable**. Slot thứ 6 là abstain/review:

```python
STATIONARY  = 0   # Đứng yên tại chỗ
APPROACHING = 1   # Đang tiến về phía robot
DEPARTING   = 2   # Đang rời xa robot
CROSSING    = 3   # Đi ngang qua trường nhìn của robot
ERRATIC     = 4   # Hành vi bất thường, cần review nếu sinh từ heuristic
UNCERTAIN   = 5   # Abstain: thiếu chắc chắn, không train trực tiếp
```

> **Lưu ý thiết kế:** `ERRATIC` và `UNCERTAIN` đi qua human-in-the-loop review. `UNCERTAIN` bị loại khỏi training; `ERRATIC` chỉ được train khi đã được xác nhận.

---

## 3. Kiến Trúc Tổng Thể (Full Pipeline)

> **Cập nhật:** Đây là đồ thị mạng nơ-ron minh họa trực quan dưới dạng "Nodes & Edges" thể hiện đầy đủ chiều sâu thực tế 54 Layer của kiến trúc (1 Input + 50 Backbone + 1 Pooling + 1 Hidden + 1 Output).

![Intent CNN 54-Layer Architecture Diagram](file:///d:/nckh/context-aware/docs/architecture/intent_cnn_54_layers.png)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Camera Frame (BGR)                          │
│                      Shape: (H, W, 3) @ 30Hz                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      YOLODetector + ByteTrack                       │
│  • Phát hiện person (class 0)                                       │
│  • Gán track_id ổn định qua các frame                              │
│  • Output: FrameDetections { persons: List[DetectionResult] }       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ FrameDetections
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ROIExtractor                                │
│  • Crop bbox với padding 10%                                        │
│  • Resize → (128 × 256) px [W × H]                                 │
│  • Tính (cx_norm, cy_norm) vị trí tương đối trong frame            │
│  • Output: List[PersonROI]                                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ List[PersonROI]
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         IntentCNN (Async)                           │
│                                                                     │
│   Main thread: predict_batch() → trả cache ngay (non-blocking)     │
│   Daemon thread: _worker() → chạy inference ngầm                   │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │              _IntentModel (PyTorch)                         │  │
│   │                                                             │  │
│   │  Input: (B, T, 3, 256, 128) — T=15 ROI frames/track       │  │
│   │                    │                                        │  │
│   │           MobileNetV3-Small Backbone                        │  │
│   │           (pretrained ImageNet, classifier=Identity)        │  │
│   │                    │                                        │  │
│   │           AdaptiveAvgPool2d(1) → per-frame feature          │  │
│   │           Output: (B, T, 576)                               │  │
│   │                    │                                        │  │
│   │           Depthwise Conv1D + Pointwise Conv1D               │  │
│   │           Output: (B, 256)                                  │  │
│   │                    │                                        │  │
│   │         ┌──────────┴──────────┐                            │  │
│   │         │                     │                            │  │
│   │   Intent Head           Direction Head                     │  │
│   │   Linear(256→256)       Linear(256→256)                    │  │
│   │   ReLU                  ReLU                               │  │
│   │   Dropout(0.2)          Linear(256→2)                      │  │
│   │   Linear(256→5)         Tanh → (dx, dy)                    │  │
│   │   Temperature softmax   ∈ [-1, 1]²                         │  │
│   │   + UNCERTAIN abstain                                      │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│   Output per person:                                                │
│     • intent_class  : argmax(probs)                                 │
│     • probabilities : float[6]                                      │
│     • confidence    : max(probs)                                    │
│     • dx, dy        : hướng chuyển động dự đoán                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ List[IntentPrediction]
                               ▼
                    Navigation / ExperienceBuffer
```

---

## 4. Chi Tiết Backbone: MobileNetV3-Small

### Tại sao MobileNetV3-Small?

- **Thiết kế cho edge/mobile**: squeeze-and-excite + hard-swish, tối ưu FLOP/accuracy
- **Feature dim = 576**: đầu ra `backbone.features` sau `AdaptiveAvgPool2d(1)` đủ giàu cho phân loại 5 class trainable
- **Pretrained ImageNet**: transfer learning mạnh kể cả khi dataset intent còn nhỏ
- **FP16 friendly**: không có batch norm issue khi chuyển sang half precision

```python
backbone = tv_models.mobilenet_v3_small(
    weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
)
backbone.classifier = nn.Identity()   # Bỏ head gốc, lấy features thô
```

> **Lưu ý:** `backbone.features(x)` trả về feature map `(B, 576, H', W')`. `AdaptiveAvgPool2d(1)` global average pool xuống `(B, 576, 1, 1)` rồi `flatten(1)` → `(B, 576)`.

---

## 5. Dual-Head Design

### Intent Head (Classification)

```
Conv1D temporal aggregation → Linear(256 → 256) → ReLU → Dropout(0.2) → Linear(256 → 5)
```

- **Dropout(0.2)**: regularization tránh overfitting, quan trọng vì dataset intent thường mất cân bằng
- **Softmax** áp dụng ở inference (`torch.softmax` sau khi lấy logits), **không** trong forward pass — để tương thích với `CrossEntropyLoss` khi training

### Direction Head (Regression)

```
Linear(576 → 256) → ReLU → Linear(256 → 2) → Tanh
```

- Output: `(dx, dy) ∈ [-1, 1]²` — véc-tơ hướng chuyển động chuẩn hóa
- **Tanh**: ép output vào [-1, 1], tránh gradient explode
- Head này là **auxiliary task** — giúp backbone học feature về motion, cải thiện gián tiếp accuracy của intent head

### Tại sao thiết kế 2 head chia sẻ backbone?

```
Backbone (shared weights)
    ├── Intent Head    → loss = CrossEntropy
    └── Direction Head → loss = MSE hoặc L1

Total loss = α·L_intent + β·L_direction
```

- **Multi-task learning**: direction là signal tự nhiên hơn, dễ học trước, kéo backbone học feature tốt hơn
- **Ít tham số hơn** 2 mạng riêng biệt
- **Inference cost bằng nhau** vì chỉ chạy backbone 1 lần

---

## 6. ROI Preprocessing Pipeline

```python
# roi_extractor.py
CNN_INPUT_W = 128
CNN_INPUT_H = 256   # aspect ratio 1:2 → phù hợp với body người đứng
```

### Bước xử lý ảnh (trong `IntentCNN._preprocess`):

```
BGR (H, W, 3)
    → RGB conversion (cv2.COLOR_BGR2RGB)
    → /255.0 → float32 [0, 1]
    → normalize với mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
       (ImageNet stats — phù hợp với pretrained MobileNetV3)
    → transpose HWC → CHW
    → stack batch → (N, 3, 256, 128)
```

### Padding trong ROI crop:

```python
padding_ratio = 0.10   # 10% padding quanh bbox
```

Padding giúp model nhìn thấy **context xung quanh người** (tay, chân, vật xung quanh), quan trọng để phân biệt `CROSSING`, `APPROACHING` và mẫu `UNCERTAIN`.

---

## 7. Thiết Kế Inference Bất Đồng Bộ (Non-Blocking)

Đây là một trong những thiết kế quan trọng nhất, đảm bảo inference CNN không block pipeline camera 30Hz.

```
Main Thread (30Hz)              Daemon Thread (inference)
─────────────────────           ──────────────────────────
frame → YOLO → ROIs             loop:
  │                               lock → lấy rois_queue
  ├─ predict_batch(rois):         _preprocess(images)
  │    lock → push rois_queue     _infer_pytorch(batch) ← GPU call
  │    lock → đọc cache           update cache per track_id
  │    return cache kết quả       dọn dead track_ids
  │                               sleep(10ms) nếu queue rỗng
  └─ tiếp tục pipeline...
```

**Trade-off**: Kết quả intent luôn là của **frame trước đó** (1 inference cycle lag). Chấp nhận được vì:

1. Intent người thay đổi chậm (>100ms)
2. Robot không cần phản ứng ngay lập tức theo từng frame
3. FPS ổn định quan trọng hơn latency inference

### Cache management:

```python
# Tự động dọn track_id không còn active
active_ids = {r.track_id for r in rois}
dead_ids = [tid for tid in self._cache if tid not in active_ids]
for tid in dead_ids:
    del self._cache[tid]
```

---

## 8. Quản Lý Device & Precision

```python
# Auto-detect: CUDA FP16 (Jetson) hoặc CPU FP32 (dev)
if self.device == "cuda" and not torch.cuda.is_available():
    self.device = "cpu"

self._dtype = torch.float16 if self.device == "cuda" else torch.float32
```

| Môi trường          | Device | Dtype     | Lý do                            |
| ------------------- | ------ | --------- | -------------------------------- |
| Jetson Orin Nano    | `cuda` | `float16` | Tiết kiệm VRAM, Tensor Core FP16 |
| Laptop Dev (no GPU) | `cpu`  | `float32` | FP16 không stable trên CPU       |
| Laptop Dev (RTX)    | `cuda` | `float16` | Có thể dùng FP16                 |

> **Quan trọng:** Softmax và Tanh được cast về `float32` trước khi trả về để tránh mất độ chính xác numerics, dù model chạy FP16:
>
> ```python
> probs = torch.softmax(intent_logits.float(), dim=-1).cpu().numpy()
> dirs  = torch.tanh(direction.float()).cpu().numpy()
> ```

---

## 9. Output: IntentPrediction Dataclass

```python
@dataclass
class IntentPrediction:
    track_id: int           # ID từ ByteTrack
    intent_class: int       # 0-5 (argmax của probabilities)
    intent_name: str        # "STATIONARY", "APPROACHING", ...
    probabilities: np.ndarray  # shape (6,) — softmax scores
    dx: float               # hướng x [-1, 1]
    dy: float               # hướng y [-1, 1]
    confidence: float       # max(probabilities)
    inference_ms: float     # thời gian inference (per-sample)
```

Downstream module (`navigation/`, `experience/buffer.py`) dùng `intent_class` và `confidence` để:

- Quyết định hành vi robot (tiếp cận, giữ khoảng cách, dừng)
- Ghi vào HDF5 experience buffer cho training offline

---

## 10. Luồng Training (Offline)

```
ExperienceBuffer (HDF5)
    → autolabel.py (motion compensation + heuristics)
    → dataset: ROI images + intent labels
    → training loop:
         loss = α·CrossEntropy(intent_probs, label)
               + β·MSE(direction, target_direction)
    → save: model_state_dict → models/cnn_intent/intent_cnn.pt
    → load tại runtime: IntentCNN(model_path="models/cnn_intent/intent_cnn.pt")
```

> Xem `docs/architecture/` cho thiết kế autolabel pipeline (motion compensation).

---

## 11. Checklist Mở Rộng / Cải Tiến

| Hạng mục                | Trạng thái     | Ghi chú                         |
| ----------------------- | -------------- | ------------------------------- |
| MobileNetV3 backbone    | ✅ Done        | FP16 on Jetson                  |
| Temporal Conv1D/TCN     | ✅ Done        | Window ROI per track            |
| Intent head (5 class)   | ✅ Done        | Runtime thêm `UNCERTAIN`        |
| Direction head (aux)    | ✅ Done        |                                 |
| Async inference daemon  | ✅ Done        | Non-blocking 30Hz               |
| Auto device detection   | ✅ Done        | CUDA/CPU                        |
| Dataset manifest gate   | ✅ Done        | Chặn legacy FOLLOW/review pending |
| Confidence calibration  | ✅ Done        | Temperature scaling metadata    |
| Distillation/quantize   | 🔄 In Progress | Export script có dynamic quantize |
| Continual learning      | ✅ Baseline    | EWC + replay buffer             |

---

## 12. Sơ Đồ Luồng Dữ Liệu Tổng Thể

```
Camera
  │ 30Hz RGB frames
  ▼
YOLOv8 (640×480)
  │ bbox + track_id
  ▼
ROIExtractor
  │ crops (128×256) per person
  ▼
IntentCNN._worker (daemon, ~10-30Hz)
  │ batch inference on GPU
  ▼
cache: {track_id → IntentPrediction}
  │ read by main thread
  ▼
NavigationController
  │ điều chỉnh cmd_vel
  ▼
ZMQ → Raspberry Pi → ROS2 twist_mux
```

---

_File này được generate tự động từ code analysis — cập nhật khi có thay đổi kiến trúc quan trọng._

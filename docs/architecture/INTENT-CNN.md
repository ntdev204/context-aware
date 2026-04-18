# Intent CNN — Mô Tả Chi Tiết Module Dự Đoán Ý Định Con Người

> **Module**: `src/perception/intent_cnn.py`
> **Phiên bản**: v1.0
> **Cập nhật**: 2026-04-17
> **Trạng thái**: Phase 1 — Data Collection (chưa có trained weights)

---

## Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Intent CNN Học Cái Gì?](#2-intent-cnn-học-cái-gì)
3. [Intent CNN Cải Thiện Cái Gì Cho Hệ Thống?](#3-intent-cnn-cải-thiện-cái-gì-cho-hệ-thống)
4. [Kiến Trúc Mạng Chi Tiết](#4-kiến-trúc-mạng-chi-tiết)
5. [6 Lớp Ý Định (Intent Classes)](#5-6-lớp-ý-định-intent-classes)
6. [Dữ Liệu Đầu Vào — Đầu Ra](#6-dữ-liệu-đầu-vào--đầu-ra)
7. [Luồng Xử Lý Trong Pipeline](#7-luồng-xử-lý-trong-pipeline)
8. [Chiến Lược Huấn Luyện](#8-chiến-lược-huấn-luyện)
9. [Vai Trò Trong Hệ Thống Navigation](#9-vai-trò-trong-hệ-thống-navigation)
10. [Hiệu Năng & Tối Ưu](#10-hiệu-năng--tối-ưu)
11. [Hạn Chế & Hướng Phát Triển](#11-hạn-chế--hướng-phát-triển)
12. [File HDF5 & Mối Quan Hệ Với Intent CNN](#12-file-hdf5--mối-quan-hệ-với-intent-cnn)

---

## 1. Tổng Quan

**Intent CNN** là module **dự đoán ý định hành vi** (behavioral intent prediction) của con người được phát hiện trong khung hình camera của robot. Đây là **core differentiator** của hệ thống context-aware navigation — thay vì chỉ biết "có người ở đây", robot còn hiểu "người này **định làm gì**".

### Tại sao cần Intent CNN?

Hầu hết robot navigation truyền thống chỉ dựa vào:
- **Vị trí hiện tại** của vật thể (từ YOLO detection)
- **Khoảng cách** (từ sensor depth)
- **Occupancy map** tĩnh

Điều này dẫn đến phản ứng **reactive** (phản ứng sau khi sự việc xảy ra). Ví dụ: robot chỉ dừng khi người đã chặn đường, thay vì **dự đoán trước** rằng người đang **đi tới** và chủ động né tránh sớm.

Intent CNN bổ sung khả năng **proactive** (chủ động):

```
 Reactive (Truyền thống)          Proactive (Intent CNN)
 ────────────────────────         ────────────────────────
 "Có người ở trước mặt"    →     "Người này đang tiến về phía robot"
 → Dừng khi quá gần             → Giảm tốc sớm + chuẩn bị tránh

 "Có người bên trái"       →     "Người này đang cắt ngang đường đi"
 → Không phản ứng gì            → Dừng/chờ cho người đi qua
```

---

## 2. Intent CNN Học Cái Gì?

Intent CNN học **2 task đồng thời** (multi-task learning) từ ảnh cropped của mỗi người:

### Task 1: Phân Loại Ý Định (Intent Classification)

Mạng học **ánh xạ**:

```
f: Ảnh cropped người (128×256×3) → Phân phối xác suất trên 6 lớp ý định
```

Cụ thể, mạng học nhận diện các **visual cues** (dấu hiệu thị giác) cho biết ý định di chuyển:

| Visual Cue | Ý Nghĩa | Ví Dụ |
|---|---|---|
| **Hướng cơ thể** (body orientation) | Người đang quay mặt/lưng về hướng nào | Mặt hướng robot = APPROACHING |
| **Tư thế chân** (leg pose) | Đang bước/đứng yên | Chân bước ngang = CROSSING |
| **Góc nghiêng thân** (torso lean) | Đang trong quá trình di chuyển | Nghiêng về phía robot = APPROACHING |
| **Vị trí tương đối** (relative position) | Vùng ảnh crop thay đổi theo frame | Người thu nhỏ dần = DEPARTING |
| **Biểu cảm không gian** (spatial context) | Nền phía sau, hành lang, cửa | Gần cửa + hướng ngang = CROSSING |

### Task 2: Ước Lượng Hướng Di Chuyển (Direction Estimation)

Mạng học **regression**:

```
g: Ảnh cropped người → Vector hướng (dx, dy) ∈ [-1, 1]²
```

| Thành phần | Ý nghĩa | Giá trị |
|---|---|---|
| `dx` | Hướng di chuyển ngang | -1 = sang trái, +1 = sang phải |
| `dy` | Hướng di chuyển dọc | -1 = lại gần robot, +1 = ra xa robot |

### Tóm tắt: Mạng học gì từ ảnh?

```
                         ┌────────────────────────────────────────┐
                         │   Từ 1 ảnh crop 128×256 của 1 người   │
                         │                                        │
                         │   Mạng trích xuất:                    │
                         │                                        │
                         │   1. Tư thế cơ thể                    │
                         │   2. Hướng nhìn / quay mặt            │
                         │   3. Trạng thái di chuyển              │
                         │   4. Ngữ cảnh không gian xung quanh   │
                         │                                        │
                         │   Để suy ra:                           │
                         │                                        │
                         │   ● Ý định hành vi (6 classes)        │
                         │   ● Vector hướng di chuyển (dx, dy)   │
                         └────────────────────────────────────────┘
```

---

## 3. Intent CNN Cải Thiện Cái Gì Cho Hệ Thống?

### 3.1 Cải thiện trực tiếp — Navigation Decision Quality

Intent CNN cung cấp thông tin cho **3 module downstream**:

#### A. Heuristic Policy (quyết định điều hướng)

```python
# Trong heuristic_policy.py — Intent ảnh hưởng trực tiếp đến hành vi robot:

# RULE 2: ERRATIC → STOP (ưu tiên cao nhất sau proximity)
for pred in intent_preds:
    if pred.intent_class == ERRATIC and pred.confidence > 0.6:
        return STOP  # Dừng ngay khi phát hiện hành vi bất thường

# RULE 3: CROSSING / APPROACHING → AVOID
for person in persons:
    if intent in (CROSSING, APPROACHING) and confidence > 0.5:
        return AVOID  # Giảm tốc + đổi hướng
```

**Không có Intent CNN**: Robot chỉ dừng khi người đã quá gần (reactive).
**Có Intent CNN**: Robot phát hiện người đang APPROACHING từ xa → giảm tốc sớm (proactive).

#### B. Safety Monitor (giám sát an toàn)

```python
# Trong safety_monitor.py — Intent override bảo vệ tối thượng:

for pred in intent_preds:
    if pred.intent_class == ERRATIC and pred.confidence > 0.6:
        return EMERGENCY_STOP  # Override mọi quyết định khác
```

**Cải thiện**: Phát hiện hành vi nguy hiểm (người say, trẻ chạy loạn) → dừng ngay dù khoảng cách chưa vi phạm.

#### C. Context Builder (xây dựng observation vector cho RL)

```python
# Trong context_builder.py — Intent trở thành feature cho RL training:

# observation[70:94] = 3 người × 8 features mỗi người
for i, person in enumerate(top_3_persons):
    obs[70 + i*8 : 70 + i*8 + 6] = intent_probabilities  # 6 prob
    obs[70 + i*8 + 6] = pred.dx                           # hướng x
    obs[70 + i*8 + 7] = pred.dy                           # hướng y
```

**Cải thiện**: Observation vector cho RL Policy chứa thông tin ý định → RL agent có thể học quyết định phức tạp hơn (ví dụ: "4 người đang CROSSING + 1 APPROACHING → dừng và chờ thay vì né").

### 3.2 Cải thiện gián tiếp — System-wide Benefits

```
┌──────────────────────────────────────────────────────────────────┐
│                  Matrix Cải Thiện Tổng Thể                       │
│                                                                   │
│  ┌──────────────┐   Intent CNN    ┌──────────────────────────┐  │
│  │   Trước CNN   │  ────────────→ │      Sau CNN              │  │
│  ├──────────────┤                 ├──────────────────────────┤  │
│  │ Reactive only │                │ Proactive + Reactive      │  │
│  │ Dừng khi gần  │                │ Dự đoán & phản ứng sớm   │  │
│  │ Mọi người     │                │ Phân loại ý định          │  │
│  │  như nhau      │                │  từng người riêng biệt    │  │
│  │ Heuristic     │                │ Context-Aware             │  │
│  │  đơn giản     │                │  navigation               │  │
│  │ Safety =      │                │ Safety =                  │  │
│  │  proximity    │                │  proximity + behavior     │  │
│  │ RL blind to   │                │ RL understands             │  │
│  │  human intent │                │  human behavior            │  │
│  └──────────────┘                 └──────────────────────────┘  │
│                                                                   │
│  Metric cải thiện kỳ vọng:                                       │
│  • Collision avoidance:  +40-60% (phản ứng sớm hơn)             │
│  • Navigation smoothness: +30-50% (giảm dừng đột ngột)          │
│  • Social appropriateness: +50-70% (biết chờ khi cần)           │
│  • Processing overhead: chỉ thêm ~3ms/person                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Kiến Trúc Mạng Chi Tiết

### 4.1 Tổng quan sơ đồ

```
  Input: Ảnh BGR 128×256×3 (Cropped Person ROI)
         │
         ▼ Preprocessing: BGR→RGB, /255, ImageNet normalize
         │
  ┌──────┴──────────────────────────────────────────┐
  │           MobileNetV3-Small (Backbone)           │
  │                                                   │
  │  Pretrained trên ImageNet (1000 classes)          │
  │  Tầng classifier gốc bị thay bằng Identity()     │
  │                                                   │
  │  features(x) → Feature Map: (B, 576, H', W')     │
  │  B = batch size, 576 = channels ở layer cuối      │
  │  H', W' phụ thuộc spatial resolution              │
  └──────────────────────┬────────────────────────────┘
                         │
                         ▼
              AdaptiveAvgPool2d(1)
                         │
                    flatten(1)
                         │
            Feature Vector: (B, 576)
                    ┌────┴────┐
                    │         │
                    ▼         ▼
          ┌─────────────┐ ┌──────────────┐
          │ Intent Head  │ │Direction Head│
          │              │ │              │
          │ Linear(576→  │ │ Linear(576→  │
          │        256)  │ │        256)  │
          │ ReLU         │ │ ReLU         │
          │ Dropout(0.2) │ │ Linear(256→  │
          │ Linear(256→  │ │        2)    │
          │        6)    │ │              │
          └──────┬───────┘ └──────┬───────┘
                 │                │
                 ▼                ▼
           Softmax(dim=-1)   Tanh()
                 │                │
                 ▼                ▼
          6 probabilities    (dx, dy)
          [P(STATIONARY),    dx ∈ [-1, 1]
           P(APPROACHING),   dy ∈ [-1, 1]
           P(DEPARTING),
           P(CROSSING),
           P(FOLLOWING),
           P(ERRATIC)]
```

### 4.2 Chi tiết từng thành phần

#### Backbone: MobileNetV3-Small

| Thuộc tính | Giá trị |
|---|---|
| Architecture | MobileNetV3-Small |
| Pretrained | ImageNet (IMAGENET1K_V1) |
| Parameters | ~2.54M |
| Feature dim | 576 (output channels of last inverted residual block) |
| Input resolution | 128×256 (width × height) |
| Tại sao MobileNetV3? | Nhẹ (2.5M params), nhanh (~2ms trên Jetson Ampere), đủ mạnh cho single-person classification |

#### Intent Head (Classification)

```python
self.intent_head = nn.Sequential(
    nn.Linear(576, 256),     # Giảm chiều + non-linearity
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),         # Regularization chống overfit
    nn.Linear(256, 6),       # 6 intent classes
)
# Softmax áp dụng ở inference time (trong _infer_pytorch)
```

- **Loss function** (training): `CrossEntropyLoss` — standard cho multi-class classification
- **Output**: Phân phối xác suất trên 6 classes, tổng = 1.0

#### Direction Head (Regression)

```python
self.direction_head = nn.Sequential(
    nn.Linear(576, 256),     # Giảm chiều
    nn.ReLU(inplace=True),
    nn.Linear(256, 2),       # 2 outputs: dx, dy
)
# Tanh áp dụng ở inference time → range [-1, 1]
```

- **Loss function** (training): `MSELoss` — standard cho regression
- **Output**: Vector hướng di chuyển (dx, dy), mỗi thành phần trong [-1, 1]

### 4.3 Tại sao Dual-Head (Multi-task Learning)?

```
         Shared Backbone (MobileNetV3-Small)
                ┌────────┴────────┐
                │                 │
         Intent Head        Direction Head
        (Classification)     (Regression)
```

**Lợi ích Multi-task Learning:**

1. **Shared features**: Backbone trích xuất features hữu ích cho cả 2 tasks → tiết kiệm computation
2. **Regularization**: 2 tasks bổ sung cho nhau, giảm overfitting
3. **Richer representation**: Direction và intent có correlation mạnh (CROSSING → dx lớn, APPROACHING → dy âm)
4. **Efficiency**: 1 forward pass = 2 outputs, thay vì 2 mạng riêng biệt

---

## 5. 6 Lớp Ý Định (Intent Classes)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        6 INTENT CLASSES                                │
├──────┬──────────────┬────────────────────────┬────────────────────────┤
│  ID  │   Tên        │   Mô tả                │   Robot phản ứng       │
├──────┼──────────────┼────────────────────────┼────────────────────────┤
│  0   │ STATIONARY   │ Người đứng yên,        │ Giảm tốc nhẹ,          │
│      │              │ không di chuyển         │ giữ khoảng cách       │
├──────┼──────────────┼────────────────────────┼────────────────────────┤
│  1   │ APPROACHING  │ Người đang đi về phía  │ Giảm tốc + chuẩn bị  │
│      │              │ robot (collision path)  │ AVOID mode             │
├──────┼──────────────┼────────────────────────┼────────────────────────┤
│  2   │ DEPARTING    │ Người đang đi ra xa    │ Duy trì / tăng tốc,   │
│      │              │ khỏi robot             │ không cần lo ngại      │
├──────┼──────────────┼────────────────────────┼────────────────────────┤
│  3   │ CROSSING     │ Người cắt ngang đường  │ **DỪNG** hoặc chờ    │
│      │              │ đi của robot           │ cho đến khi đường      │
│      │              │                        │ thông thoáng           │
├──────┼──────────────┼────────────────────────┼────────────────────────┤
│  4   │ FOLLOWING    │ Người đi theo / đi cùng│ Giữ tốc độ ổn định,  │
│      │              │ hướng robot            │ không tránh            │
├──────┼──────────────┼────────────────────────┼────────────────────────┤
│  5   │ ERRATIC      │ Chuyển động bất thường │ **DỪNG KHẨN CẤP**   │
│      │              │ (trẻ chạy, người say)  │ + cảnh báo             │
└──────┴──────────────┴────────────────────────┴────────────────────────┘
```

### Phân cấp ưu tiên trong Navigation:

```
          ERRATIC (5)   ← Priority 1: STOP ngay lập tức
              │
         CROSSING (3)   ← Priority 2: STOP/chờ
         APPROACHING(1) ← Priority 2: AVOID (giảm tốc + đổi hướng)
              │
         FOLLOWING (4)  ← Priority 3: Giữ tốc ổn định
         STATIONARY(0)  ← Priority 3: Đi chậm, cẩn thận
              │
         DEPARTING (2)  ← Priority 4: An toàn, duy trì tốc
```

---

## 6. Dữ Liệu Đầu Vào — Đầu Ra

### 6.1 Đầu Vào (Input)

```python
# 1 ảnh crop từ ROIExtractor
PersonROI:
    image:              np.ndarray  # shape (256, 128, 3), BGR uint8
    bbox:               (x1, y1, x2, y2)  # pixel coords trong frame gốc
    track_id:           int               # từ ByteTrack/Fallback tracker
    relative_position:  (cx_norm, cy_norm) # vị trí trung tâm normalize [0,1]
```

**Preprocessing pipeline** (bên trong `_preprocess()`):

```
BGR uint8 (256,128,3)
         │
    cv2.cvtColor(BGR→RGB)
         │
    .astype(float32) / 255.0    →  range [0, 1]
         │
    Normalize với ImageNet stats:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
         │
    .transpose(2,0,1)            →  HWC → CHW
         │
Output: float32 (3, 256, 128) per image
Batch:  float32 (N, 3, 256, 128) where N ≤ max_batch_size=5
```

### 6.2 Đầu Ra (Output)

```python
@dataclass
class IntentPrediction:
    track_id:       int          # ID tracking (liên kết với person)
    intent_class:   int          # 0-5 (argmax của probabilities)
    intent_name:    str          # "STATIONARY", "APPROACHING", etc.
    probabilities:  np.ndarray   # shape (6,), tổng = 1.0
    dx:             float        # hướng di chuyển x ∈ [-1, 1]
    dy:             float        # hướng di chuyển y ∈ [-1, 1]
    confidence:     float        # max probability trong 6 classes
    inference_ms:   float        # thời gian inference per person
```

### 6.3 Ví dụ output thực tế

```python
IntentPrediction(
    track_id=3,
    intent_class=1,                # APPROACHING
    intent_name="APPROACHING",
    probabilities=np.array([
        0.05,   # P(STATIONARY)
        0.72,   # P(APPROACHING) ← max
        0.03,   # P(DEPARTING)
        0.15,   # P(CROSSING)
        0.02,   # P(FOLLOWING)
        0.03,   # P(ERRATIC)
    ]),
    dx=-0.1,   # hơi lệch trái
    dy=-0.6,   # mạnh về phía robot
    confidence=0.72,
    inference_ms=2.3,
)
```

---

## 7. Luồng Xử Lý Trong Pipeline

### 7.1 Vị trí trong Inference Loop

```
Frame từ Camera
     │
     ▼
┌─────────────────┐
│  YOLOv11s       │  → FrameDetections (persons, obstacles)
│  detect()       │       │
└─────────────────┘       │
                          ▼
                   ┌──────────────┐
                   │  ByteTrack   │  → Gán track_id ổn định cho mỗi person
                   │  update()    │       │
                   └──────────────┘       │
                                          ▼
                                   ┌──────────────┐
                                   │ ROI Extractor │  → List[PersonROI]
                                   │  extract()   │    (128×256 crops)
                                   └──────────────┘       │
                                                          ▼
                              ┌────────────────────────────────────────┐
                              │  ★ Intent CNN  ★                       │
                              │  predict_batch(rois)                   │
                              │                                        │
                              │  1. _preprocess() : BGR→RGB→normalize │
                              │  2. _infer_pytorch() : forward pass    │
                              │  3. Softmax + Tanh postprocess        │
                              │                                        │
                              │  Output: List[IntentPrediction]        │
                              └───────────────────┬────────────────────┘
                                                  │
                              ┌────────────────────┼────────────────────┐
                              │                    │                    │
                              ▼                    ▼                    ▼
                     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                     │Context Builder│   │Heuristic     │    │Safety Monitor│
                     │observation    │   │Policy decide │    │  check()     │
                     │[70:94] =     │   │ERRATIC→STOP  │    │ERRATIC→STOP  │
                     │intent feats  │   │CROSS→AVOID   │    │(override any)│
                     └──────────────┘   └──────────────┘    └──────────────┘
                              │                    │                    │
                              └────────┬───────────┘                   │
                                       │                               │
                              NavigationCommand ◄──────────────────────┘
                                       │
                              ┌────────▼────────┐
                              │Experience       │  → Log to HDF5 for
                              │Collector        │    future RL training
                              └─────────────────┘
```

### 7.2 Code trong main inference loop

```python
# src/main.py — dòng 245-262
while self._running:
    frame, depth_frame = camera.grab()
    frame_det    = yolo.detect(frame, depth_frame=depth_frame)
    frame_det    = tracker.update(frame_det, frame.shape)
    rois         = roi_ex.extract(frame, frame_det)

    intent_preds = cnn.predict_batch(rois)      # ← Intent CNN ở đây

    observation  = ctx_bld.build(frame_det, intent_preds)
    cmd          = policy.decide(observation, frame_det, intent_preds)
    cmd          = safety.check(cmd, frame_det, intent_preds)
```

---

## 8. Chiến Lược Huấn Luyện

### 8.1 Trạng thái hiện tại: Phase 1 — Data Collection

Hiện tại Intent CNN chạy ở **bypass mode** (không có trained weights):

```python
# Khi không có model file (.pt):
if self._model is None:
    # Trả về uniform distribution — mỗi intent 1/6 = 16.7%
    intent_probs = np.ones((n, 6)) / 6
    directions   = np.zeros((n, 2))
```

Pipeline vẫn chạy bình thường để **thu thập data** (ảnh ROI + observation + action) qua `ExperienceCollector` → HDF5 files.

### 8.2 Training Pipeline (Dự kiến)

```
Phase 1: Data Collection (hiện tại)
├── Robot chạy với HeuristicPolicy (rule-based)
├── ExperienceCollector ghi lại mọi frame:
│   ├── raw_image_jpeg (BGR frame gốc)
│   ├── detections (bbox, track_id, distance)
│   ├── observation (104-float vector)
│   └── action (NavigationCommand → 7-float vector)
└── HDF5 files tích lũy trên Jetson → sync về Laptop

Phase 2: Labeling (tiếp theo)
├── Tách ROI crops từ HDF5 data
├── Gemma 4 VLM pre-label:
│   ├── Input: ảnh crop + context xung quanh
│   ├── Prompt: "Người này đang có ý định gì?"
│   └── Output: intent label + confidence
├── Human review + correction
└── Target: tối thiểu 10,000 labeled sequences

Phase 3: Training (trên Laptop)
├── Dataset: labeled ROI sequences + direction labels
├── Loss = CrossEntropyLoss(intent) + λ × MSELoss(direction)
│   └── λ = loss weighting hyperparameter
├── Optimizer: Adam, lr=3e-4
├── Augmentation:
│   ├── ColorJitter (brightness, contrast, saturation)
│   ├── RandomHorizontalFlip
│   ├── RandomCrop/Resize
│   └── Gaussian noise
├── Freeze backbone first N blocks, fine-tune upper layers + heads
└── Output: checkpoint.pt (state_dict)

Phase 4: Deploy
├── Copy .pt file vào models/intent_cnn/
├── Config: cnn_intent.model_path = "models/intent_cnn/intent_v1.pt"
├── No restart needed (hoặc hot-reload qua gRPC)
└── Verify: FPS không giảm dưới 15, accuracy > baseline

Phase 5 (Future): TensorRT Optimization
├── PyTorch → ONNX → TensorRT .engine (FP16)
├── Inference time giảm từ ~3ms → ~1ms per person
└── Chỉ cần export lại trên Jetson
```

### 8.3 Multi-task Loss Function

```python
# Dự kiến training loss:
total_loss = (
    CrossEntropyLoss(intent_logits, intent_labels)       # Intent classification
    + lambda_dir * MSELoss(direction_pred, direction_gt)  # Direction regression
)
# lambda_dir ≈ 0.5 — weight direction task thấp hơn intent
```

---

## 9. Vai Trò Trong Hệ Thống Navigation

### 9.1 Ảnh hưởng lên Observation Vector (104 floats)

```
Observation Vector Layout:
────────────────────────────────────────────────────────────
[0]         num_persons (normalized)
[1]         nearest_person_distance
[2]         nearest_person_angle
[3]         nearest_obstacle_distance
[4]         nearest_obstacle_angle
[5]         free_space_ratio
[6:70]      occupancy_grid (8×8 = 64 cells)

[70:94]     ★ INTENT CNN OUTPUT ★ (24 floats)
            ├── Person 0: [6 intent probs, dx, dy]  = 8 floats
            ├── Person 1: [6 intent probs, dx, dy]  = 8 floats
            └── Person 2: [6 intent probs, dx, dy]  = 8 floats

[94:97]     robot_velocity (vx, vy, vθ)
[97:104]    previous_action (v_scale, h_offset, mode×5 one-hot)
────────────────────────────────────────────────────────────

Intent CNN output chiếm 24/104 = 23% của observation vector
→ Đây là phần thông tin "high-level" nhất trong observation
```

### 9.2 Decision Flow: Từ Intent → Robot Action

```
Ví dụ: Robot đang CRUISE, phát hiện 1 người

┌─────────────────────────────────────────────────────────────┐
│ YOLO: person detected, track_id=5, distance=2.3m           │
│                                                              │
│ Intent CNN outputs:                                         │
│   P(CROSSING)=0.75, P(APPROACHING)=0.15, others low        │
│   dx=+0.6 (đi sang phải), dy=+0.1 (hơi ra xa)             │
│                                                              │
│ ┌─ Heuristic Policy Decision ──────────────────────────────┐│
│ │ RULE 3 triggered:                                         ││
│ │   intent=CROSSING, confidence=0.75 > 0.5 threshold       ││
│ │   → Mode: AVOID                                          ││
│ │   → Velocity: 0.3 × (2.3/1.0) = 0.69 → clamped to 0.3  ││
│ │   → Heading: steer opposite to person's position          ││
│ └───────────────────────────────────────────────────────────┘│
│                                                              │
│ ┌─ Safety Monitor Check ───────────────────────────────────┐│
│ │ distance=2.3m > hard_stop=0.5m → OK                      ││
│ │ intent ≠ ERRATIC → OK                                     ││
│ │ Slow-down zone: 2.3m > 1.0m → no additional slowdown     ││
│ │ Result: PASS (no override)                                ││
│ └───────────────────────────────────────────────────────────┘│
│                                                              │
│ Final Command: AVOID, v=0.3, heading=-15°                   │
│ → Robot giảm tốc và hơi tránh sang bên đối diện người       │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Hiệu Năng & Tối Ưu

### 10.1 Specifications

| Metric | PyTorch (FP16 CUDA) | PyTorch (FP32 CPU) | TensorRT (dự kiến) |
|---|---|---|---|
| Inference/person | ~2-4ms | ~15-25ms | ~1-2ms |
| Max batch | 5 persons | 5 persons | 5 persons |
| VRAM | ~30-60MB | N/A | ~20-40MB |
| Parameters | ~3M | ~3M | ~3M |
| Input size | 128×256×3 | 128×256×3 | 128×256×3 |

### 10.2 Tối ưu đã áp dụng

1. **FP16 tự động trên CUDA**: Jetson Orin Ampere hỗ trợ FP16 native → throughput gấp 2× so với FP32
2. **Batch inference**: Ghép tất cả person crops vào 1 batch → 1 forward pass cho nhiều người
3. **Bypass mode**: Khi không có model → trả về uniform distribution mà không tốn GPU
4. **Lazy import**: PyTorch/torchvision chỉ import khi `load()` được gọi → startup nhanh hơn
5. **ImageNet normalize**: Dùng stats chuẩn → tận dụng pretrained features tối đa

### 10.3 Budget trong Memory Budget hệ thống

```
Total Jetson VRAM: 8GB (shared CPU+GPU)
├── OS + System:     ~1.5GB
├── YOLOv11s:        ~80-120MB
├── Intent CNN:      ~30-60MB    ← Module này
├── RL Policy:       ~30-50MB    (future)
├── Camera buffers:  ~50-100MB
├── PyTorch runtime: ~300-500MB
└── Headroom:        ~5.5-6GB
```

---

## 11. Hạn Chế & Hướng Phát Triển

### 11.1 Hạn chế hiện tại

| Hạn chế | Mô tả | Impact |
|---|---|---|
| **Chưa có trained weights** | Model chạy bypass (uniform output) | Intent không có tác dụng thực tế trong Phase 1 |
| **Single-frame** | Chỉ nhìn 1 ảnh crop, không có temporal context | Khó phân biệt STATIONARY vs vừa dừng lại |
| **Không có pose info** | Dựa hoàn toàn vào appearance, không dùng skeleton | Có thể miss intent từ tư thế tinh tế |
| **Max 5 person** | Batch size cố định = 5 | Bỏ qua người thứ 6+ |

### 11.2 Hướng phát triển (Phase 2+)

```
Phase 2: Temporal Aggregator
├── Thêm Conv1D/LSTM trên lịch sử 5 frames per track_id
├── temporal_stack_size: 1 → 5 trong config
├── Intent accuracy kỳ vọng: +15-25%
└── Code path đã chuẩn bị trong ContextBuilder (deque ring-buffer)

Phase 3: Enhanced Features
├── Kết hợp depth map vào CNN input (RGBD 4-channel)
├── Thêm optical flow features (motion cues ngoại vi)
└── Person re-identification across occlusions

Phase 4: TensorRT Optimization
├── Export ONNX → TensorRT .engine FP16
├── Inference time: ~3ms → ~1ms per person
└── Cho phép tăng max_batch_size hoặc thêm head
```

---

## Tham Chiếu Code

| File | Vai trò |
|---|---|
| `src/perception/intent_cnn.py` | Model definition + inference |
| `src/perception/roi_extractor.py` | Crop + resize person ROI |
| `src/perception/tracker.py` | Track ID assignment (ByteTrack) |
| `src/navigation/heuristic_policy.py` | Uses intent for STOP/AVOID decisions |
| `src/navigation/safety_monitor.py` | ERRATIC override |
| `src/navigation/context_builder.py` | Packs intent into observation vector |
| `src/experience/collector.py` | Logs intent predictions to HDF5 |
| `src/main.py` | Pipeline orchestration |
| `tests/test_perception.py` | Unit tests for IntentCNN |
| `docs/system-design.md` | Section 5: CNN architecture design |

---

## 12. File HDF5 & Mối Quan Hệ Với Intent CNN

### 12.1 File HDF5 Hiện Tại Đang Lưu Gì?

File HDF5 (`session_{id}.h5`) được `ExperienceBuffer` ghi **mỗi frame inference** với cấu trúc:

```
session_abc12345.h5
│
├── frame_00000000/          ← Group cho frame đầu tiên
│   ├── [attrs] frame_id     = 0
│   ├── [attrs] timestamp    = 12345.678
│   ├── [attrs] wall_time    = 1713373200.123
│   ├── [attrs] session_id   = b"abc12345"
│   ├── [attrs] vx           = 0.3        ← Vận tốc robot
│   ├── [attrs] vy           = 0.0
│   ├── [attrs] vtheta       = 0.05
│   ├── [attrs] battery      = 87.5
│   │
│   ├── image_jpeg           dtype=uint8, shape=(N,) ← Ảnh JPEG TOÀN FRAME
│   ├── observation          dtype=float32, shape=(104,) ← Observation vector
│   ├── action               dtype=float32, shape=(7,)   ← NavigationCommand
│   ├── intent_classes       dtype=int32, shape=(K,)  ← Intent IDs per person
│   ├── intent_confs         dtype=float32, shape=(K,) ← Confidence per person
│   ├── person_distances     dtype=float32, shape=(K,) ← Distance per person
│   └── distance_sources     dtype=string, shape=(K,)  ← "depth" hoặc "bbox"
│
├── frame_00000001/
│   └── ... (cấu trúc tương tự)
│
└── frame_XXXXXXXX/
```

> **Mục đích thiết kế**: File HDF5 này được thiết kế **chủ yếu cho RL training** (Reinforcement Learning), chứa tuple `(observation, action, robot_state)` — KHÔNG phải cho CNN training.

### 12.2 File HDF5 Có Giúp Ích Cho Intent CNN Không?

**Có, nhưng CHƯA ĐỦ.** Dưới đây là phân tích chi tiết:

#### ✅ Dữ liệu CÓ THỂ tận dụng

| Dataset trong HDF5 | Dùng cho CNN như thế nào? | Cần xử lý thêm? |
|---|---|---|
| `image_jpeg` | Ảnh toàn frame → decode → crop ROI theo bbox | ⚠️ Cần bbox per person (THIẾU) |
| `observation[70:94]` | Chứa intent features hiện tại (nhưng là uniform 1/6 vì bypass) | ❌ Không hữu ích (uniform) |
| `person_distances` | Context bổ trợ — khoảng cách mỗi person | ✅ Dùng được |
| `timestamp` + `frame_id` | Sắp xếp temporal sequence per track | ⚠️ Cần track_id (THIẾU) |

#### ❌ Dữ liệu THIẾU cho CNN Training

```
┌──────────────────────────────────────────────────────────────────┐
│              GAP ANALYSIS: HDF5 vs CNN Training Needs            │
│                                                                   │
│  ❌ THIẾU #1: Bounding Box Per Person                            │
│     HDF5 lưu image_jpeg (full frame) nhưng KHÔNG lưu bbox       │
│     (x1,y1,x2,y2) của từng person                               │
│     → Không thể tách ROI crop 128×256 từ ảnh                     │
│                                                                   │
│  ❌ THIẾU #2: Track ID Per Person                                │
│     Không lưu track_id → không ghép temporal sequence            │
│     → Không biết person ở frame N là cùng person ở frame N+1     │
│                                                                   │
│  ❌ THIẾU #3: Cropped ROI Images                                 │
│     Lý tưởng nhất là lưu sẵn ảnh crop 128×256 per person        │
│     → Tránh phải decode JPEG + re-crop offline                    │
│                                                                   │
│  ❌ THIẾU #4: Intent Ground Truth Labels                         │
│     Không có intent label thật (chỉ có output từ bypass CNN)     │
│     → CẦN Gemma 4 VLM pre-label + human review                  │
│                                                                   │
│  ❌ THIẾU #5: Direction Ground Truth (dx_gt, dy_gt)              │
│     Không có hướng di chuyển thật                                │
│     → CẦN tính từ bbox displacement giữa consecutive frames     │
│                                                                   │
│  ⚠️ THIẾU #6: Số lượng person per frame                         │
│     intent_classes/intent_confs là flat arrays                   │
│     → Không biết person nào map với distance nào                 │
└──────────────────────────────────────────────────────────────────┘
```

### 12.3 Cần Thêm Dữ Liệu Gì Để Intent CNN Học Được?

Có **2 cách tiếp cận**:

#### Cách 1: Thêm dữ liệu vào HDF5 Buffer (Khuyến nghị)

Sửa `ExperienceBuffer._write_hdf5()` để lưu thêm **per-person data**:

```python
# === DỮ LIỆU CẦN THÊM VÀO HDF5 ===

# Trong mỗi frame group, thêm per-person sub-groups:
frame_00000042/
    ├── ... (giữ nguyên datasets hiện tại) ...
    │
    ├── num_persons        int           ← Số person trong frame
    │
    ├── person_000/                      ← Sub-group cho person thứ 0
    │   ├── [attrs] track_id    = 5      ← Track ID (từ ByteTrack)
    │   ├── [attrs] bbox_x1     = 120    ← Bounding box
    │   ├── [attrs] bbox_y1     = 80
    │   ├── [attrs] bbox_x2     = 230
    │   ├── [attrs] bbox_y2     = 450
    │   ├── [attrs] distance    = 2.3    ← Khoảng cách (metres)
    │   ├── [attrs] dist_source = "depth"
    │   ├── [attrs] confidence  = 0.92   ← YOLO confidence
    │   │
    │   ├── roi_jpeg     uint8 (M,)      ← ★ Ảnh crop 128×256 JPEG ★
    │   │
    │   │  --- Labels (CẦN THÊM SAU KHI ANNOTATE) ---
    │   ├── [attrs] intent_label  = 3    ← ★ Ground truth: CROSSING ★
    │   ├── [attrs] dx_gt         = 0.6  ← ★ Ground truth direction ★
    │   └── [attrs] dy_gt         = 0.1
    │
    ├── person_001/
    │   └── ... (tương tự)
    │
    └── person_002/
        └── ... (tương tự)
```

#### Cách 2: Script Offline Extraction

Viết script tách ROI từ HDF5 hiện tại → dataset mới cho CNN:

```python
# scripts/extract_cnn_dataset.py (pseudo-code)
"""Trích xuất CNN training data từ HDF5 experience."""

import h5py, cv2, numpy as np

def extract_from_hdf5(h5_path, output_dir):
    with h5py.File(h5_path, 'r') as f:
        for frame_name in sorted(f.keys()):
            grp = f[frame_name]

            # 1. Decode full frame JPEG
            jpeg_bytes = grp["image_jpeg"][:]
            frame = cv2.imdecode(
                np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
            )

            # 2. Cần bbox — HIỆN TẠI KHÔNG CÓ TRONG HDF5!
            #    → Phải chạy YOLO lại trên ảnh đã decode
            #    → HOẶC sửa buffer.py để lưu bbox từ đầu
            detections = re_run_yolo(frame)  # costly!

            # 3. Crop ROI per person
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                roi = crop_and_resize(frame, x1, y1, x2, y2)
                save_roi(output_dir, frame_name, det.track_id, roi)

            # 4. Vẫn THIẾU intent labels → cần VLM hoặc human annotation
```

> ⚠️ **Cách 2 rất tốn chi phí** vì phải chạy lại YOLO trên toàn bộ frames đã lưu. **Cách 1 (sửa buffer)** tiết kiệm hơn nhiều.

### 12.4 Tổng Hợp: Dữ Liệu Cần Có Cho CNN Training

```
┌──────────────────────────────────────────────────────────────────────┐
│ BẢNG TỔNG HỢP: Dữ Liệu Cần Cho Intent CNN Training                │
├────────────────────────┬──────────┬──────────┬──────────────────────┤
│ Dữ liệu               │ Hiện có? │ Bắt buộc │ Cách lấy             │
├────────────────────────┼──────────┼──────────┼──────────────────────┤
│ ROI crop 128×256       │ ❌       │ ✅ BẮT   │ Thêm vào buffer.py   │
│                        │          │ BUỘC     │ hoặc re-extract      │
├────────────────────────┼──────────┼──────────┼──────────────────────┤
│ Bbox (x1,y1,x2,y2)    │ ❌       │ ✅ BẮT   │ Thêm vào buffer.py   │
│                        │          │ BUỘC     │                      │
├────────────────────────┼──────────┼──────────┼──────────────────────┤
│ Track ID               │ ❌       │ ✅ BẮT   │ Thêm vào buffer.py   │
│                        │          │ BUỘC     │ (đã có trong         │
│                        │          │          │  DetectionResult)    │
├────────────────────────┼──────────┼──────────┼──────────────────────┤
│ Intent label (GT)      │ ❌       │ ✅ BẮT   │ VLM pre-label +      │
│ (0-5 class)            │          │ BUỘC     │ human review         │
├────────────────────────┼──────────┼──────────┼──────────────────────┤
│ Direction label (GT)   │ ❌       │ ✅ BẮT   │ Tính từ bbox         │
│ (dx_gt, dy_gt)         │          │ BUỘC     │ displacement giữa    │
│                        │          │          │ consecutive frames   │
├────────────────────────┼──────────┼──────────┼──────────────────────┤
│ Full frame JPEG        │ ✅       │ ⚠️ Có   │ Đã có (image_jpeg)   │
│                        │          │ thì tốt  │                      │
├────────────────────────┼──────────┼──────────┼──────────────────────┤
│ Person distance        │ ✅       │ ⚠️ Có   │ Đã có                │
│                        │          │ thì tốt  │ (person_distances)   │
├────────────────────────┼──────────┼──────────┼──────────────────────┤
│ YOLO confidence        │ ❌       │ ⚠️ Phụ  │ Thêm vào buffer.py   │
├────────────────────────┼──────────┼──────────┼──────────────────────┤
│ Timestamp              │ ✅       │ ✅ BẮT   │ Đã có                │
│                        │          │ BUỘC     │                      │
├────────────────────────┼──────────┼──────────┼──────────────────────┤
│ Relative position      │ ❌       │ ⚠️ Phụ  │ Tính từ bbox +       │
│ (cx_norm, cy_norm)     │          │          │ frame size           │
└────────────────────────┴──────────┴──────────┴──────────────────────┘
```

### 12.5 Tính Direction Label Tự Động (dx_gt, dy_gt)

Direction ground truth **có thể tính tự động** từ bbox displacement giữa 2 frame liên tiếp cùng track_id:

```python
# Pseudo-code: tính direction từ tracking data
def compute_direction_label(
    bbox_t: tuple,     # (x1,y1,x2,y2) tại frame t
    bbox_t1: tuple,    # (x1,y1,x2,y2) tại frame t+1
    frame_w: int,
    frame_h: int,
) -> tuple[float, float]:
    """Tính (dx_gt, dy_gt) từ bbox center displacement."""
    cx_t  = (bbox_t[0]  + bbox_t[2])  / 2.0
    cy_t  = (bbox_t[1]  + bbox_t[3])  / 2.0
    cx_t1 = (bbox_t1[0] + bbox_t1[2]) / 2.0
    cy_t1 = (bbox_t1[1] + bbox_t1[3]) / 2.0

    # Pixel displacement → normalized [-1, 1]
    dx = (cx_t1 - cx_t) / (frame_w / 2)    # >0 = sang phải
    dy = (cy_t1 - cy_t) / (frame_h / 2)    # >0 = xuống (xa robot)

    # Clamp to [-1, 1]
    dx = max(-1.0, min(1.0, dx))
    dy = max(-1.0, min(1.0, dy))

    return dx, dy
```

> 💡 **Lưu ý**: Direction label tính tự động chỉ phản ánh **chuyển động pixel** (apparent motion), không phải **ý định thật** (true intent). Tuy nhiên, đây là approximation tốt nhất khi không có sensor chuyên dụng.

### 12.6 Số Lượng Mẫu Tối Thiểu

| Dataset Scale | Số ROI Samples | Thời gian thu thập | Chất lượng kỳ vọng |
|---|---|---|---|
| **Minimum viable** | 5,000 labeled | ~3-5 giờ chạy robot | Accuracy ~60-70% |
| **Recommended** | 10,000-20,000 | ~10-20 giờ | Accuracy ~75-85% |
| **Production** | 50,000+ | ~50+ giờ + diverse environments | Accuracy ~85-95% |

**Phân bố class nên nhắm tới:**

```
STATIONARY:   ~30% (phổ biến nhất — người đứng chờ)
APPROACHING:  ~20% (người đi về phía robot)
DEPARTING:    ~15% (người đi xa)
CROSSING:     ~20% (người cắt ngang)
FOLLOWING:    ~10% (người đi theo robot)
ERRATIC:      ~5%  (hiếm nhưng quan trọng — cần augmentation)
```

> ⚠️ **Class imbalance**: ERRATIC sẽ rất hiếm trong dữ liệu thực → cần **oversampling**, **data augmentation**, hoặc **focal loss** khi training.

### 12.7 Tóm Tắt: 3 Bước Để HDF5 → CNN Training

```
BƯỚC 1: Sửa ExperienceBuffer (buffer.py)
────────────────────────────────────────
Thêm per-person sub-groups với:
  • track_id
  • bbox (x1,y1,x2,y2)
  • roi_jpeg (cropped 128×256)
  • YOLO confidence

→ Data mới sẽ tự động lưu từ inference loop

BƯỚC 2: Thu Thập & Label
────────────────────────
• Chạy robot trong các môi trường đa dạng
• Sync HDF5 về laptop
• VLM (Gemma 4) pre-label: intent_label per ROI
• Tính dx_gt, dy_gt tự động từ bbox tracking
• Human review + correction qua tool

BƯỚC 3: Build Dataset & Train
─────────────────────────────
• Script đọc HDF5 → PyTorch Dataset
• Filter: chỉ lấy ROIs có intent_label valid
• Split: 80% train, 10% val, 10% test
• Train MobileNetV3 + dual-head
• Evaluate → Export .pt → Deploy về Jetson
```


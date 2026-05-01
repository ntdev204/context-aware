# Phase 1.5 — Intent CNN Training Server

> **Slug**: `intent-training-server`
> **Project Type**: AI / ML Training Pipeline (Backend)
> **Created**: 2026-04-18
> **Updated**: 2026-04-18 — v5: Combined with ROI Specs
> **Status**: ✅ COMPLETED

---

## Goal

Biến Laptop thành **Training Server** tự động:
- Khi có batch ROI mới từ Jetson → tự động khai phá → tiền xử lý → fine-tune model
- Hỗ trợ cả **auto-labeled data** (6 class) lẫn **human-labeled data** (Label Studio / Roboflow)
- Tách môi trường Docker cho Jetson (inference) vs Laptop (training)
- Dataset chứa trong **Docker volume**, không nằm trong codebase

---

## Quyết định kiến trúc

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Codebase | Giữ nguyên monorepo | Đỡ maintain 2 repo, chia sẻ model definition |
| Docker | 2 profiles: `jetson-prod` (có sẵn) + `laptop-train` (mới) | Cùng `docker-compose.yml`, khác profile |
| Dataset storage | Docker named volume `intent-dataset` + `roi-incoming` | Không pollute codebase, dễ backup |
| Auto-label | Nâng từ 3 class → 6 class, dùng **depth distance** + cx/cy | Bbox size ≠ khoảng cách (ngã, cúi → bbox to nhưng không approach) |
| Distance source | **Depth camera (Astra S)** primary, bbox ratio secondary | Depth đo khoảng cách thật (mm → m), không bị lừa bởi tư thế |
| Human label | Label Studio self-hosted (Docker) hoặc Roboflow | Export YOLO-format → convert |
| Training mode | Fine-tune (continual) | Tránh mất knowledge khi có data mới |
| Anti-forgetting | EWC (Elastic Weight Consolidation) hoặc Replay Buffer | Standard cho continual learning |

### Tại sao dùng Depth thay vì Bbox size?

```
❌ CŨ (bbox-only):                    ✅ MỚI (depth + bbox):

bbox_area tăng → "APPROACHING"         depth_mm giảm → "APPROACHING"
                                       depth_mm tăng → "DEPARTING"
Vấn đề:                               
• Người ngã  → bbox to → sai label    Lợi ích:
• Người cúi  → bbox to → sai label    • Đo khoảng cách vật lý thật
• Người vẫy tay → bbox dao động       • Không bị ảnh hưởng bởi tư thế
• FOV edge distortion                 • Fallback bbox khi depth invalid
```

**Data đã có sẵn**: `DetectionResult.distance` (metres) và `DetectionResult.distance_source` ("depth" | "bbox") — chỉ cần truyền vào `roi_collector.py`.

---

## Sơ đồ tổng thể

```
JETSON (inference)                       LAPTOP (training server)
━━━━━━━━━━━━━━━━━━━                      ━━━━━━━━━━━━━━━━━━━━━━━━━━
                                          Docker: laptop-train profile
YOLO + DepthCam → Track → ROI           ┌─────────────────────────┐
  │  distance(m), distance_source        │                         │
  ▼ cứ 5000 ảnh                          │  Volume: roi-incoming   │
batch_*.tar.gz ──rsync─────────────────► │  ↓                      │
  (JPG + metadata.jsonl per batch)       │  auto_watcher.py        │
                                         │  ↓ detect new batch     │
                                         │  ┌──────────────────┐   │
                                         │  │ 1. EXTRACT       │   │
                                         │  │ 2. EXPLORE       │←── explore_roi.py
                                         │  │ 3. AUTO-LABEL    │←── autolabel.py (6 classes)
                                         │  │    depth Δd(t)   │    + depth distance series
                                         │  │    + cx/cy Δ     │    + spatial movement
                                         │  │ 4. VALIDATE      │←── validate_dataset.py
                                         │  │ 5. MERGE         │   │
                                         │  └────────┬─────────┘   │
                                         │           ▼             │
                                         │  Volume: intent-dataset │
                                         │  ├─ auto/   (auto-labeled)
                                         │  ├─ human/  (Label Studio)
                                         │  └─ merged/ (combined)  │
                                         │           ▼             │
                                         │  ┌──────────────────┐   │
                                         │  │ 6. FINE-TUNE     │   │
                                         │  │    (continual)   │   │
                                         │  └────────┬─────────┘   │
                                         │           ▼             │
                                         │  models/cnn_intent/     │
                                         │  intent_v1.pt (updated) │
                                         └─────────────────────────┘
                                                     │
                                                     ▼ rsync / scp
                                             Deploy lên Jetson
        models/cnn_intent/intent_v1.pt ◄────────────┘
```

---

## ROI Dataset Collection Specification

> **Status:** Active — replaces HDF5 ExperienceBuffer for INTENT_CNN dataset collection.

### ROI Filename Format

```
roi_<track_id>_<cx>_<cy>_<dist_mm>_<dsrc>_<fid>_<ts_ms>.jpg
     │          │    │    │          │       │     │
     │          │    │    │          │       │     └── Unix timestamp (ms)
     │          │    │    │          │       └──────── Frame ID in session
     │          │    │    │          └──────────────── Distance source: d=depth, b=bbox
     │          │    │    └─────────────────────────── Distance in mm (1523 = 1.523m)
     │          │    └──────────────────────────────── Y centre of bbox (pixel)
     │          └───────────────────────────────────── X centre of bbox (pixel)
     └──────────────────────────────────────────────── ByteTrack track ID
```

**Example:** `roi_7_354_240_1523_d_000123_1713420001234.jpg`
- Track ID=7, centre x=354px, y=240px, distance=1.523m (depth camera), frame 123

### Sidecar Metadata

Each batch archive also contains `metadata.jsonl` — one JSON object per image:
```jsonl
{"file":"roi_7_354_240_1523_d_000123_1713420001234.jpg","tid":7,"cx":354,"cy":240,"dist_mm":1523,"dsrc":"depth","bw":89,"bh":210,"fid":123,"ts":1713420001234}
```
The sidecar stores `bbox_w/h` and other fields that exceed filename length limits, and is the primary source for depth-aware auto-labeling.

### Runtime Label Mapping

| Class | ID | Auto-label signal |
|---|---|---|
| **STATIONARY** | 0 | `abs(Δdist) < 100mm AND abs(Δcx) < threshold` |
| **APPROACHING** | 1 | `dist[t+N] - dist[t] < -200mm` |
| **DEPARTING** | 2 | `dist[t+N] - dist[t] > +200mm` |
| **CROSSING** | 3 | `abs(Δcx) > threshold AND abs(Δdist) < 150mm` |
| **ERRATIC** | 4 | high variance/sign changes; requires human review |
| **UNCERTAIN** | 5 | abstain; skipped unless manually relabeled |

### Logging (Production)

Per-component log files are written under `/app/logs/` (Jetson) or `logs/` (Laptop):
| File | Contents |
|---|---|
| `logs/perception.log` | YOLO, tracker, intent CNN |
| `logs/experience.log` | ROI collector, batch shipping |
| `logs/errors.log` | ERROR+ from all components |
Console output (production): **INFO+** with color (green=INFO, yellow=WARNING, red=ERROR).

---

## File Structure Changes

```
context-aware/
├── docker/
│   ├── Dockerfile.jetson               # unchanged
│   ├── Dockerfile.dev                  # unchanged
│   ├── Dockerfile.laptop               # DONE — training server image
│   └── docker-compose.yml              # DONE — laptop-train profile + volumes
│
├── config/
│   ├── production.yaml                 # DONE — logging config
│   ├── development.yaml                # DONE — logging config
│   └── training.yaml                   # DONE — continual fine-tuning bounds
│
├── scripts/
│   ├── data/
│   │   ├── autolabel.py                # DONE — 6 class depth-aware logic
│   │   ├── auto_watcher.py             # DONE — full pipeline orchestrator
│   │   ├── explore_data.py             # DONE
│   │   ├── explore_roi.py              # DONE — data exploration + HTML report
│   │   ├── import_labels.py            # DONE — human label import
│   │   └── validate_dataset.py         # DONE — pre-train quality gate
│   ├── train/
│   │   └── train_intent_cnn.py         # DONE — EWC & Replay buffer logic added
│   ├── deploy/
│   │   ├── automate_benchmarks.py
│   │   ├── benchmark.py
│   │   ├── export_engine.py
│   │   └── model_manager.py
│   └── infra/
│       ├── generate_proto.py
│       ├── health_check.py
│       ├── setup_ssh_server.ps1
│       └── sync_experience.py
│
├── src/
│   ├── logging_config.py               # DONE
│   ├── experience/
│   │   └── roi_collector.py            # DONE
│   └── main.py                         # DONE
│
└── Makefile                            # DONE — targets updated to matching structure
```

---

## Verification Checklist

### Milestone 1 — Docker ✅
- [x] `make laptop-build` succeeds
- [x] `make laptop-up` starts container, watcher listening
- [x] `make jetson-up` still works (not broken)

### Milestone 2 — Depth-Aware Data ✅
- [x] ROI filename contains depth: `roi_7_354_240_1523_d_000123_*.jpg`
- [x] Sidecar `metadata.jsonl` present in each batch archive
- [x] Auto-label depth test: person falls (dist stable, bbox grows) → NOT labeled APPROACHING
- [x] Fake batch `.tar.gz` → watcher detects → extract → labels 6 classes
- [x] `train_intent_cnn.py` outputs 6-class confusion matrix

### Milestone 3 — Data Exploration ✅
- [x] `explore_roi.py` generates HTML report with all 8 analysis sections
- [x] Corrupt image injected → detected and flagged
- [x] `validate_dataset.py` → exit 0 on clean data, exit 1 on < 500 images

### Milestone 4 — Continual Training ✅
- [x] Fake batch → full pipeline: extract → explore → validate → fine-tune
- [x] New checkpoint `.pt` saved after training
- [x] `training_log.csv` records new run
- [x] Fine-tune round 2 → accuracy on old data drops ≤ 10%
- [x] `import_labels.py` → human labels appear in `merged/`

### Cross-cutting ✅
- [x] Scripts reorganized into logical subfolders (`data/`, `train/`, `deploy/`, `infra/`)
- [x] Structured logging: per-component files under `logs/`
- [x] Console shows INFO+ with ANSI color (green/yellow/red)
- [x] `logs/errors.log` captures ERROR+ from all components
- [x] Codebase cleaned: no Vietnamese comments in code, no unicode separators, no emoji in code

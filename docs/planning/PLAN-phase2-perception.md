# PLAN - Phase 2: Temporal Intent CNN + Dataset Gate

> **Updated:** 2026-05-01  
> **Status:** implementation baseline added  
> **Scope:** dataset chuẩn trước khi export model phase 2

## Decisions

- `FOLLOW`/`FOLLOWING` is removed. Person-following is not a navigation behavior and is not an intent class.
- Runtime intent vector remains 6 slots for observation compatibility: 5 trainable labels + `UNCERTAIN`.
- Trainable labels: `STATIONARY`, `APPROACHING`, `DEPARTING`, `CROSSING`, `ERRATIC`.
- `UNCERTAIN` is an abstain/review class and is excluded from training.
- `ERRATIC` must be human reviewed before it is allowed into training.
- Dataset split must be by `track_uid`, not random per image.

## Pipeline

1. Collect ROI sequences with `ROISaver`.
2. Auto-label with depth, lateral motion, ego-motion compensation, and sliding window.
3. Route `UNCERTAIN` and `ERRATIC` to `review_queue/`.
4. Explore dataset and validate quality gates.
5. Build `manifest.json` as the phase-2 dataset gate artifact.
6. Train Temporal Intent CNN with MobileNetV3-Small + lightweight TCN/Conv1D on 15-frame ROI sequences.
7. Fit confidence calibration temperature on validation logits.
8. Export TorchScript + metadata sidecar.

## Commands

```bash
python scripts/data/auto_watcher.py --watch roi_dataset --output intent_dataset
python scripts/data/explore_roi.py --dataset intent_dataset --output intent_dataset
python scripts/data/validate_dataset.py intent_dataset/reports/<latest>.json
python scripts/data/build_intent_manifest.py --dataset intent_dataset --temporal-window 15
python scripts/train/train_intent_cnn.py --dataset intent_dataset --temporal-window 15
python scripts/deploy/export_intent_model.py --checkpoint models/cnn_intent/intent_v1.pt
```

## Model

```text
Input: (B, T=15, 3, 256, 128)
MobileNetV3-Small shared frame encoder
Depthwise Conv1D + Pointwise Conv1D temporal aggregator
Intent head: 5 trainable classes
Direction head: dx, dy
Runtime: temperature softmax + confidence/margin abstain -> UNCERTAIN
```

## Quality Gates

| Gate | Rule |
| --- | --- |
| Ontology | `FOLLOW` and `FOLLOWING` count must be zero |
| Review | Pending `ERRATIC` review must be zero |
| Trainability | At least 500 trainable ROI samples |
| Temporal | At least 2 tracks for track-level split |
| Dataset | `manifest.json` must be generated before model export |

## Optimization Path

- Confidence calibration: scalar temperature saved in checkpoint metadata.
- Continual learning: EWC + replay buffer enabled in training script.
- Quantization: `export_intent_model.py --quantize-dynamic` for CPU dynamic quantization experiments.
- Distillation: `--distill-from` reserved in the training CLI for teacher-student experiments.

# Temporal Intent CNN

> Updated: 2026-05-01

This module predicts human movement intent from tracked ROI sequences. It is no longer a single-frame classifier and it no longer contains any person-following behavior.

## Ontology

Runtime probabilities use six slots for backward-compatible observation layout:

| ID | Label | Role |
| --- | --- | --- |
| 0 | `STATIONARY` | Trainable |
| 1 | `APPROACHING` | Trainable |
| 2 | `DEPARTING` | Trainable |
| 3 | `CROSSING` | Trainable |
| 4 | `ERRATIC` | Trainable only after review |
| 5 | `UNCERTAIN` | Abstain/review, excluded from training |

`FOLLOW` and `FOLLOWING` are intentionally removed. Legacy samples with those labels are mapped to `UNCERTAIN` and blocked by the dataset manifest until cleaned.

## Architecture

```text
Input: (B, T=15, 3, 256, 128)
MobileNetV3-Small frame encoder
Depthwise Conv1D + pointwise Conv1D temporal aggregator
Intent head: 5 trainable logits
Direction head: dx, dy
Runtime calibration: temperature softmax + confidence/margin abstain
```

## Dataset Gate

Required before phase-2 export:

```bash
python scripts/data/explore_roi.py --dataset intent_dataset --output intent_dataset
python scripts/data/validate_dataset.py intent_dataset/reports/<latest>.json
python scripts/data/build_intent_manifest.py --dataset intent_dataset --temporal-window 15
```

The manifest must show:

- no legacy `FOLLOW`/`FOLLOWING` samples;
- no pending `ERRATIC` review;
- enough trainable images;
- at least two tracks for track-level train/validation split.

## Training And Export

```bash
python scripts/train/train_intent_cnn.py --dataset intent_dataset --temporal-window 15
python scripts/deploy/export_intent_model.py --checkpoint models/cnn_intent/intent_v1.pt
```

Training uses EWC + replay buffer for controlled continual learning. Checkpoints include confidence-calibration temperature and runtime abstention thresholds.

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
python scripts/data/autolabel.py --batch-dir roi_dataset/<raw_session_or_extracted_zip> --output-dir intent_dataset
python scripts/data/explore_roi.py --dataset intent_dataset --output intent_dataset
python scripts/data/validate_dataset.py intent_dataset/reports/<latest>.json
python scripts/data/build_intent_manifest.py --dataset intent_dataset --temporal-window 15
```

The manifest must show:

- no legacy `FOLLOW`/`FOLLOWING` samples;
- no pending `ERRATIC` review;
- enough trainable images;
- at least two tracks for track-level train/validation split.

## Fast Natural-Run Collection

For urgent data collection, do not design manual action scenarios. Run the robot in a normal environment with people naturally moving around it, record `intent_cnn` sessions from the Edge API, then download the raw dataset zip. The Jetson writes `track_sequence_v1` data:

```text
session_001/
  track_0001/
    frames/0001.jpg ...
    meta.json
```

`autolabel.py` now reads this raw sequence format directly. Labels are inferred from each person's full tracked ROI sequence using depth change, lateral motion, and robot ego-motion compensation. In `rai_website`, the server can also ask the configured VLM to inspect sampled ROI frames and the configured LLM to arbitrate between heuristic and VLM proposals. These AI labels are proposals only: every sequence remains `needs_review` until a user approves or corrects it in the web UI. `UNCERTAIN` is excluded from training, and `ERRATIC` remains review-gated.

The trainable artifact is sequence-level. `sequence_manifest.jsonl` stores one K-frame sample per track, from first appearance until the track disappears from the frame, with one label for that whole track. The per-label image folders and `metadata.jsonl` are retained only for visual inspection and review queues. During training, `--temporal-window` is only the model input length: short tracks are padded and long tracks are sampled across the whole K-frame track.

Practical collection loop in `rai_website`:

1. Open `Dataset`.
2. Press `Start` in Robot Collection and drive the robot naturally.
3. Press `Stop`, then `Save`, then `Import`.
4. Press `Auto label`; heuristic, VLM, and LLM agents propose one label for each whole track.
5. Review every pending track and use the intent dropdown to approve or correct the sequence label.
6. Open `Training`, keep dataset as `latest`, set `Model Frames` as the CNN input length, and start training.

Use unreviewed `ERRATIC` only for an emergency first model. For the deployable research checkpoint, review the web queue and train without allowing unreviewed `ERRATIC`.

## Training And Export

```bash
python scripts/train/train_intent_cnn.py --dataset intent_dataset --temporal-window 15
python scripts/deploy/export_intent_model.py --checkpoint models/cnn_intent/intent_v1.pt
```

Training uses EWC + replay buffer for controlled continual learning. Checkpoints include confidence-calibration temperature and runtime abstention thresholds.

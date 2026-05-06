# PLAN - Phase 2: Temporal Intent CNN + Dataset Gate

> **Updated:** 2026-05-01  
> **Status:** planned / implementation baseline exists, but phase-2 training is not approved until dataset gates pass  
> **Scope:** collect a clean temporal ROI dataset, train the first exportable Temporal Intent CNN, calibrate confidence, and prepare the artifact for runtime evaluation on Jetson.

Phase 2 is the perception-training phase. Phase 1 proved that the robot can run detection, tracking, ROI extraction, asynchronous intent inference, STOP-first safety behavior, and dataset logging. Phase 2 turns that foundation into a trainable and exportable temporal intent model. The main output is not only a checkpoint; it is a controlled dataset pipeline with review gates so the model can be retrained later without reintroducing noisy labels or legacy behavior.

The phase must be treated as unfinished until the dataset manifest, validation report, trained checkpoint, calibration metadata, and export artifact all exist together and describe the same ontology.

---

## 1. Phase Objectives

### 1.1 Primary objectives

1. Build a standard ROI sequence dataset for human movement intent.
2. Remove person-following behavior from the ontology and from all training assumptions.
3. Use whole-track K-frame temporal samples instead of single-frame snapshot samples.
4. Train a lightweight Temporal Intent CNN that can run on the Jetson perception path.
5. Add confidence calibration and abstention so low-quality predictions become `UNCERTAIN` instead of unsafe high-confidence labels.
6. Use heuristic, VLM, and LLM labeling as web-visible proposals, with human approval required before any sequence is allowed to influence training.
7. Export a versioned model artifact with metadata that exactly matches runtime class mapping.

### 1.2 Secondary objectives

1. Prepare the dataset for future continual learning from real logs.
2. Keep the training pipeline reproducible from CLI commands.
3. Add quality reports so dataset problems are visible before training.
4. Establish a baseline for later TCN/GRU/LSTM experiments without changing the dataset contract again.
5. Preserve compatibility with the observation vector shape used by downstream modules, while making the model API temporal-native.

### 1.3 Non-goals

Phase 2 does not implement autonomous person following. That behavior was removed deliberately. The robot may use human intent to slow, stop, avoid, or plan later, but it should not track and follow a person as a navigation objective.

Phase 2 does not train the final RL policy. It prepares perception outputs and temporal logs that Phase 3 can use for reward shaping and policy work.

Phase 2 does not claim that `UNCERTAIN` is a learned class. `UNCERTAIN` is an abstain/review state created by confidence gating, margin gating, and human-in-the-loop data handling.

---

## 2. Current Decisions

### 2.1 Intent ontology

The active ontology is:

| ID | Runtime Slot | Trainable | Meaning |
| --- | --- | --- | --- |
| 0 | `STATIONARY` | Yes | Person is approximately static relative to the robot/camera. |
| 1 | `APPROACHING` | Yes | Person is moving toward the robot or depth is decreasing. |
| 2 | `DEPARTING` | Yes | Person is moving away from the robot or depth is increasing. |
| 3 | `CROSSING` | Yes | Person is moving laterally across the robot path. |
| 4 | `ERRATIC` | Yes, after review | Person movement is unstable, abrupt, or difficult to predict. |
| 5 | `UNCERTAIN` | No | Runtime abstain/review state for low-confidence or ambiguous samples. |

`FOLLOW` and `FOLLOWING` are removed from the active behavior model. Legacy samples with these labels must be treated as invalid for training and mapped into review/cleanup flow, not silently trained as a new behavior.

### 2.2 Temporal API decision

The model and dataset API are temporal-native:

```text
Dataset sample image shape: (T, C, H, W)
Batch model input shape:    (B, T, C, H, W)
Default temporal window:    T = 15
ROI size:                   256 x 128
Channels:                   RGB, C = 3
```

Even if `temporal_window=1` is used for debugging, the dataset must return `(1, C, H, W)`. The old snapshot API `(C, H, W)` or `(B, C, H, W)` is not supported in the intent path.

### 2.3 Model decision

The phase-2 baseline model is:

```text
15 ROI frames per track
  -> shared MobileNetV3-Small frame encoder
  -> temporal Conv1D head
  -> intent classification head
  -> direction regression head
  -> temperature softmax
  -> confidence/margin abstain gate
```

This is a Temporal Intent CNN, not a pure TCN and not a single-frame CNN. The temporal Conv1D head is intentionally lightweight for the first exportable baseline. A deeper dilated TCN, GRU, or LSTM can be evaluated later using the same dataset contract.

### 2.4 Review decision

`UNCERTAIN` and `ERRATIC` must be visible to humans:

- `UNCERTAIN` is excluded from training.
- `ERRATIC` is blocked from training until reviewed.
- The review queue stores images and metadata so the operator can confirm, correct, or reject samples.
- Dataset validation must fail if pending `ERRATIC` review exists.

### 2.5 Split decision

Train/validation split must happen by `track_uid`, not by random image. A single person track must not leak across train and validation splits because temporal samples from the same track are highly correlated.

---

## 3. Phase Inputs and Outputs

### 3.1 Inputs from Phase 1

| Input | Source | Required for Phase 2 |
| --- | --- | --- |
| ROI images | `ROISaver` / Jetson runtime | Training image data |
| `metadata.jsonl` | ROI logging sidecar | Track, frame, bounding box, depth, velocity, and timestamp fields |
| Tracker IDs | ByteTrack | Temporal grouping |
| Depth or depth proxy | RealSense or bbox fallback | Auto-label approach/depart decision |
| Robot ego-motion | runtime metadata | Compensation for robot movement |
| Runtime configs | `config/*.yaml` | Temporal window, thresholds, model path |

### 3.2 Phase outputs

| Output | Location | Purpose |
| --- | --- | --- |
| Labeled ROI dataset | `intent_dataset/<CLASS>/...` | Trainable image sequences |
| Dataset metadata | `intent_dataset/metadata.jsonl` | One metadata row per ROI image |
| Review queue | `intent_dataset/review_queue/` | Human review for uncertain/risky samples |
| Dataset report | `intent_dataset/reports/*.json` | Explore and validation artifact |
| Dataset manifest | `intent_dataset/manifest.json` | Phase-2 gate artifact |
| Checkpoint | `models/cnn_intent/intent_v1.pt` | Trainable PyTorch checkpoint |
| Exported artifact | TorchScript + metadata sidecar | Runtime deployable model |
| Calibration metadata | checkpoint/export metadata | Temperature, thresholds, class map |

---

## 4. Dataset Layout

### 4.1 Raw ROI collection layout

Raw collection is session-oriented. A session contains many ROI images and one sidecar metadata file:

```text
roi_dataset/
└── session_2026_05_01_001/
    ├── roi_t1_f000015.jpg
    ├── roi_t1_f000030.jpg
    ├── roi_t2_f000045.jpg
    └── metadata.jsonl
```

Each JSONL row maps to one ROI image:

```json
{"file":"roi_t1_f000015.jpg","track_id":1,"frame_id":15,"cx":320.5,"cy":210.0,"bw":62.0,"bh":180.0,"dist_mm":2150.0,"timestamp":1777580000.123}
```

### 4.2 Labeled dataset layout

After auto-labeling and manual import:

```text
intent_dataset/
├── STATIONARY/
├── APPROACHING/
├── DEPARTING/
├── CROSSING/
├── ERRATIC/
├── UNCERTAIN/
├── review_queue/
│   ├── ERRATIC/
│   ├── UNCERTAIN/
│   └── metadata.jsonl
├── reports/
├── metadata.jsonl
├── imported_metadata.jsonl
└── manifest.json
```

`metadata.jsonl` is the main sidecar generated by auto-labeling. `imported_metadata.jsonl` is reserved for manual import tooling so concurrent import and auto-label jobs do not append to the same file.

### 4.3 Metadata contract

The dataset sidecar must preserve enough information to rebuild temporal sequences and diagnose labels:

| Field | Required | Purpose |
| --- | --- | --- |
| `file` | Yes | Relative path to ROI image |
| `label` / `intent` | Yes | Assigned class |
| `track_id` | Yes | Tracker ID within session |
| `track_uid` | Yes | Stable split key across sessions |
| `frame_id` | Yes | Temporal ordering |
| `timestamp` | Recommended | `dt` calculation and audit |
| `cx`, `cy`, `bw`, `bh` | Recommended | Crossing, bbox depth proxy, quality analysis |
| `dist_mm` | Recommended | Approach/depart labels |
| `review_required` | Yes for review classes | Human-in-the-loop gate |
| `review_status` | Yes for review classes | `pending`, `approved`, `rejected`, or corrected label |
| `autolabel_reason` | Recommended | Debuggable label provenance |

---

## 5. End-to-End Workflow

### 5.1 Step A - Collect ROI sequences

Run the robot in a controlled environment with realistic pedestrian movement:

1. Person standing still.
2. Person approaching the robot.
3. Person departing from the robot.
4. Person crossing left-to-right and right-to-left.
5. Person moving irregularly, stopping suddenly, or changing direction.
6. Background scenes without relevant movement, to test false positives and low-confidence outputs.

The collection must keep `track_id`, `frame_id`, timestamp, bounding box, and depth information. Each dataset sample is the full track from first appearance until disappearance; training may pad or sample that K-frame track into the model input length.

### 5.2 Step B - Auto-label raw ROI data

Auto-labeling groups frames by track and estimates motion using:

- depth change over time,
- lateral center movement over time,
- bounding-box fallback when real depth is missing,
- focal-length scaling based on frame width,
- ego-motion compensation from robot velocity,
- sliding-window stability checks.

Expected command:

```bash
python scripts/data/auto_watcher.py --watch roi_dataset --output intent_dataset
```

Manual single-batch equivalent:

```bash
python scripts/data/autolabel.py --input roi_dataset/session_2026_05_01_001 --output intent_dataset
```

### 5.3 Step C - Human review

Review must happen before training if there are pending `ERRATIC` samples. `UNCERTAIN` samples can remain out of training, but they should still be inspected because they reveal dataset gaps.

Review actions:

| Action | Meaning |
| --- | --- |
| approve `ERRATIC` | Sample can be used for `ERRATIC` training |
| relabel | Sample is moved to one of the trainable classes |
| reject | Sample stays excluded from training |
| keep `UNCERTAIN` | Sample remains an abstain/review example, not a train example |

The review process should update metadata, not only move images. Training decisions must be derived from metadata gates.

### 5.4 Step D - Explore dataset

Run:

```bash
python scripts/data/explore_roi.py --dataset intent_dataset --output intent_dataset
```

The report must answer:

1. How many samples exist per class?
2. How many unique tracks exist per class?
3. How many short tracks exist?
4. How many samples lack sidecar metadata?
5. How many review-pending samples exist by class?
6. Is there obvious class imbalance?
7. Are there duplicate or missing image references?

### 5.5 Step E - Validate dataset

Run:

```bash
python scripts/data/validate_dataset.py intent_dataset/reports/<latest>.json
```

Validation should block or warn on:

- pending `ERRATIC` review,
- old reports missing review-gate fields,
- legacy `FOLLOW` / `FOLLOWING` labels,
- too few trainable samples,
- class imbalance,
- missing metadata,
- too few tracks for track-level split.

### 5.6 Step F - Build manifest

Run:

```bash
python scripts/data/build_intent_manifest.py --dataset intent_dataset --temporal-window 15
```

The manifest is the official phase-2 dataset gate artifact. Training can be attempted during development before the manifest is perfect, but export approval should require the manifest to pass.

### 5.7 Step G - Train

Run:

```bash
python scripts/train/train_intent_cnn.py --dataset intent_dataset --temporal-window 15
```

Training requirements:

- only train on 5 trainable classes,
- exclude `UNCERTAIN`,
- block unreviewed `ERRATIC`,
- split by `track_uid`,
- keep temporal sequence padding deterministic,
- log per-class metrics,
- save class map, temporal window, threshold config, and calibration metadata.

### 5.8 Step H - Calibrate confidence

Calibration must be fitted on validation logits. The scalar temperature is saved with the checkpoint and exported artifact.

Runtime prediction flow:

```text
logits -> temperature scaling -> softmax over 5 trainable classes
      -> append runtime UNCERTAIN slot
      -> confidence + margin gate
      -> final 6-slot probability vector
```

If the prediction abstains to `UNCERTAIN`, runtime confidence should represent navigational confidence and must be `0.0`, not the probability mass assigned to the `UNCERTAIN` slot.

### 5.9 Step I - Export

Run:

```bash
python scripts/deploy/export_intent_model.py --checkpoint models/cnn_intent/intent_v1.pt
```

Export must include:

- TorchScript model,
- class names,
- class IDs,
- temporal window,
- ROI shape,
- calibration temperature,
- confidence threshold,
- margin threshold,
- training version,
- dataset manifest hash or path if available.

---

## 6. Milestones

### M0 - Phase-2 readiness check

| Task | Done Criteria |
| --- | --- |
| Confirm ontology | No active `FOLLOW` or `FOLLOWING` training labels |
| Confirm temporal API | Runtime and training reject snapshot tensors |
| Confirm logging | ROI images and sidecar metadata are written |
| Confirm review queue | `ERRATIC` and `UNCERTAIN` can be routed to review |
| Confirm docs | Phase 2 plan, guide, and design docs describe the same flow |

### M1 - Dataset collection baseline

| Task | Done Criteria |
| --- | --- |
| Collect first sessions | At least one session per intent scenario |
| Verify metadata | `metadata.jsonl` has file, track, frame, bbox, and timestamp fields |
| Check sequence length | Tracks are long enough to produce meaningful 15-frame samples |
| Check device logs | No repeated ROI extraction failures |

### M2 - Auto-label and review loop

| Task | Done Criteria |
| --- | --- |
| Run auto-label | All raw sessions converted into labeled dataset folders |
| Inspect class distribution | Report exists and shows non-empty trainable classes |
| Review risky samples | Pending `ERRATIC` count is zero before train approval |
| Preserve uncertain samples | `UNCERTAIN` remains excluded from trainable set |

### M3 - Dataset gate

| Task | Done Criteria |
| --- | --- |
| Explore report | Latest report generated after final review pass |
| Validate report | No blocking validation failures |
| Build manifest | `manifest.json` exists and records `temporal_window=15` |
| Audit split keys | At least 2 distinct `track_uid` values; target is many more |

### M4 - Baseline training

| Task | Done Criteria |
| --- | --- |
| Train model | Checkpoint created successfully |
| Validation metrics | Per-class precision/recall/F1 logged |
| Calibration | Temperature fitted and saved |
| Failure cases | Confusion matrix inspected for `CROSSING` and `ERRATIC` |

### M5 - Export and runtime smoke test

| Task | Done Criteria |
| --- | --- |
| Export artifact | TorchScript + metadata sidecar created |
| Load artifact | Runtime loader accepts artifact and class map |
| Shape test | Model accepts `(B,15,3,256,128)` only |
| Abstain test | Low confidence becomes `UNCERTAIN` with confidence `0.0` |
| Jetson smoke test | Inference loop runs without blocking camera pipeline |

### M6 - Phase-2 completion

Phase 2 can be considered complete only when:

1. Dataset manifest passes quality gates.
2. No pending `ERRATIC` review remains.
3. A trained checkpoint exists with 5 trainable classes.
4. Calibration metadata exists.
5. Exported runtime artifact loads successfully.
6. Runtime confidence and abstain behavior are verified.
7. Documentation and report are updated to match the actual model and dataset flow.

---

## 7. Quality Gates

### 7.1 Blocking gates

| Gate | Rule | Reason |
| --- | --- | --- |
| Ontology | `FOLLOW` and `FOLLOWING` count must be zero | Removed behavior must not return through training data |
| Review | Pending `ERRATIC` review must be zero | High-risk labels need human confirmation |
| Metadata | Missing sidecar metadata must be investigated | Track split and crossing labels depend on metadata |
| Temporal | Dataset must support 15-frame samples | Phase 2 is temporal, not snapshot |
| Split | At least 2 distinct tracks | Prevent invalid train/val split |
| Export metadata | Artifact must include class map and temporal window | Runtime must interpret logits correctly |

### 7.2 Warning gates

| Gate | Warning Condition | Action |
| --- | --- | --- |
| Sample count | Fewer than 500 trainable images | Continue only for smoke test, not final export |
| Class imbalance | One class dominates dataset | Collect or review underrepresented classes |
| Short tracks | Many tracks shorter than 15 frames | Collect longer clips |
| Old report | Report lacks review gate fields | Re-run `explore_roi.py` |
| Bbox depth proxy | Many labels rely on bbox fallback | Treat approach/depart labels as lower confidence |

### 7.3 Target dataset size

The minimum gate can be low for engineering smoke tests, but the target for first serious training should be:

| Class | Target Tracks | Target Images |
| --- | ---: | ---: |
| `STATIONARY` | 20+ | 1,000+ |
| `APPROACHING` | 20+ | 1,000+ |
| `DEPARTING` | 20+ | 1,000+ |
| `CROSSING` | 30+ | 1,500+ |
| `ERRATIC` | 10+ reviewed | 300+ reviewed |

`ERRATIC` is naturally rarer, so precision and review quality matter more than raw count.

---

## 8. Training Plan

### 8.1 Baseline run

The first run should establish whether the dataset is usable:

```bash
python scripts/train/train_intent_cnn.py ^
  --dataset intent_dataset ^
  --temporal-window 15 ^
  --epochs 10 ^
  --batch-size 16
```

Expected result:

- checkpoint is produced,
- validation does not crash,
- per-class metrics are logged,
- `UNCERTAIN` is absent from train labels,
- class map matches `intent_labels.py`.

### 8.2 First production candidate

After dataset gate passes:

```bash
python scripts/train/train_intent_cnn.py ^
  --dataset intent_dataset ^
  --temporal-window 15 ^
  --epochs 50 ^
  --batch-size 32
```

Tune batch size based on GPU memory. If validation for `CROSSING` or `ERRATIC` is unstable, collect more data before changing the architecture.

### 8.3 Metrics to record

| Metric | Required | Notes |
| --- | --- | --- |
| Accuracy | Yes | Useful but not sufficient |
| Per-class precision | Yes | Especially `ERRATIC` and `CROSSING` |
| Per-class recall | Yes | Avoid missing risky movement |
| F1-score | Yes | Compare class balance |
| Confusion matrix | Yes | Identify systematic class confusion |
| Abstain rate | Yes | Runtime quality indicator |
| Calibration error | Recommended | Needed before trust in confidence |
| Jetson latency | Required before deploy | Must not block perception loop |

### 8.4 Expected confusion risks

| Confusion | Likely Cause | Mitigation |
| --- | --- | --- |
| `STATIONARY` vs slow `CROSSING` | Lateral motion below threshold | Collect slow crossing examples |
| `APPROACHING` vs camera motion | Ego-motion compensation wrong or missing | Verify robot velocity metadata |
| `DEPARTING` vs bbox shrink noise | Depth unavailable | Prefer depth sensors or mark bbox-proxy samples |
| `ERRATIC` vs noisy track | Tracker jitter | Review queue and tracker quality filters |
| `UNCERTAIN` too frequent | Threshold too strict or dataset weak | Calibrate threshold after validation |

---

## 9. Runtime Integration Plan

### 9.1 Runtime prediction contract

At runtime the intent model receives per-track ROI history. If a track has fewer than 15 frames, history is left-padded with the earliest available ROI so the model still receives a fixed temporal tensor.

```text
track_id -> deque(maxlen=15) -> sequence tensor -> model -> calibrated prediction
```

### 9.2 Output contract

Each prediction should include:

| Field | Meaning |
| --- | --- |
| `track_id` | Person track |
| `intent_class` | Runtime ID 0-5 |
| `intent_name` | Human-readable class name |
| `probabilities` | 6-slot runtime vector |
| `confidence` | Navigational confidence, `0.0` on abstain |
| `direction` | Estimated motion direction `(dx, dy)` |
| `review_required` | True for `ERRATIC` or abstained uncertain cases |

### 9.3 Runtime safety policy

Phase 2 should not directly add aggressive navigation behavior. Intent output should be consumed conservatively:

- `ERRATIC`: force STOP or high caution.
- `CROSSING`: reduce speed or stop depending on distance.
- `APPROACHING`: increase caution if distance is decreasing.
- `UNCERTAIN`: treat as low-confidence and avoid relying on intent for positive motion.
- `STATIONARY` / `DEPARTING`: allow normal obstacle-aware logic, not blind acceleration.

---

## 10. Optimization Roadmap

### 10.1 Calibration

Calibration is part of Phase 2, not a later optional feature. A model with poor confidence is unsafe even if top-1 accuracy looks acceptable.

Implementation expectations:

- fit scalar temperature on validation logits,
- store temperature in checkpoint,
- store thresholds in metadata,
- test abstain path,
- set abstain confidence to `0.0`.

### 10.2 Quantization

Quantization should happen after the baseline model is correct. Dynamic quantization can be tested first for CPU paths:

```bash
python scripts/deploy/export_intent_model.py ^
  --checkpoint models/cnn_intent/intent_v1.pt ^
  --quantize-dynamic
```

For Jetson GPU deployment, TensorRT/ONNX optimization can be evaluated later, but the first phase-2 artifact can remain TorchScript if latency is acceptable.

### 10.3 Distillation

Distillation is a future acceleration and robustness path:

1. Train a larger teacher on the same temporal dataset.
2. Train the MobileNetV3 + Conv1D student with hard labels and teacher logits.
3. Compare student accuracy, abstain rate, and latency.

Distillation should not be introduced before the dataset gate is stable, because it can hide label noise rather than fix it.

### 10.4 Continual learning

Continual learning must be controlled:

- only reviewed real-log samples can enter training,
- replay buffer must preserve old classes,
- EWC can reduce catastrophic forgetting,
- each new training run must produce a new manifest/checkpoint pair,
- no model should overwrite the current runtime artifact without validation.

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Noisy auto-labels | Model learns wrong motion classes | Human review queue, validation report, conservative thresholds |
| Class imbalance | Poor recall on rare risky classes | Targeted collection for `CROSSING` and `ERRATIC` |
| Tracker ID switches | Temporal sequences mix people | Track-level audit, short-track warnings |
| Missing depth | Approach/depart labels become weak | Mark bbox proxy samples and collect depth-backed data |
| Overconfident wrong predictions | Unsafe downstream behavior | Temperature calibration and abstain gate |
| Dataset leakage | Inflated validation metrics | Split by `track_uid` |
| Latency on Jetson | Perception loop stalls | Async inference, lightweight head, quantization path |
| Legacy labels return | FOLLOW behavior reappears | Ontology gate blocks `FOLLOW` / `FOLLOWING` |

---

## 12. Detailed Checklist

### Dataset checklist

- [ ] Raw sessions collected for all five trainable classes.
- [ ] Every raw session has `metadata.jsonl`.
- [ ] Metadata includes `track_id` and `frame_id`.
- [ ] Track UID generation is stable.
- [ ] Auto-label completes without missing-file errors.
- [ ] `FOLLOW` and `FOLLOWING` do not appear in output labels.
- [ ] `UNCERTAIN` exists only as abstain/review data.
- [ ] `ERRATIC` pending review is zero before approved training.
- [ ] `review_queue/metadata.jsonl` has no duplicate rows for repeated runs.
- [ ] Manual imports write to `imported_metadata.jsonl`.

### Training checklist

- [ ] `ROIDataset` returns `(T,C,H,W)`.
- [ ] DataLoader batches are `(B,T,C,H,W)`.
- [ ] `temporal_window=15` is used consistently.
- [ ] `UNCERTAIN` is excluded from trainable labels.
- [ ] Split is by `track_uid`.
- [ ] Class weights or imbalance strategy are recorded.
- [ ] Checkpoint includes class map and temporal metadata.
- [ ] Calibration temperature is saved.

### Export checklist

- [ ] Export command succeeds.
- [ ] Metadata sidecar exists.
- [ ] Runtime class map has 6 slots with `UNCERTAIN`.
- [ ] Model artifact rejects snapshot input.
- [ ] Low-confidence output becomes `UNCERTAIN`.
- [ ] Abstain confidence is `0.0`.
- [ ] Artifact path is updated in config only after smoke test.

### Documentation checklist

- [ ] Phase 2 plan matches actual ontology.
- [ ] System design describes Temporal Intent CNN, not snapshot CNN.
- [ ] Guides explain dataset, review, train, export commands.
- [ ] Report uses one consistent name: Temporal Intent CNN.
- [ ] No active docs describe person following as a current behavior.

---

## 13. Command Reference

### Full dataset gate sequence

```bash
python scripts/data/auto_watcher.py --watch roi_dataset --output intent_dataset
python scripts/data/explore_roi.py --dataset intent_dataset --output intent_dataset
python scripts/data/validate_dataset.py intent_dataset/reports/<latest>.json
python scripts/data/build_intent_manifest.py --dataset intent_dataset --temporal-window 15
```

### Manual import sequence

```bash
python scripts/data/import_labels.py --input reviewed_labels --output intent_dataset
python scripts/data/explore_roi.py --dataset intent_dataset --output intent_dataset
python scripts/data/validate_dataset.py intent_dataset/reports/<latest>.json
```

### Training sequence

```bash
python scripts/train/train_intent_cnn.py --dataset intent_dataset --temporal-window 15
```

### Export sequence

```bash
python scripts/deploy/export_intent_model.py --checkpoint models/cnn_intent/intent_v1.pt
```

### Verification sequence

```bash
python -m ruff check src scripts tests
python -m pytest -q
```

---

## 14. Phase Exit Criteria

Phase 2 is complete when the following are true:

1. `intent_dataset/manifest.json` exists and records `temporal_window=15`.
2. Dataset validation has no blocking failures.
3. `ERRATIC` pending review count is zero.
4. The trained model uses exactly five trainable logits.
5. Runtime output exposes six slots including `UNCERTAIN`.
6. The exported artifact contains ontology and calibration metadata.
7. The Jetson runtime can load the model and run temporal inference asynchronously.
8. The system remains STOP-first for uncertain or high-risk intent.
9. Docs and report describe the same model, dataset, and deployment flow.

After these criteria pass, Phase 3 can start using intent trajectories for reward shaping and policy training.

# PLAN — Phase 5: Ablation Study & Publication

> **Project**: Context-Aware Navigation for Mecanum Robot
> **Phase**: 5 of 5
> **Timeline**: Week 13+
> **Status**: 📋 Planning
> **Dependencies**: Phase 4 hoàn thành (stable system with GAP #1-#4, benchmark data, optimized models)
> **Ref**: [system-design.md](../architecture/system-design.md) — Section 16, 17 (Phase 5)

---

## Mục tiêu

Validate all 4 algorithmic contributions (GAP #1-#4) + engineering contribution (GAP #5) through ablation study, benchmark against baselines, and produce a research paper.

> **Publication angle**: _"Temporally-Aware VLM-Guided Reward Shaping with State Alignment for Context-Aware Social Navigation"_

---

## Prerequisites (từ Phase 4)

| Requirement                        | Cần từ Phase 4                               | Status |
| ---------------------------------- | -------------------------------------------- | ------ |
| Stable RL policy (GAP #1,#2,#3,#4) | Chạy > 4 giờ liên tục, không crash           | ⬜     |
| Trajectory predictor (GAP #3)      | Trained and integrated into reward           | ⬜     |
| Benchmark suite                    | Navigation quality + system performance data | ⬜     |
| Multiple model versions            | Phase 3 (base) + Phase 4 (+ GAP #3)          | ⬜     |
| Optimized system                   | ≥ 30 FPS, < 3.5GB memory                     | ⬜     |

---

## Deliverables

| #   | Deliverable             | Mô tả                                                  |
| --- | ----------------------- | ------------------------------------------------------ |
| D1  | Ablation Study          | Per-GAP contribution analysis (4 algorithmic + 1 eng.) |
| D2  | Baseline Comparisons    | vs DWB, TEB, pure RL, rule-based                       |
| D3  | Quantitative Results    | Tables, charts, statistical significance               |
| D4  | Demonstration Video     | 3-5 min video showing system capabilities              |
| D5  | Research Paper          | Full paper (conference/journal format)                 |
| D6  | Supplementary Materials | Code docs, dataset description, reproducibility guide  |

---

## Task Breakdown

### 1. Ablation Study (D1)

> **Mục đích**: Chứng minh mỗi GAP contribution đóng góp có ý nghĩa.

- [ ] **Ablation Matrix — Aligned with System Design GAPs**:

  | ID  | Variant                           | GAP #1 (Structured VLM) | GAP #2 (Temporal Intent) | GAP #3 (Traj Pred) | GAP #4 (VLM State) | Purpose                  |
  | --- | --------------------------------- | :---------------------: | :----------------------: | :----------------: | :----------------: | ------------------------ |
  | A0  | **Full System**                   |           ✅            |            ✅            |         ✅         |         ✅         | Upper bound              |
  | A1  | No Structured VLM (opaque scalar) |    ❌ (scalar only)     |            ✅            |         ✅         |         ✅         | GAP #1 contribution      |
  | A2  | No Temporal Intent                |           ✅            |     ❌ (no R_intent)     |         ✅         |         ✅         | GAP #2 contribution      |
  | A3  | No Trajectory Prediction          |           ✅            |            ✅            |    ❌ (removed)    |         ✅         | GAP #3 contribution      |
  | A4  | No VLM State Embedding            |           ✅            |            ✅            |         ✅         |  ❌ (zero vector)  | GAP #4 contribution      |
  | A5  | No RL (Heuristic)                 |            —            |            —             |         —          |         —          | RL contribution baseline |
  | A6  | Minimal (no AI)                   |            —            |            —             |         —          |         —          | Lower bound              |

  **Key ablation comparisons**:
  - A0 vs A1: Does structured scalarization beat opaque scalar? → GAP #1 value
  - A0 vs A2: Does temporal intent reward improve anticipation? → GAP #2 value
  - A0 vs A3: Does trajectory prediction improve collision avoidance? → GAP #3 value
  - A0 vs A4: Does VLM state embedding improve navigation quality? → GAP #4 value
  - A0 vs A5: Does VLM-guided RL beat hand-crafted heuristics? → Overall contribution

- [ ] **Ablation Procedure**:
  1. Train/configure mỗi variant
  2. Chạy cùng bộ test scenarios (≥ 100 episodes per variant)
  3. Record metrics: collision rate, speed efficiency, social compliance, anticipation accuracy, smoothness
  4. Statistical tests: paired t-test hoặc Wilcoxon signed-rank (p < 0.05)

- [ ] `scripts/run_ablation.py`:
  - Auto-run all variants
  - Generate comparison tables
  - Statistical significance testing

**Metrics per variant**:

| Metric                | Unit  | How                                         |
| --------------------- | ----- | ------------------------------------------- |
| Safe Navigation Rate  | %     | Episodes without any safety violation       |
| Collision Rate        | %     | Episodes with collision                     |
| Speed Efficiency      | %     | Actual speed / max safe speed               |
| Social Compliance     | Score | VLM evaluation (unbiased separate set)      |
| Anticipation Accuracy | %     | Robot reacted BEFORE intent change (GAP #2) |
| Path Smoothness       | m/s³  | Average jerk                                |
| Collision Prediction  | AUC   | Trajectory predictor accuracy (GAP #3)      |

**Files**:

```
scripts/run_ablation.py          (NEW)
scripts/analyze_ablation.py      (NEW)
```

---

### 2. Baseline Comparisons (D2)

- [ ] **Baselines**:

  | Baseline | Method            | Mô tả                                   |
  | -------- | ----------------- | --------------------------------------- |
  | B1       | DWB Local Planner | Nav2 default Dynamic Window (no AI)     |
  | B2       | TEB Local Planner | Timed Elastic Band (no AI)              |
  | B3       | Pure RL (no VLM)  | SAC with hand-crafted reward only       |
  | B4       | Rule-based        | Heuristic policy (Phase 1)              |
  | B5       | **Ours (Full)**   | VLM-guided context-aware RL + GAP #1-#4 |

- [ ] **Test Scenarios** (standardized, same for all):

  | #   | Scenario               | Duration | Persons  | Difficulty |
  | --- | ---------------------- | -------- | -------- | ---------- |
  | S1  | Empty corridor         | 60s      | 0        | Easy       |
  | S2  | Single person crossing | 60s      | 1        | Easy       |
  | S3  | Two-way traffic        | 120s     | 2-3      | Medium     |
  | S4  | Narrow passage         | 60s      | 1-2      | Medium     |
  | S5  | Crowd navigation       | 120s     | 4-6      | Hard       |
  | S6  | Ambiguous trajectory   | 120s     | 1        | Medium     |
  | S7  | Erratic person         | 60s      | 1        | Hard       |
  | S8  | Mixed scenario         | 180s     | 3-5      | Hard       |
  - Mỗi scenario chạy ≥ 10 lần per baseline
  - Total: 8 scenarios × 5 baselines × 10 runs = **400 runs**

- [ ] **GAP #5 validation** (real-world deployment):
  - Run on Jetson Orin Nano 8GB + consumer GPU server
  - Document: inference latency, memory usage, deployment cycle time
  - Compare: real hardware vs simulation-only results (if available)

**Files**:

```
scripts/run_baselines.py         (NEW)
scripts/compare_baselines.py     (NEW)
```

---

### 3. Quantitative Results (D3)

- [ ] **Result Tables**:

  #### Table 1: Main Results

  | Method     | SNR ↑ | CR ↓ | SE ↑ | SC ↑ | AA ↑ | PS ↑ |
  | ---------- | ----- | ---- | ---- | ---- | ---- | ---- |
  | DWB        | -     | -    | -    | -    | -    | -    |
  | TEB        | -     | -    | -    | -    | -    | -    |
  | Pure RL    | -     | -    | -    | -    | -    | -    |
  | Rule-based | -     | -    | -    | -    | -    | -    |
  | **Ours**   | -     | -    | -    | -    | -    | -    |

  _(SNR: Safe Nav Rate, CR: Collision Rate, SE: Speed Efficiency, SC: Social Compliance, AA: Anticipation Accuracy, PS: Path Smoothness)_

  #### Table 2: Ablation Results

  | Variant                | SNR | CR  | SE  | SC  | AA  | PS  | Δ vs Full |
  | ---------------------- | --- | --- | --- | --- | --- | --- | --------- |
  | A0: Full System        | -   | -   | -   | -   | -   | -   | —         |
  | A1: No Structured VLM  | -   | -   | -   | -   | -   | -   | -         |
  | A2: No Temporal Intent | -   | -   | -   | -   | -   | -   | -         |
  | A3: No Traj Prediction | -   | -   | -   | -   | -   | -   | -         |
  | A4: No VLM State       | -   | -   | -   | -   | -   | -   | -         |
  | A5: No RL (Heuristic)  | -   | -   | -   | -   | -   | -   | -         |

  #### Table 3: System Performance (GAP #5)

  | Metric        | Value | Hardware                    |
  | ------------- | ----- | --------------------------- |
  | Pipeline FPS  | -     | Jetson Orin Nano 8GB        |
  | E2E Latency   | -     | Camera → ZMQ                |
  | Memory Usage  | -     | GPU + CPU combined          |
  | YOLO Latency  | -     | Per-frame                   |
  | CNN Latency   | -     | Per-person                  |
  | RL Latency    | -     | Per-decision                |
  | VLM Eval Time | -     | Per-evaluation on Server    |
  | Deploy Cycle  | -     | Code push → running on edge |

- [ ] **Charts & Visualizations**:
  - Bar chart: SNR comparison across methods
  - Radar chart: multi-metric comparison (per GAP)
  - Box plot: reward distribution per variant
  - Line chart: RL training curve
  - Ablation bar chart: Δ per GAP contribution

- [ ] `scripts/generate_plots.py`

**Files**:

```
scripts/generate_plots.py        (NEW)
results/                         (NEW dir)
results/ablation/                (NEW dir)
results/baselines/               (NEW dir)
results/figures/                 (NEW dir)
```

---

### 4. Demonstration Video (D4)

- [ ] **Video Storyboard** (3-5 minutes):

  | Time      | Scene               | Content                                |
  | --------- | ------------------- | -------------------------------------- |
  | 0:00-0:30 | Intro               | Problem statement, thesis              |
  | 0:30-1:00 | Architecture        | Edge-cloud diagram, data flow (GAP #5) |
  | 1:00-1:30 | Empty Corridor      | CRUISE mode, high speed                |
  | 1:30-2:00 | Person Crossing     | AVOID + temporal anticipation (GAP #2) |
  | 2:00-2:30 | Crowd Navigation    | Trajectory prediction effect (GAP #3)  |
  | 2:30-3:00 | Safety Demo         | STOP, erratic detection                |
  | 3:00-3:30 | VLM Evaluation      | Structured scoring overlay (GAP #1)    |
  | 3:30-4:00 | VLM State Embedding | Staleness decay visualization (GAP #4) |
  | 4:00-4:30 | Results             | Key metrics, comparison table          |
  | 4:30-5:00 | Conclusion          | Contributions, future work             |

**Files**:

```
scripts/video_overlay.py         (NEW)
results/videos/                  (NEW dir)
```

---

### 5. Research Paper (D5)

- [ ] **Paper Structure** (IEEE/Conference format):

  #### Abstract (~200 words)
  - Problem: VLM-guided social navigation lacks structured reward, temporal awareness, and real-world validation
  - Approach: Structured scalarization + temporal intent reward + VLM-RL state alignment
  - Key result: X% improvement over baselines

  #### I. Introduction
  - Motivation: robots in human environments
  - Gap: existing VLM-guided RL collapses reasoning into opaque scalar
  - **4 contributions (aligned with GAPs)**:
    1. Structured VLM reward scalarization (GAP #1)
    2. Temporal intent reward shaping (GAP #2)
    3. Intent-conditioned trajectory prediction (GAP #3)
    4. VLM-RL state alignment with staleness decay (GAP #4)
    - (+) Real-world validation on edge-cloud hardware (GAP #5)

  #### II. Related Work

  | Area              | Key Papers           | Our Difference                               |
  | ----------------- | -------------------- | -------------------------------------------- |
  | Social Navigation | SARL, CrowdNav       | Structured VLM reward vs hand-crafted        |
  | RL for Navigation | DRL-VO, NavRL        | Temporal intent reward + VLM state alignment |
  | VLM for Robotics  | RT-2, PaLM-E         | Offline reward shaping with staleness decay  |
  | Intent Prediction | Social Force, STGCNN | Lightweight CNN + trajectory conditioning    |

  #### III. System Architecture
  - Edge-cloud architecture (Jetson + Server)
  - Perception pipeline (YOLO + temporal CNN)
  - VLM-guided reward shaping framework

  #### IV. Methodology
  - A. Structured VLM Reward Scalarization (GAP #1)
  - B. Temporal Intent Reward R_intent (GAP #2)
  - C. Intent-Conditioned Trajectory Prediction (GAP #3)
  - D. VLM-RL State Alignment with Staleness Decay (GAP #4)
  - E. SAC Training with Composite Scalar Reward

  #### V. Experiments
  - A. Setup (hardware: Jetson Orin Nano 8GB, Server with GPU)
  - B. Baselines (DWB, TEB, pure RL, heuristic)
  - C. Main Results (Table 1)
  - D. Ablation Study (Table 2) — per-GAP analysis
  - E. System Performance (Table 3) — GAP #5 validation
  - F. Qualitative Analysis

  #### VI. Discussion
  - Key findings per GAP
  - Limitations (VLM latency, staleness, trajectory predictor accuracy)
  - Future work (game-theoretic modeling, multi-robot)

  #### VII. Conclusion

- [ ] **Key Claims to Support**:

  | Claim                                           | Evidence Needed   | GAP |
  | ----------------------------------------------- | ----------------- | --- |
  | Structured scalarization > opaque scalar        | Ablation A0 vs A1 | #1  |
  | Temporal intent improves anticipation           | Ablation A0 vs A2 | #2  |
  | Trajectory prediction reduces collisions        | Ablation A0 vs A3 | #3  |
  | VLM state embedding improves navigation quality | Ablation A0 vs A4 | #4  |
  | System works on real edge hardware              | Performance table | #5  |

- [ ] **Writing Schedule**:
      | Week | Section | Status |
      |------|---------|--------|
      | W13 | Related Work + Methodology | ⬜ |
      | W14 | Experiments + Results | ⬜ |
      | W15 | Introduction + Discussion + Conclusion | ⬜ |
      | W16 | Revision + polish + submission | ⬜ |

**Files**:

```
docs/paper/                      (NEW dir)
docs/paper/main.tex              (NEW)
docs/paper/figures/              (NEW dir)
docs/paper/references.bib        (NEW)
```

---

### 6. Supplementary Materials (D6)

- [ ] README.md update: full setup guide, architecture + GAP overview
- [ ] Dataset description: collection protocol, labeling, statistics
- [ ] Reproducibility guide: hardware, software versions, training steps

**Files**:

```
README.md                        (UPDATE)
docs/dataset.md                  (NEW)
docs/reproducibility.md          (NEW)
```

---

## Verification Checklist

### Research Quality

| #   | Check                    | Criteria                               | Status |
| --- | ------------------------ | -------------------------------------- | ------ |
| V1  | Ablation complete        | All 7 variants tested, ≥ 10 runs each  | ⬜     |
| V2  | Baselines complete       | All 5 baselines tested, ≥ 10 runs each | ⬜     |
| V3  | Statistical significance | p < 0.05 for per-GAP claims            | ⬜     |
| V4  | Tables complete          | 3 main tables filled                   | ⬜     |
| V5  | Charts generated         | ≥ 5 publication-quality figures        | ⬜     |
| V6  | Demo video               | 3-5 min, all GAPs demonstrated         | ⬜     |
| V7  | Paper draft              | All sections written                   | ⬜     |
| V8  | Paper reviewed           | Internal review passed                 | ⬜     |

### Publication Readiness

| Item                | Target                    | Status |
| ------------------- | ------------------------- | ------ |
| Paper length        | 6-8 pages (conference)    | ⬜     |
| Figures             | ≥ 5 high-quality          | ⬜     |
| References          | ≥ 20 relevant papers      | ⬜     |
| Code release        | Public repo (cleaned)     | ⬜     |
| Supplementary video | Uploaded to YouTube/cloud | ⬜     |

---

## Target Venues

| Venue | Type       | Deadline (Typical) | Fit                                |
| ----- | ---------- | ------------------ | ---------------------------------- |
| IROS  | Conference | March              | ⭐⭐⭐ Robot navigation + learning |
| ICRA  | Conference | September          | ⭐⭐⭐ Robot + AI                  |
| RA-L  | Journal    | Rolling            | ⭐⭐⭐ Robotics + Automation       |
| CoRL  | Conference | June               | ⭐⭐ Robot learning                |

---

## Risk Register (Phase 5)

| Risk                                     | Probability | Impact   | Mitigation                                             |
| ---------------------------------------- | ----------- | -------- | ------------------------------------------------------ |
| Results not statistically significant    | Medium      | Critical | More runs (30+), different environments                |
| GAP contributions show small improvement | Medium      | High     | Emphasize qualitative analysis + real-world validation |
| Paper rejected                           | Medium      | Medium   | Address reviewer concerns, resubmit to backup venue    |
| Not enough real-world test scenarios     | Medium      | High     | Supplement with recorded scenarios                     |
| Reproducibility issues                   | Low         | Medium   | Document everything, Docker container for training     |

---

## Final Checklist — Project Completion

| #   | Milestone                                          | Status |
| --- | -------------------------------------------------- | ------ |
| 1   | System runs stable ≥ 4 hours                       | ⬜     |
| 2   | RL policy outperforms heuristic                    | ⬜     |
| 3   | GAP #1: Structured scalarization proven (ablation) | ⬜     |
| 4   | GAP #2: Temporal intent reward proven (ablation)   | ⬜     |
| 5   | GAP #3: Trajectory prediction proven (ablation)    | ⬜     |
| 6   | GAP #4: VLM state alignment proven (ablation)      | ⬜     |
| 7   | GAP #5: Real-world deployment validated            | ⬜     |
| 8   | Benchmark vs ≥ 3 baselines completed               | ⬜     |
| 9   | Demo video produced                                | ⬜     |
| 10  | Paper written and internally reviewed              | ⬜     |
| 11  | Code documented and cleaned                        | ⬜     |
| 12  | Paper submitted to target venue                    | ⬜     |

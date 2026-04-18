# PLAN — Phase 4: Intent-Aware Interaction + Optimization (GAP #3)

> **Project**: Context-Aware Navigation for Mecanum Robot
> **Phase**: 4 of 5
> **Timeline**: Week 9–12
> **Status**: 📋 Planning
> **Dependencies**: Phase 3 hoàn thành (RL policy deployed with GAP #1/#2/#4, VLM reward stable, training loop running)
> **Ref**: [system-design.md](../architecture/system-design.md) — Section 15, 16 (GAP #3), 17
> **GAP addressed**: #3 (intent-conditioned trajectory prediction)

---

## Mục tiêu

Implement **GAP #3** (intent-conditioned trajectory prediction) + tối ưu toàn bộ hệ thống: train lightweight MLP để predict human trajectory given robot action, integrate collision probability vào reward, benchmark performance, và đạt trạng thái **production-ready** cho demo/deployment.

> **Scope decision (from system-design.md)**: We do NOT attempt game-theoretic modeling, Nash equilibrium, or full closed-loop interactive RL. GAP #3 is scoped to a tractable, data-driven extension using a lightweight MLP.

---

## Prerequisites (từ Phase 3)

| Requirement                   | Cần từ Phase 3                              | Status |
| ----------------------------- | ------------------------------------------- | ------ |
| RL policy v1+ deployed        | Navigates tốt hơn heuristic                 | ⬜     |
| VLM structured reward stable  | Consistent 3-axis scores, low variance      | ⬜     |
| Temporal intent reward active | R*intent = Σ γ^i · risk(intent*{t+i}) works | ⬜     |
| VLM state alignment working   | Staleness decay producing valid embeddings  | ⬜     |
| Training loop                 | End-to-end cycle hoạt động                  | ⬜     |
| ≥ 50K experience frames       | Rich data cho trajectory prediction         | ⬜     |
| **Tracked human positions**   | robot_action + human_position logged        | ⬜     |

---

## Deliverables

| #   | Deliverable                       | Mô tả                                       | Bottleneck | GAP |
| --- | --------------------------------- | ------------------------------------------- | ---------- | --- |
| D1  | Intent-Conditioned Trajectory MLP | Predict human trajectory given robot action | B3 ⚠️      | #3  |
| D2  | Collision Probability → R_safety  | Predicted collision_prob feeds into reward  | —          | #3  |
| D3  | CNN Backbone Upgrade (if needed)  | EfficientNet hoặc improved MobileNetV3      | A1         | —   |
| D4  | VLM Prompt Optimization           | Better prompts, higher consistency          | A3         | —   |
| D5  | RL Policy Refinement              | Iterative SAC with GAP #3 integrated        | A2         | —   |
| D6  | Performance Optimization          | Latency, memory, throughput tuning          | —          | —   |
| D7  | System Benchmarks                 | Comprehensive benchmark suite               | —          | —   |

---

## Task Breakdown

### 1. Intent-Conditioned Trajectory Prediction (D1) — GAP #3

> **This is the core algorithmic contribution of Phase 4.**
> Predicts how a person's trajectory changes given the robot's current action.
> Lightweight, data-driven alternative to full game-theoretic modeling.

- [ ] `src/navigation/trajectory_predictor.py` (NEW):

  #### 1.1 Architecture — Lightweight MLP (3 layers)

  ```python
  class IntentConditionedPredictor(nn.Module):
      """Predict human trajectory conditioned on robot action.

      Input:
        - person_intent[t:t-5]   (5 × 6 = 30) — intent history (one-hot)
        - robot_action[t]        (7)           — current action
        - relative_position      (4)           — dx, dy, d_dist, d_angle

      Output:
        - predicted_trajectory[t+1:t+5]  (5 × 2 = 10) — future positions (dx, dy)
        - collision_probability           (1)           — scalar [0, 1]
      """
      def __init__(self):
          self.mlp = nn.Sequential(
              nn.Linear(41, 128),    # 30 + 7 + 4 = 41 input
              nn.ReLU(),
              nn.Linear(128, 64),
              nn.ReLU(),
              nn.Linear(64, 11),     # 10 (trajectory) + 1 (collision_prob)
          )
          self.collision_head = nn.Sigmoid()

      def forward(self, intent_history, robot_action, relative_pos):
          x = torch.cat([intent_history, robot_action, relative_pos], dim=-1)
          out = self.mlp(x)
          trajectory = out[..., :10].reshape(-1, 5, 2)
          collision_prob = self.collision_head(out[..., 10:11])
          return trajectory, collision_prob
  ```

  #### 1.2 Training Data
  - Source: Phase 1-3 deployment data
  - Labels: actual human positions at t+1..t+5 (from ByteTrack)
  - Collision label: did human come within 0.3m of robot within 5 frames?
  - Dataset size estimate: ≥ 50K (track_id, intent_history, robot_action, human_trajectory) tuples

  #### 1.3 Training
  - Loss: MSE (trajectory) + BCE (collision)
  - Epochs: 100
  - Validation: 20% holdout
  - Export: ONNX → TensorRT on Jetson (inference < 1ms)

- [ ] `training/train_trajectory_predictor.py` (NEW):
  - Data loader from Phase 1-3 experience logs
  - Training script with metrics logging

**Files**:

```
src/navigation/trajectory_predictor.py        (NEW)
training/train_trajectory_predictor.py        (NEW)
training/configs/trajectory_config.yaml       (NEW)
```

---

### 2. Collision Probability → Reward Integration (D2) — GAP #3

- [ ] Update `training/reward_shaping.py`:

  ```python
  def compute_safety_reward(transition, collision_predictor=None) -> float:
      """R_safety with optional predicted collision probability (GAP #3)."""
      r_safety = 0.0

      # Hard-coded constraints (unchanged from Phase 3)
      if transition.collision:
          r_safety -= 10.0
      if transition.min_person_distance < SAFETY_DISTANCE:
          r_safety -= 2.0 * (SAFETY_DISTANCE - transition.min_person_distance)
      if transition.mode == STOP and transition.danger_detected:
          r_safety += 1.0

      # NEW: Predicted collision penalty (GAP #3)
      if collision_predictor is not None:
          pred_collision_prob = collision_predictor.predict(
              transition.intent_history,
              transition.action_taken,
              transition.relative_positions,
          )
          r_safety -= 1.5 * pred_collision_prob  # penalty scales with probability

      return r_safety
  ```

**Files**:

```
training/reward_shaping.py   (MODIFY)
```

---

### 3. CNN Backbone Evaluation & Upgrade (D3)

- [ ] **Decision gate**: Upgrade hay không?

  | Condition                  | Action                                |
  | -------------------------- | ------------------------------------- |
  | Accuracy ≥ 80% all classes | ✅ KEEP MobileNetV3, chỉ fine-tune    |
  | Accuracy 70-80%            | ⚠️ Thêm data + augmentation, re-train |
  | Accuracy < 70%             | 🔴 Upgrade backbone                   |

- [ ] **Upgrade candidates** (nếu cần):

  | Backbone                    | Params | Latency (Jetson) | Expected Accuracy |
  | --------------------------- | ------ | ---------------- | ----------------- |
  | MobileNetV3-Small (current) | 2.5M   | ~2ms             | 75-80%            |
  | EfficientNet-Lite0          | 4.7M   | ~3ms             | 79-84%            |

**Files**:

```
training/benchmark_cnn.py          (NEW)
training/train_cnn.py              (MODIFY if upgrade)
```

---

### 4. VLM Prompt Optimization (D4)

- [ ] Prompt A/B Testing (3-5 variations, 100 test frames)
- [ ] Few-Shot Prompting (3-5 examples per prompt)
- [ ] Temperature tuning (0.1, 0.2, 0.3, 0.5)
- [ ] Document best prompt → update `training/configs/vlm_config.yaml`

**Files**:

```
training/prompt_optimizer.py       (NEW)
training/configs/vlm_config.yaml   (MODIFY)
```

---

### 5. RL Policy Refinement (D5)

- [ ] **Hyperparameter Sweep**:

  | Parameter      | Search Range                        | Method      |
  | -------------- | ----------------------------------- | ----------- |
  | Learning Rate  | [1e-4, 3e-4, 1e-3]                  | Grid search |
  | Batch Size     | [128, 256, 512]                     | Grid search |
  | Reward weights | w_vlm [0.3-0.5], w_intent [0.1-0.3] | Manual tune |

- [ ] **Iterative Training with GAP #3**:

  ```
  Repeat 3-5 times:
    1. Deploy current policy
    2. Collect 10K frames
    3. Train trajectory predictor (GAP #3) on new data
    4. VLM evaluate with structured output (GAP #1)
    5. Compute R_t with all components (GAP #1,#2,#3,#4)
    6. Continue SAC training
    7. Deploy improved policy
  ```

**Files**:

```
training/hyperparameter_sweep.py   (NEW)
training/train_rl.py               (MODIFY)
training/configs/rl_config.yaml    (MODIFY)
```

---

### 6. Performance Optimization & Benchmarks (D6, D7)

- [ ] **Jetson Performance Tuning**: TensorRT batching, CUDA streams, memory pools
- [ ] **Pipeline Profiling**: per-stage latency, memory monitoring
- [ ] **Navigation Quality Benchmarks**:

  | Scenario                      | Metric                   | Target       |
  | ----------------------------- | ------------------------ | ------------ |
  | Empty corridor                | Speed efficiency         | ≥ 90% of max |
  | Single person crossing        | Stop/avoid success rate  | ≥ 95%        |
  | Crowded (3+ persons)          | Safe navigation rate     | ≥ 90%        |
  | **With trajectory predictor** | Collision avoidance rate | **≥ 97%**    |

- [ ] **Comparison Table**:

  | Method                | Safe Nav Rate | Collision | Speed Eff | Latency  |
  | --------------------- | ------------- | --------- | --------- | -------- |
  | Heuristic (Phase 1)   | baseline      | baseline  | baseline  | baseline |
  | RL v1 (Phase 3)       | -             | -         | -         | -        |
  | RL + GAP #3 (Phase 4) | -             | -         | -         | -        |

**Files**:

```
scripts/benchmark_suite.py   (NEW)
scripts/benchmark.py         (MODIFY)
```

---

### 7. Integration & Stress Testing

- [ ] Full pipeline: camera → YOLO → CNN → context → **trajectory predictor** → RL → safety → ZMQ
- [ ] Run 1000 frames continuous, verify no memory leaks
- [ ] Edge case tests: sudden person appearance, erratic trajectories, etc.

**Files**:

```
tests/test_integration.py     (MODIFY)
tests/test_stress.py           (NEW)
tests/test_trajectory_pred.py  (NEW)
```

---

## Verification Checklist

### GAP #3 Specific

| #   | Test                       | Criteria                                     | Status |
| --- | -------------------------- | -------------------------------------------- | ------ |
| G1  | Trajectory prediction      | MSE < threshold on validation set            | ⬜     |
| G2  | Collision probability      | AUC-ROC > 0.80 on collision detection        | ⬜     |
| G3  | Reward integration         | R_safety uses predicted collision_prob       | ⬜     |
| G4  | Inference latency          | Trajectory predictor < 1ms on Jetson         | ⬜     |
| G5  | RL improvement with GAP #3 | Collision avoidance rate improves vs Phase 3 | ⬜     |

### Navigation Quality

| #   | Scenario               | Criteria                           | Status |
| --- | ---------------------- | ---------------------------------- | ------ |
| V1  | Empty corridor         | CRUISE mode, ≥ 90% max speed       | ⬜     |
| V2  | Single crossing person | AVOID/STOP, 0 collisions           | ⬜     |
| V3  | Crowd (3+ persons)     | Safe navigation, appropriate speed | ⬜     |
| V4  | Erratic person         | STOP immediately                   | ⬜     |

### System Stability

| #   | Test             | Duration   | Criteria                | Status |
| --- | ---------------- | ---------- | ----------------------- | ------ |
| S1  | Continuous run   | 1 hour     | No crash                | ⬜     |
| S2  | Memory stability | 4 hours    | No OOM                  | ⬜     |
| S3  | Hot-reload       | 10 deploys | No inference gap > 33ms | ⬜     |

---

## Phase 4 → Phase 5 Handoff

Khi HOÀN THÀNH Phase 4, Phase 5 cần:

1. ✅ RL policy with GAP #1, #2, #3, #4 deployed → reliable data cho ablation
2. ✅ Trajectory predictor trained → ablation: with/without GAP #3
3. ✅ Benchmark results → baseline comparisons cho paper
4. ✅ System runs 4+ hours → reliable demo recording
5. ✅ Multiple RL versions → compare in paper (Phase 3 base vs Phase 4 + GAP #3)

---

## Risk Register (Phase 4)

| Risk                                      | Probability | Impact | Mitigation                                         |
| ----------------------------------------- | ----------- | ------ | -------------------------------------------------- |
| Trajectory predictor insufficient data    | Medium      | High   | Collect more with RL policy in varied environments |
| Collision prediction false positives      | Medium      | Medium | Conservative threshold. Don't dominate reward.     |
| CNN upgrade exceeds VRAM budget           | Low         | High   | Benchmark trước. Keep MobileNetV3 as fallback      |
| RL overfits to training environment       | Medium      | High   | Test in new environments. Curriculum learning      |
| Performance regression after optimization | Low         | Medium | Benchmark before/after every change                |

---

## What We Do NOT Attempt (Phase 4)

> As defined in system-design.md GAP #3 scope decision:

- ✗ Game-theoretic dual-policy optimization
- ✗ Nash equilibrium computation
- ✗ Full closed-loop interactive RL
- ✗ Interaction graph networks

These are separate research contributions that would require:

- Extensive interaction datasets
- Convergence guarantees for dual-policy optimization
- Significantly more compute and training time

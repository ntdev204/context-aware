# PLAN — Phase 3: VLM-Guided RL + Core Contributions (GAP #1, #2, #4)

> **Project**: Context-Aware Navigation for Mecanum Robot
> **Phase**: 3 of 5
> **Timeline**: Week 6–8
> **Status**: 📋 Planning
> **Dependencies**: Phase 2 hoàn thành (temporal state, CNN trained, intent trajectory buffer, gRPC, experience streaming)
> **Ref**: [system-design.md](../architecture/system-design.md) — Section 6, 8, 16, 17
> **GAPs addressed**: #1 (structured scalarization), #2 (temporal intent reward), #4 (VLM-RL state alignment)

---

## Mục tiêu

Tích hợp **3 core algorithmic contributions**:

1. **GAP #1**: Structured VLM reward scalarization (safety/social/efficiency → scalar R_vlm)
2. **GAP #2**: Temporal intent reward R*intent = Σ γ^i · risk(intent*{t+i}), k=5, γ=0.9
3. **GAP #4**: VLM-RL state alignment (VLMStateAligner + staleness decay)

Train SAC agent với composite scalar reward + deploy RL policy đầu tiên lên Jetson thay thế heuristic policy.

---

## Prerequisites (từ Phase 2)

| Requirement                  | Cần từ Phase 2                           | Status |
| ---------------------------- | ---------------------------------------- | ------ |
| Temporal state stacking      | k=3-5 observation hoạt động              | ⬜     |
| CNN Intent trained           | Accuracy ≥ 70% trên test set             | ⬜     |
| **Intent trajectory buffer** | Per-track intent history logging         | ⬜     |
| gRPC server (Jetson)         | Model deployment + hot-reload            | ⬜     |
| Experience streaming         | Jetson → Server pipeline                 | ⬜     |
| Collected experiences        | ≥ 10,000 frames với detections + actions | ⬜     |

---

## Deliverables

| #   | Deliverable               | Mô tả                                             | Bottleneck | GAP   |
| --- | ------------------------- | ------------------------------------------------- | ---------- | ----- |
| D1  | VLM Structured Evaluator  | Gemma 4 outputs {safety, social, efficiency}      | B2 ⚠️      | #1    |
| D2  | Reward Scalarization      | Structured VLM → scalar R_vlm ∈ ℝ                 | B2 ⚠️      | #1    |
| D3  | Temporal Intent Reward    | R*intent = Σ γ^i · risk(intent*{t+i}), k=5        | —          | #2    |
| D4  | VLM State Aligner         | 64-dim embedding + staleness decay                | —          | #4    |
| D5  | Composite Reward Function | R_t = w_vlm·R_vlm + w_safety·R_safety + ...       | —          | #1,#2 |
| D6  | SAC Training Pipeline     | Stable Baselines3 SAC với composite scalar reward | —          | —     |
| D7  | RL Policy (Jetson)        | TensorRT RL policy thay heuristic                 | —          | —     |
| D8  | Model Deployment          | gRPC deploy + hot-reload workflow                 | —          | —     |
| D9  | Training Loop             | End-to-end: collect → evaluate → train → deploy   | —          | —     |

---

## Task Breakdown

### 1. VLM Structured Evaluator — Gemma 4 (D1) — GAP #1

- [ ] `training/vlm_evaluator.py`:
  - Connect to Ollama (local Server, port 11434)
  - Model: `gemma4:e4b` (9.6GB) hoặc `e2b` (7.2GB) fallback

  - **Structured JSON output** (thay vì scalar score):

    ```python
    @dataclass
    class VLMRewardSignal:
        """Structured intermediate signal from VLM (GAP #1).

        VLM evaluates on 3 axes → combined into scalar R_vlm.
        This is standard scalarization, NOT MORL.
        """
        safety_score: float          # [-1, 1]
        social_score: float          # [-1, 1]
        efficiency_score: float      # [-1, 1]
        reasoning: str
        reasoning_embedding: np.ndarray  # (64,) — for GAP #4
        confidence: float            # [0, 1]
    ```

  - **Prompt template** (structured JSON output):

    ```
    You are evaluating a robot's navigation decision.

    [IMAGE: camera_frame]

    Context:
    - Detected: {num_persons} persons, {num_obstacles} obstacles
    - Nearest person: {distance}m, intent: {intent_name}
    - Intent trajectory (last 5 frames): {intent_sequence}
    - Free space: {ratio}%

    Robot action:
    - Mode: {mode_name}, Speed: {velocity}%, Direction: {heading}deg

    Evaluate this decision on THREE independent axes.
    Respond with JSON:
    {
      "safety_score": float,       // [-1, 1]
      "social_score": float,       // [-1, 1]
      "efficiency_score": float,   // [-1, 1]
      "reasoning": "string",
      "confidence": float          // [0, 1]
    }
    ```

  - Output parsing: extract JSON, validate ranges
  - Error handling: timeout, parse failure → return neutral signal (all 0.0, confidence=0.0)
  - Throughput target: ~500-1000 evaluations/hour

**Files**:

```
training/vlm_evaluator.py          (NEW)
training/configs/vlm_config.yaml   (NEW)
```

---

### 2. Reward Scalarization + Stabilization (D2, D5) — GAP #1

> **RL Theory**: Final reward must be scalar R_t ∈ ℝ (standard MDP).
> VLM sub-scores are intermediate — combined via fixed-weight scalarization.
> This is NOT multi-objective RL / MORL (no Pareto front).

- [ ] `training/reward_shaping.py`:

  #### 2.1 VLM Score Scalarization (GAP #1)

  ```python
  W_VLM_SAFETY = 0.4
  W_VLM_SOCIAL = 0.3
  W_VLM_EFFICIENCY = 0.3

  def scalarize_vlm(signal: VLMRewardSignal) -> float:
      """Combine structured VLM output into scalar R_vlm ∈ ℝ."""
      r = (W_VLM_SAFETY * signal.safety_score
           + W_VLM_SOCIAL * signal.social_score
           + W_VLM_EFFICIENCY * signal.efficiency_score)
      return r * signal.confidence  # down-weight uncertain evaluations
  ```

  #### 2.2 Multi-Query Stabilization
  - Gửi cùng 1 (frame, context, action) tới VLM **3 lần**
  - Average each sub-score independently
  - Outlier rejection: > 2σ per axis

  #### 2.3 Composite Scalar Reward

  ```python
  W_VLM = 0.35
  W_SAFETY = 0.25
  W_INTENT = 0.20
  W_EFF = 0.20

  def compute_reward(transition, vlm_signal, intent_trajectory) -> float:
      """R_t ∈ ℝ — standard scalar reward for SAC."""
      r_vlm = scalarize_vlm(vlm_signal)
      r_safety = compute_safety_reward(transition)       # hard-coded
      r_intent = compute_temporal_intent_reward(          # GAP #2
          intent_trajectory, transition.action_taken
      )
      r_eff = compute_efficiency_reward(transition)       # heuristic

      R_t = W_VLM * r_vlm + W_SAFETY * r_safety + W_INTENT * r_intent + W_EFF * r_eff
      return R_t
  ```

  #### 2.4 Reward Statistics Tracking
  - Log: R_t distribution, per-component breakdown (R_vlm, R_safety, R_intent, R_eff)
  - Alert nếu VLM scores biased (>70% cùng 1 direction)
  - Periodically spot-check: human review 10% of VLM evaluations

**Files**:

```
training/reward_shaping.py    (NEW)
```

---

### 3. Temporal Intent Reward (D3) — GAP #2

> Uses intent trajectory data collected in Phase 2 (D4).

- [ ] `training/temporal_intent_reward.py`:

  #### 3.1 Formal Definition

  ```
  R_intent(t) = Σ_{i=0}^{k-1} γ^i · risk(intent_{t+i})

  Parameters (fixed, not learned):
    k = 5 frames      (~167ms at 30 FPS)
    γ = 0.9            (temporal discount)
    risk(·) ∈ [-1, 1]  (per-class mapping, see below)
  ```

  #### 3.2 Intent Risk Mapping

  | Intent      | risk(·) | Rationale                          |
  | ----------- | ------- | ---------------------------------- |
  | STATIONARY  | 0.0     | No risk                            |
  | APPROACHING | -0.3    | Moderate — robot should slow       |
  | DEPARTING   | +0.2    | Positive — person moving away      |
  | CROSSING    | -0.6    | High risk — robot should stop/wait |
  | FOLLOWING   | 0.0     | Neutral                            |
  | ERRATIC     | -0.8    | Highest risk — unpredictable       |

  #### 3.3 Implementation

  ```python
  INTENT_RISK = {0: 0.0, 1: -0.3, 2: 0.2, 3: -0.6, 4: 0.0, 5: -0.8}
  TEMPORAL_GAMMA = 0.9
  LOOKAHEAD_K = 5

  def compute_temporal_intent_reward(
      intent_trajectory: List[IntentPrediction],
      action_taken: ActionSpace,
  ) -> float:
      """Discounted temporal intent risk (GAP #2).

      Non-differentiable — shaped reward signal, not a loss function.
      SAC treats it as any other scalar reward component.
      """
      # ... (see system-design.md Section 6 for full implementation)
  ```

  #### 3.4 Properties
  - Non-differentiable (reward shaping, not backpropagated)
  - Penalizes late reaction to CROSSING/ERRATIC
  - Rewards proactive deceleration BEFORE high-risk intents occur

**Files**:

```
training/temporal_intent_reward.py   (NEW)
```

---

### 4. VLM State Aligner (D4) — GAP #4

> Bridges semantic gap between VLM (sees images) and RL (operates on float vectors).

- [ ] `training/vlm_state_aligner.py`:

  #### 4.1 Architecture

  ```python
  class VLMStateAligner(nn.Module):
      STALENESS_HALF_LIFE_S = 2.0  # w drops to 0.5 after 2s

      def __init__(self, vlm_hidden_dim=2048, projection_dim=64):
          self.projector = nn.Sequential(
              nn.Linear(vlm_hidden_dim, 256),
              nn.ReLU(),
              nn.Linear(256, projection_dim),
              nn.LayerNorm(projection_dim),
          )
          self._cache = np.zeros(projection_dim)
          self._last_update_time = 0.0

      def encode(self, image) -> np.ndarray:
          hidden = self.vlm.encode_image(image)
          projected = self.projector(hidden[:, 0, :])
          self._cache = projected.detach().numpy()
          self._last_update_time = time.monotonic()
          return self._cache

      @property
      def staleness_weight(self) -> float:
          """w = 2^(-age / half_life). At 2s: w=0.5, at 10s: w≈0.03"""
          age = time.monotonic() - self._last_update_time
          return 2.0 ** (-age / self.STALENESS_HALF_LIFE_S)
  ```

  #### 4.2 Integration with RL State

  ```python
  # Updated observation space (from ~102 → ~167 floats)
  state = np.concatenate([
      scene_features,              # [6]
      occupancy_grid,              # [64]
      person_intents,              # [24]
      robot_state,                 # [8]
      vlm_embedding * w,           # [64] weighted by staleness
      [w],                         # [1]  staleness weight itself
  ])                               # Total: ~167 floats
  ```

  #### 4.3 Design Decisions

  | Decision         | Choice                           | Rationale                              |
  | ---------------- | -------------------------------- | -------------------------------------- |
  | Projection dim   | 64                               | Balance information vs RL input size   |
  | Update frequency | Async (2-5s)                     | VLM too slow for real-time             |
  | Staleness        | Exponential decay (half-life=2s) | Stale embeddings progressively ignored |
  | Training         | End-to-end with RL               | Projector learns what features matter  |
  | Fallback         | Zero vector + w=0                | RL must work without VLM               |

- [ ] Implement async VLM encoding thread on Server
- [ ] Cache and transmit embeddings via gRPC to Jetson

**Files**:

```
training/vlm_state_aligner.py   (NEW)
src/navigation/vlm_embedding.py (NEW — Jetson-side cache + staleness)
```

---

### 5. SAC Training Pipeline (D6)

- [ ] `training/train_rl.py`:
  - Framework: **Stable Baselines3** (PyTorch)
  - Algorithm: **SAC (Soft Actor-Critic)**

  #### 5.1 Environment Wrapper

  ```python
  class RobotNavEnv(gym.Env):
      """Offline RL environment using collected experience."""
      observation_space = spaces.Box(
          low=-np.inf, high=np.inf,
          shape=(167 * temporal_k,),  # ~501 when k=3 (with VLM embedding)
      )
      action_space = spaces.Box(
          low=np.array([0.0, -np.pi/4, 0, 0, 0, 0, 0]),
          high=np.array([1.0, np.pi/4, 1, 1, 1, 1, 1]),
      )
  ```

  #### 5.2 SAC Hyperparameters

  | Parameter     | Value          | Rationale                   |
  | ------------- | -------------- | --------------------------- |
  | Network       | MLP [256, 256] | Sufficient cho ~501-dim obs |
  | Learning Rate | 3e-4           | SAC default                 |
  | Batch Size    | 256            | Balance speed/stability     |
  | Replay Buffer | 100K           | Limited by Server RAM       |
  | γ (discount)  | 0.99           | Long-horizon navigation     |
  | α (entropy)   | Auto-tuned     | SAC automatic temperature   |

  #### 5.3 Config

  ```yaml
  reward:
    w_vlm: 0.35
    w_safety: 0.25
    w_intent: 0.20
    w_efficiency: 0.20
    collision_penalty: -10.0
    safety_distance: 0.5
    # VLM sub-weights
    w_vlm_safety: 0.4
    w_vlm_social: 0.3
    w_vlm_efficiency: 0.3
    # Temporal intent
    intent_lookahead_k: 5
    intent_gamma: 0.9
  ```

**Files**:

```
training/train_rl.py               (NEW)
training/robot_env.py              (NEW — gym.Env wrapper)
training/configs/rl_config.yaml    (NEW/MODIFY)
```

---

### 6. RL Policy Inference — Jetson Side (D7)

- [ ] `src/navigation/rl_policy.py`:
  - Load TensorRT `.engine` (FP16)
  - Input: observation vector (~501 floats khi k=3 with VLM embedding)
  - Output: action (velocity_scale, heading_offset, mode_logits)
  - Inference time target: < 2ms

- [ ] Update `src/main.py`:
  - **Policy switcher**: heuristic ↔ RL policy
  - VLM embedding cache integration

  ```python
  if config.policy_type == "rl" and rl_policy.is_loaded():
      nav_cmd = rl_policy.predict(observation)
  else:
      nav_cmd = heuristic_policy.decide(observation)

  # Safety monitor ALWAYS runs regardless of policy
  nav_cmd = safety_monitor.check(nav_cmd, detections, observation)
  ```

**Files**:

```
src/navigation/rl_policy.py   (NEW)
src/main.py                   (MODIFY)
```

---

### 7. Model Deployment + E2E Loop (D8, D9)

- [ ] `training/deploy_model.py`: Export SAC actor → ONNX → gRPC → Jetson
- [ ] `training/training_loop.py`: Orchestrate full cycle

  ```
  while training:
      1. Receive experiences from Jetson (rsync/gRPC)
      2. Batch evaluate with VLM (structured output — GAP #1)
      3. Compute intent trajectories from Phase 2 data — GAP #2
      4. Compute VLM embeddings for state alignment — GAP #4
      5. Compute composite scalar reward R_t
      6. Add (s, a, R_t, s') to SAC replay buffer
      7. Train SAC (N gradient steps)
      8. Every M steps: export → deploy → verify
      9. Log metrics (per-component reward breakdown)
  ```

**Files**:

```
training/deploy_model.py       (NEW/MODIFY)
training/training_loop.py      (NEW)
```

---

### 8. Testing

- [ ] `tests/test_vlm_evaluator.py`:
  - Test structured JSON output parsing
  - Test 3-axis score validation (range [-1, 1])
  - Test confidence filtering

- [ ] `tests/test_reward_shaping.py` 🔴:
  - Test scalarization: VLM structured → scalar R_vlm
  - Test temporal intent: R_intent computation with known trajectories
  - Test composite reward: all 4 components combine correctly
  - Test safety penalty overrides everything else
  - **Test R_t is always scalar ∈ ℝ**

- [ ] `tests/test_vlm_state_aligner.py`:
  - Test projection output shape (64,)
  - Test staleness decay: weight at 0s, 2s, 4s, 10s
  - Test fallback: zero vector when VLM unavailable

- [ ] `tests/test_rl_training.py`:
  - SAC trains on synthetic data with composite reward
  - Loss decreases over 100 steps

**Files**:

```
tests/test_vlm_evaluator.py       (NEW)
tests/test_reward_shaping.py      (NEW)
tests/test_vlm_state_aligner.py   (NEW)
tests/test_rl_training.py         (NEW)
tests/test_deployment.py          (NEW)
```

---

## Verification Checklist

### Functional Tests

| #   | Test                    | Criteria                                             | Status |
| --- | ----------------------- | ---------------------------------------------------- | ------ |
| V1  | VLM structured output   | Returns valid {safety, social, efficiency} ∈ [-1,1]  | ⬜     |
| V2  | VLM throughput          | ≥ 500 evaluations/hour                               | ⬜     |
| V3  | Scalarization           | R_vlm = weighted sum of 3 axes, scaled by confidence | ⬜     |
| V4  | Temporal intent reward  | R_intent correct for known trajectories (k=5, γ=0.9) | ⬜     |
| V5  | VLM staleness decay     | w=1.0 at 0s, w=0.5 at 2s, w≈0.03 at 10s              | ⬜     |
| V6  | Composite reward scalar | R_t ∈ ℝ (NOT vector) for all inputs                  | ⬜     |
| V7  | SAC training            | Loss converges on collected data                     | ⬜     |
| V8  | Policy export           | ONNX output matches PyTorch                          | ⬜     |
| V9  | gRPC deploy             | Model deployed + hot-reloaded on Jetson              | ⬜     |
| V10 | RL inference            | < 2ms latency on Jetson                              | ⬜     |
| V11 | Safety override         | Safety monitor works with RL policy                  | ⬜     |
| V12 | E2E loop                | Collect → Evaluate → Train → Deploy cycle works      | ⬜     |

### RL Quality Metrics

| Metric              | Target                                 | Status |
| ------------------- | -------------------------------------- | ------ |
| Episode reward (RL) | > Episode reward (heuristic)           | ⬜     |
| Collision rate      | < 1% of episodes                       | ⬜     |
| Smooth velocity     | Jerk < heuristic baseline              | ⬜     |
| Mode accuracy       | Correct mode > 80% scenarios           | ⬜     |
| VLM agreement       | RL decisions scored > 0.3 by VLM ≥ 60% | ⬜     |

---

## Phase 3 → Phase 4 Handoff

Khi HOÀN THÀNH Phase 3, Phase 4 cần:

1. ✅ RL policy deployed using GAP #1, #2, #4 → iterate & refine
2. ✅ Structured VLM reward pipeline → interpretable reward diagnosis
3. ✅ Temporal intent reward active → rewards anticipation over reaction
4. ✅ VLM-RL state alignment → semantic understanding in RL observations
5. ✅ Baseline metrics → compare with improvements in Phase 4
6. ✅ Intent trajectory data (Phase 2) + tracked human positions → ready for GAP #3

---

## Risk Register (Phase 3)

| Risk                             | Probability | Impact | Mitigation                                                                    |
| -------------------------------- | ----------- | ------ | ----------------------------------------------------------------------------- |
| VLM hallucination (wrong scores) | Medium      | High   | Safety penalties hard-coded. Confidence weighting. Human spot-check 10%.      |
| SAC training diverges            | Medium      | High   | Clip actions. Monitor Q-values. Fallback to heuristic.                        |
| VLM embedding stale              | High        | Medium | Staleness decay (w = 2^(-age/2s)). RL learns to work with decayed embeddings. |
| Reward non-stationarity          | High        | Medium | Track per-component reward distribution. Re-calibrate if drifted.             |
| RL policy worse than heuristic   | Medium      | Medium | Keep heuristic as fallback. Adjust reward weights.                            |
| Gemma 4 OOM trên Server          | Low         | High   | Switch to `e2b` (7.2GB). Reduce batch size.                                   |

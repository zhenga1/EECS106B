# Training Session — 16 April 2026
**Authors:** Kevin Y., Richik S. (with Claude)
**Branch:** `kevin-richik-rl-mpc`
**Task:** DroneRaceEnv — PPO training on circular 4-gate track

---

## 1. Session Overview

This session focused on diagnosing why the PPO training run produced no meaningful learning,
identifying and fixing several bugs in the reward function and reset logic, tuning PPO
hyperparameters, and setting up the codebase to also support SAC.

---

## 2. Initial Observations & Diagnosis

### 2.1 Training Speed

**Observation:** Epochs were completing at roughly 1 per second.

**Diagnosis:** This is expected and healthy. Each "epoch" in the training loop is one PPO
iteration: collect `num_envs × train_every` frames, then run gradient updates.

```
frames_per_batch = 500 envs × 32 steps = 16,000 frames/iteration
rollout_fps ≈ 18,000–26,000 (GPU-accelerated Isaac)
```

At 16k frames/iteration and ~18k fps throughput, ~1 iteration/second is correct. The noisy
`rollout_fps` plot is normal — Isaac Sim's frame rate fluctuates with rendering load.

---

### 2.2 WandB Metric Analysis

The following metrics were logged and analysed:

| Metric | Observed value | Interpretation |
|---|---|---|
| `rollout_fps` | 18k–26k, noisy | Healthy GPU throughput |
| `value_loss` | ~0.0001, spike at step ~420 | Suspiciously small; spike signals critic instability |
| `policy_loss` | ~0 to −0.06 | Near-zero is normal for PPO early on |
| `explained_var` | Crashed to **−6×10⁸** at step ~420 | **Critical failure** — critic completely useless |
| `entropy` | Decreasing 5.6 → 4.5 | Policy converging (becoming more deterministic) |
| `env_frames` | Linear increase | Correct |
| `train/stats.*` | **All 0** | No gate passes, no crashes detected |
| `episode_len` | Constant 500 | Drone always hitting max episode length |

**`explained_var` interpretation:**
`explained_var = 1 − Var(returns − values) / Var(returns)`.
A value of −6×10⁸ means the critic's value predictions are eight orders of magnitude
worse than just predicting the mean return. This was caused by 8 PPO epochs on a very
small batch (`train_every=32`, batch ≈ 16k samples), where each sample was reused ~40
times per update cycle — classic overfitting of the value function.

---

### 2.3 Root Cause: Near-Zero Reward Signal

**All `train/stats` graphs at 0** initially appeared to be a stats-collection bug.
Investigation of `EpisodeStats.add()` and `IsaacEnv._step()` confirmed stats *are* captured
correctly (the correct accumulated values are in `tensordict["next"]["stats"]` at done steps).
The true cause is that the reward is genuinely near zero every step.

**Geometric analysis of the initial drone position:**

Gate 0 has `yaw = π/2` (faces world-y direction). The drone spawns 1.5 m behind gate 0 in
gate-local x, which maps to world position `(0, −1.5, 2.25)`. The global trajectory starts
at gate 0's centre `(0, 0, 2.25)`. Since gate 4 = gate 0 (same position), the trajectory
is a closed loop, and the point nearest to `(0, −1.5, 2.25)` is near the *end* of the
trajectory (idx ≈ 195/200), whose tangent points roughly in the `+y` direction.

```
error_vec = drone − closest_point ≈ (0, −1.5, 0)
lag        = error_vec · tangent ≈ (0,−1.5,0)·(0,1,0) = −1.5   (drone is BEHIND)
contouring_vec = error_vec − lag · tangent = (0,−1.5,0) − (−1.5)·(0,1,0) = (0, 0, 0)
contouring_err = 0   ← drone is geometrically ON the trajectory tangent extension
lag_err    = clamp(lag, 0, 1.5) = clamp(−1.5, 0, 1.5) = 0   ← old code clamped wrong way
delta_lag  = lag − prev_lag = −1.5 − 0 = −1.5  (step 1 only)
           ≈ 0 thereafter   (hovering drone, lag constant)
speed_bonus ≈ 0   (drone barely moves)
```

Net reward ≈ `time_penalty = −0.01/step` only → return ≈ **−5/episode**. WandB displayed
this as "0" because the y-axis scale made it visually indistinguishable. The drone had
zero directional gradient — PPO could not determine which actions lead to gate approaches.

---

## 3. Bugs Fixed

### 3.1 `prev_lag` Zero-Initialisation (Spurious First-Step Spike)

**File:** `omni_drones/envs/drone_race/drone_race.py` — `_reset_idx()`

**Bug:** `self.prev_lag[env_ids] = 0.0`. On the first step of every episode,
`delta_lag = lag − 0 = −1.5`, producing `progress_reward = −1.5 × 2.0 = −3.0` — a large
spurious penalty that confuses the critic on every episode start.

**Fix:** Compute the actual lag at the drone's spawn position using the same closest-point
projection as `_compute_reward_and_done`, and set `self.prev_lag[env_ids]` to that value.
This ensures `delta_lag = 0` on step 1.

```python
# [KY, Claude] Initialize prev_lag to the actual lag at the drone's start position
# so that delta_lag = 0 on the first step (no spurious reward spike).
reset_traj  = self.global_trajectories[env_ids]           # (M, num_points, 3)
drone_pos_exp = drone_start_pos.unsqueeze(1)               # (M, 1, 3)
dists       = torch.norm(reset_traj - drone_pos_exp, dim=-1)
closest_idx = torch.argmin(dists, dim=1)
...
self.prev_lag[env_ids] = (error_vec * tangent).sum(dim=-1).detach()
```

---

### 3.2 `lag_err` Direction Bug

**File:** `drone_race.py` — `_compute_reward_and_done()`

**Bug:** `lag_err = torch.clamp(lag, 0.0, 1.5)` — this penalises the drone for having
`lag > 0`, i.e., for being *ahead* of the nearest trajectory point. In normal forward
progress, `lag > 0` (drone has just passed the nearest point), so this code applied a
lag penalty to correct behaviour.

**Correct MPCC semantics:** The lag error `e_l` should penalise falling *behind* the
path (`lag < 0`), not being ahead.

**Fix:** `lag_err = torch.clamp(-lag, 0.0, 1.5)`

This gives `lag_err = 0` when the drone is on or ahead of the nearest point, and grows
as the drone falls behind — matching the MPCC paper's definition.

---

### 3.3 `delta_lag` Spikes from Closed-Loop Trajectory Ambiguity

**File:** `drone_race.py` — `_compute_reward_and_done()`

**Bug:** Gate 0 and gate 4 are at the same world position `(0, 0, 2.25)`. The `argmin`
over all trajectory points is therefore ambiguous near this location: the nearest point
could be idx=0 (start) or idx≈199 (end), and can flip between them in a single step.
This caused `delta_lag` to jump by the entire trajectory length (~15+ m) in one step,
producing a massive spurious `progress_reward`.

**Fix:** Clamp `delta_lag` to a physically plausible per-step range:

```python
# [KY, Claude] cap to ~10× max realistic speed to prevent argmin-flip artifacts
delta_lag = delta_lag.clamp(-0.5, 0.5)
```

At dt=0.01 s and max speed ~5 m/s, max legitimate progress per step ≈ 0.05 m. The
0.5 m cap is 10× generous and catches only genuine index discontinuities.

---

### 3.4 Missing Gate-Approach Reward (Bootstrap Signal)

**File:** `drone_race.py` — `_compute_reward_and_done()`

**Problem:** Without any approach-to-gate reward, the MPCC-style terms (`delta_lag`,
`contouring_err`) produce near-zero signal during early training when the drone is hovering
near the spawn point. PPO has no gradient direction pointing toward gates.

**Fix:** Added `approach_reward` using `self.prev_distance_to_gate` (already tracked):

```python
# 2j. [KY, Claude] Gate approach reward — positive when closing distance to current gate.
approach_reward = (self.prev_distance_to_gate - distance_to_gate) * self.w_approach
```

With `w_approach = 2.0`, the drone receives a clear `+` signal whenever it moves toward
the current target gate, and `−` when moving away. This gradient is non-zero from step 1
and acts as the bootstrap signal that allows PPO to begin learning before the MPCC terms
become meaningful.

Added to `cfg/task/DroneRace.yaml`:
```yaml
w_approach: 2.0  # [KY, Claude] bootstrap signal
```

---

## 4. Hyperparameter Tuning

### 4.1 `train_every`: 32 → 256

**Reason:** With `train_every=32` and 500 envs, each batch contained only
`500 × 32 = 16,000` samples. With `max_episode_length=500`, each batch covered only
`32/500 = 6.4%` of a typical episode. GAE return estimates over such short windows are
extremely noisy — the critic never sees the long-term consequence of early-episode actions.

At `train_every=256`, each batch covers `500 × 256 = 128,000` samples and ~51% of a
typical episode, giving much better return estimates.

**Effect on wall-clock time:** Each iteration now collects 8× more data, so iterations
are ~8× slower (roughly 8 seconds each). However, the gradient signal quality improves
dramatically, so far fewer iterations are needed.

### 4.2 `ppo_epochs`: 8 → 4

**Reason:** With `train_every=32`, 8 PPO epochs × 5 minibatches = **40 gradient updates
on 16k samples**. Each sample was reused 40 times before new data arrived. This caused
severe overfitting of the value function to the current small batch, which is the direct
cause of the `explained_var → −6×10⁸` crash at step ~420.

**Effect:** With `train_every=256` (128k samples) and 4 epochs × 5 minibatches = 20
gradient updates, each sample is reused 20 times on a much larger, more diverse batch.
The overfitting risk is substantially reduced.

**Note on risk direction:** Reducing `ppo_epochs` is the *safer* direction — it means
more conservative policy updates. The risk is *lower* sample efficiency (need more env
steps to learn the same amount), not instability.

### 4.3 Summary of Final PPO Config (`cfg/algo/DroneRace.yaml`)

```yaml
train_every: 256   # up from 32 — larger batches, less noisy GAE
ppo_epochs: 4      # down from 8 — prevents value function overfitting
num_minibatches: 5
clip_param: 0.2
entropy_coef: 0.003
gae_lambda: 0.95
gamma: 0.99
max_grad_norm: 5.0
actor:
  lr: 0.0002
  hidden_units: [256, 256, 256]
  activation: elu
critic:
  lr: 0.0002
  hidden_units: [256, 256, 256]
  activation: elu
```

---

## 5. Reset Logic Improvements

### 5.1 Position Randomisation

**Problem:** All 500 environments reset to the exact same spawn position
(`first_gate_pos + offset_local_rotated`), producing highly correlated rollouts.
The 500 parallel environments effectively behaved like a single environment, providing
no diversity benefit.

**Fix:** Enabled the previously-commented position perturbation:

```python
# [KY, Claude] Enable position randomisation for diverse starting states
pos_perturbation = self.init_pos_dist.sample(env_ids.shape) - self.init_pos_dist.mean
drone_start_pos   = first_gate_pos + offset_world + pos_perturbation
```

`init_pos_dist` is `Uniform([−1,−1,1.5], [1,1,2.5])`, so after mean-centering:
±1 m in x/y, ±0.5 m in z. Combined with the existing orientation randomisation
(±36° roll/pitch), each of the 500 envs now starts in a meaningfully different state.

**Expected effect:** More diverse batch → less gradient correlation across envs →
PPO update closer to the true policy gradient.

---

## 6. SAC Support

### 6.1 What Was Broken

`cfg/task/DroneRace.yaml` specifies `ppo_cfg: DroneRace.yaml`. The training script
unconditionally merged this file onto whichever algo config was loaded. When using
`algo=sac`, this overwrote `name: sac` with `name: ppo`, silently falling back to PPO.

### 6.2 Fix in `scripts/train.py`

```python
PPO_ALGOS = {"ppo", "ppo_rnn", "ppo_adapt", "mappo", "happo"}
ppo_cfg_name = cfg.task.get("ppo_cfg", None)
if ppo_cfg_name and cfg.algo.name.lower() in PPO_ALGOS:
    # ... load and merge task-specific PPO config
```

The `ppo_cfg` block is now skipped for SAC/TD3/etc., leaving their configs intact.

### 6.3 SAC Status: Did Not Work

After fixing the `ppo_cfg` gate in `train.py`, a SAC run was attempted with:

```bash
python3 scripts/train.py algo=sac task=DroneRace headless=true \
    wandb.run_name=mpcc_sac_DroneRace_16_April_150M \
    total_frames=150000000
```

**Result:** SAC failed to run correctly due to unresolved bugs — likely related to
incompatibilities between the `AgentSpec` old API and the current `DroneRaceEnv`
observation/action spec structure, or missing SAC-specific config keys. The error was not
fully diagnosed within this session.

**Decision:** PPO was retained as the primary training algorithm. The SAC code path
fix in `train.py` (PPO_ALGOS gate) was kept as it corrects a real bug, but SAC training
is deferred to a future session.

SAC uses the old `AgentSpec` API (already handled in `train.py`). The replay buffer
warms up for the first few thousand steps before gradients fire.

### 6.4 SAC Hyperparameter Note: `gradient_steps`

The default `sac.yaml` has `gradient_steps: 2048`. This was calibrated for a small
number of envs. With 500 envs:

```
frames_per_batch = 500 × 64 = 32,000
UTD = gradient_steps / frames_per_batch = 2048 / 32,000 ≈ 0.064
```

This is a **low update-to-data ratio** — only 1 gradient step per ~15 new samples.
Since data collection in Isaac is cheap (GPU-accelerated), gradient computation is
likely the wall-clock bottleneck. Reducing to `gradient_steps: 512` gives UTD ≈ 0.016
and faster iteration cycles. This can be scaled up if sample efficiency matters more.

---

## 7. Algorithm Comparison: PPO vs SAC

| | PPO | SAC |
|---|---|---|
| Data strategy | On-policy (discard after update) | Off-policy (replay buffer, 1M capacity) |
| Entropy | Coefficient (0.003) | Structural (maximised by objective) |
| Exploration | Degrades as std collapses | Maintained by α temperature |
| Rare event reuse | No | Yes — good trajectory replayed many times |
| Vectorised env fit | Excellent (data is cheap) | Moderate (replay buffer less critical) |
| API | New (`obs_spec, act_spec, rew_spec`) | Old (`AgentSpec`) |
| When to prefer | Stable reward signal present | Reward near-zero, need persistent exploration |

**Conclusion:** SAC provides better exploration guarantees via structural entropy
maximisation, and its replay buffer can amplify rare "accidentally good" trajectories
(drone drifts toward a gate). However, the root cause of the dead policy was a reward
function that produced near-zero gradient signal — an issue neither algorithm can
overcome without reward shaping. The approach reward (`w_approach`) and `prev_lag`
fix are more impactful than switching algorithms.

---

## 8. Complete List of Code Changes

| File | Change | Tag |
|---|---|---|
| `drone_race.py` `__init__` | Added `self.w_approach` | `[KY, Claude]` |
| `drone_race.py` `_reset_idx` | Removed zero-init of `prev_lag`; compute from spawn position | `[KY, Claude]` |
| `drone_race.py` `_reset_idx` | Enabled position randomisation (`init_pos_dist`) | `[KY, Claude]` |
| `drone_race.py` reward | `lag_err`: `clamp(lag,0,1.5)` → `clamp(-lag,0,1.5)` | `[KY, Claude]` |
| `drone_race.py` reward | Added `delta_lag.clamp(-0.5, 0.5)` | `[KY, Claude]` |
| `drone_race.py` reward | Added `approach_reward` term | `[KY, Claude]` |
| `cfg/task/DroneRace.yaml` | Added `w_approach: 2.0` | `[KY, Claude]` |
| `cfg/algo/DroneRace.yaml` | `train_every: 32 → 256` | `[KY, Claude]` |
| `cfg/algo/DroneRace.yaml` | `ppo_epochs: 8 → 4` | `[KY, Claude]` |
| `scripts/train.py` | `ppo_cfg` block now gated to PPO algos only | `[KY, Claude]` |

---

## 9. Expected Training Behaviour After Fixes

With the above changes, we expect:

- **`train/stats.return`** to be visibly negative (dominated by `time_penalty` and
  `approach_reward` when moving away from gate) and to improve (become less negative)
  as the drone learns to approach gates.
- **`train/stats.gates_passed`** to start at 0 and slowly increase as the drone
  learns to fly through gate 0.
- **`explained_var`** to stay positive (0 → 1 range) once the value function has
  a meaningful signal to predict.
- **`entropy`** to decrease gradually as the policy commits to gate-approaching actions.
- **Episode length** to decrease below 500 once crash termination begins firing
  (drone explores more aggressively and occasionally crashes).

---

## 10. Training Commands

The following commands were used (or prepared) at the end of this session:

### PPO (primary run)
```bash
python3 scripts/train.py algo=ppo task=DroneRace headless=true \
    wandb.run_name=mpcc_ppo_DroneRace_16_April_500M \
    total_frames=500000000
```

### SAC (attempted, did not complete — see §6.3)
```bash
python3 scripts/train.py algo=sac task=DroneRace headless=true \
    wandb.run_name=mpcc_sac_DroneRace_16_April_150M \
    total_frames=150000000
```

---

## 11. Open Questions / Next Steps

- [ ] Observe PPO run and verify `train/stats.return` is non-zero and improving.
- [ ] Tune `w_approach` — if too high, the policy may learn to orbit the gate without passing.
- [ ] Consider curriculum: start with fixed spawn, enable position randomisation only
      after the drone can reliably approach gate 0.
- [ ] Debug SAC compatibility with `DroneRaceEnv` and retry SAC comparison run.
- [ ] Investigate the closed-loop trajectory ambiguity more rigorously — consider
      tracking `prev_closest_idx` and constraining the search to a local window.
- [ ] Gate-specific `closest_idx` gating: constrain trajectory search to the segment
      between `gate_indices[i]` and `gate_indices[i]+1` to eliminate cross-loop jumps.

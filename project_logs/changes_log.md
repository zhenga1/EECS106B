# EECS106B Drone Racing RL — Change Log

## Architecture
- **Env**: `omni_drones/envs/drone_race/drone_race.py`
- **Policy**: PPO via `omni_drones/learning/ppo/ppo.py`
- **Config**: `cfg/task/DroneRace.yaml`, `cfg/algo/DroneRace.yaml`
- **Track**: 12-gate racetrack, 500 parallel envs, Iris drone + RateController

---

## Changes (chronological)

### 1. MPCC-style reward — gate-to-gate line segments
- Replaced global spline trajectory with per-step line segment from `prev_gate_center → current_gate_center`
- MPCC decomposition: `lag` (forward progress) + `contouring_err` (lateral deviation)
- Reward: `w_progress * Δlag - w_contouring * e_c²`
- Degenerate case (gate_idx==0): uses gate forward axis as tangent

### 2. Pruned redundant rewards
- Removed: `alive_bonus`, `vel_up_reward`, `lag_penalty`
- Kept: `approach_reward`, `altitude_bonus`, `upright_bonus`, `gate_bonus`, `smooth_penalty`

### 3. Angular rate penalty
- Added `angular_rate_penalty = -ang_rate² * w_ang_rate` to discourage spinning
- `w_ang_rate: 0.005`

### 4. Progress-based stall termination
- Counter increments each step, resets on gate passage
- Fires penalty + terminates after `stall_patience=200` steps without passing a gate
- Replaces speed/altitude-based stall detection

### 5. Cosine entropy annealing (PPO)
- Added `_step_entropy_schedule()` with cosine decay + optional SGDR warm restarts
- Config: `entropy_coef: 0.01 → entropy_coef_end: 0.00005` over `entropy_anneal_steps: 4000`

### 6. Separate value clip param (PPO)
- Added `value_clip_param: 10.0` independent of `clip_param: 0.1`
- Fixed negative explained variance caused by overly tight value clipping

### 7. Simplified collision detection
- **Before**: 4 conditions — `ground_crash | bounds_crash | flipped_crash | contact_crash`
- **After**: `contact_forces.norm() > 1.0` (physics contact only)
- Crash penalty: flat `-crashed * w_crash` (removed cone-shaped scaling)

---

## Bug Fixes

| Bug | Fix |
|-----|-----|
| `AttributeError: no attribute 'gates'` | Moved `RigidPrimView` init before first `get_world_poses()` call |
| `UnboundLocalError: first_gate_center` | Moved computation inside try block |
| `RuntimeError: dtype Bool/Float mismatch` | `self.stats.exclude("success")[env_ids] = 0.0` |
| `RuntimeError: tensors on different devices` | `register_buffer('I', ...)` in all 3 controller classes |
| `ConfigKeyError: entropy_coef_end` | `OmegaConf.set_struct(cfg.algo, False)` before merge in train.py |
| `ang_vel × ang_vel = 0` (gyroscopic term) | Changed to `torch.linalg.cross(ang_vel, I @ ang_vel)` |

### 8. Restored upright bonus (2026-04-22)
- `upright_bonus` was commented out and `drone_up` was never computed, giving zero gradient to stabilise orientation
- Fixed: `drone_up = quat_axis(drone_rot.squeeze(1), axis=2)`, `upright_bonus = drone_up[:, 2].clamp(min=0) * w_upright`, added to reward sum
- Effect: `drone_uprightness` stat was ~0.34 (tumbling); fixed to climb toward 1.0

### 9. Fixed inverted approach reward (2026-04-22)
- `approach_reward` used `squared_progress = d_current² − d_previous²`, which is **negative** when approaching → drone was rewarded for flying away from gates
- Fixed: `approach_reward = (self.prev_distance_to_gate - distance_to_gate) * w_approach`
- Effect: `gates_passed` went from 0.0 to ~0.5

### 10. Simplified progress reward block (2026-04-22)
- Removed unused `squared_progress` accumulation buffer (`recent_squared_progress`, `prev_drone_pos_flat`) from the per-step computation path
- `approach_reward` is now a single line; `recent_squared_progress` buffer kept in `__init__` for potential reuse

### 11. Added flipped_crash termination (2026-04-22)
- `crashed` only checked contact forces; an inverted drone in mid-air accumulated approach reward while falling
- Added `flipped_crash = drone_up[:, 2] < 0.0` and `crashed = contact_crash | flipped_crash`
- Provides immediate crash penalty and episode reset when drone inverts

### 12. Fixed lag frame-jump bug after gate transitions (2026-04-22)
- On gate transition at step t: `prev_lag` was set to lag in the OLD segment (large value)
- At step t+1: lag computed in the NEW segment (small value near 0) → `delta_lag` large negative → `progress_reward` strongly penalised the step right after passing a gate
- Fixed: on `gate_index_changed` steps, `prev_lag` is seeded with the drone's lag in the NEW segment's frame so that step t+1's delta is meaningful

---

## Training Results (epoch 16, ~2M frames)
```
policy_loss:    -0.0005  (healthy: negative = policy improving)
value_loss:      7.8e-8
entropy:         5.68
explained_var:   0.87    (good after value_clip_param fix)
actor_grad_norm: 1.88
critic_grad_norm: 0.002
```

---

## Key Config Values (DroneRace.yaml)
```yaml
w_progress:   2.0
w_contouring: 0.1
w_gate:       50.0
w_approach:   2.0
w_altitude:   2.0
w_crash:      20.0      # flat penalty on contact
w_stall:      30.0
stall_patience: 200
w_ang_rate:   0.005
w_smooth:     0.001
```

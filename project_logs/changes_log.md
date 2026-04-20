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

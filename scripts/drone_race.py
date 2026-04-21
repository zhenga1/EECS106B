# =============================================================================
# DRONE RACE ENVIRONMENT — MPCC-COST-BASED REWARD VERSION
# =============================================================================
#
# FILE: drone_race_mpcc.py
#
# USE THIS FILE FOR:
#   Training an RL policy whose reward terms are derived from the cost function
#   of a Model Predictive Contouring Controller (MPCC). This is the approach
#   described in the Team 9 Week 2 presentation (Slide 5).
#
# DO NOT OVERWRITE:
#   drone_race_circular.py  — PPO stage-based reward (circular track)
#   drone_race.py           — PPO stage-based reward + race track / Stage 4
#
# WHAT IS MPCC?
# ================
# Model Predictive Contouring Control (MPCC) is a classical trajectory tracking
# algorithm that parameterises the track by ARC LENGTH instead of time, then
# solves a constrained optimisation at every timestep to simultaneously:
#   1. Maximise progress along the arc (go as far along the track as possible)
#   2. Minimise contouring error (lateral drift from the racing line)
#   3. Minimise lag error (falling behind the reference speed)
#
# This file takes those same cost terms and NEGATES them to form dense RL rewards.
# "The negative of an MPCC cost term = an RL reward term that incentivises the
#  same optimal behaviour, without needing to solve an MPC optimisation."
#
# FULL REWARD FORMULA (from Team 9 presentation, Slide 5):
# -------------------------------------------------------
#   r = Δlag  · w_prog          # ARC-LENGTH PROGRESS  (+, want high)
#     − e_c²  · w_cont          # CONTOURING ERROR     (−, want low)
#     − e_l²  · w_lag           # LAG ERROR            (−, want low)
#     + gate_bonus               # GATE PASSAGE FLAT BONUS
#     + centering · w_center     # GATE CENTER QUALITY BONUS (TOGT insight)
#     + speed · w_spd            # SPEED REWARD         (+, want high)
#     − w_time                   # TIME PENALTY         (constant per step)
#     − ‖Δa‖ · w_sm             # SMOOTHNESS PENALTY   (−, want low)
#     + done · w_lap             # LAP COMPLETION BONUS (one-time)
#
# DEFINITIONS:
#   Δlag  = change in arc-length projection since last step (metres forward)
#   e_c   = contouring error = perpendicular distance from drone to centerline
#   e_l   = lag error = distance remaining to the next gate along the segment
#   speed = ‖v‖, the drone's current speed magnitude
#   Δa    = change in policy action since last step
#   done  = 1 when track_completed
#
# THREE TRAINING STAGES (controlled by training_stage: 1/2/3 in task YAML):
#   Stage 1: Progress + uprightness + crash penalty.
#            Teaches the drone to fly forward along the track without crashing.
#   Stage 2: Stage 1 + contouring penalty + gate bonus + centering + speed.
#            Teaches the drone to stay on the racing line and pass gates.
#   Stage 3: Stage 2 + lag penalty + smoothness + lap bonus. Full MPCC reward.
#            Teaches the drone to race at speed with smooth control.
#
# =============================================================================

import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite, DiscreteTensorSpec, BinaryDiscreteTensorSpec

import isaacsim.core.utils.prims as prim_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, quat_rotate, quat_rotate_inverse, quat_axis
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.robots import ASSET_PATH
from pxr import UsdPhysics

try:
    from isaacsim.util.debug_draw import _debug_draw
    DEBUG_DRAW_AVAILABLE = True
except ImportError:
    DEBUG_DRAW_AVAILABLE = False
    _debug_draw = None


class DroneRaceEnv(IsaacEnv):
    r"""
    Drone racing environment with MPCC-derived reward terms.

    ========================
    ARC-LENGTH PARAMETERIZATION
    ========================
    The track is represented as a piecewise-linear path connecting the center
    points of the gates in sequence. At each step we:
      1. Identify which segment the drone is on (from gate_indices).
      2. Project the drone's position onto that segment.
      3. Compute global arc-length θ = cumulative length to segment start
         + local projection distance along segment.
      4. Contouring error e_c = perpendicular distance from drone to segment.
      5. Lag error e_l = remaining distance to next gate along segment.
      6. Progress Δlag = θ(t) − θ(t−1) = metres gained along track this step.

    ========================
    THREE TRAINING STAGES
    ========================
    Stage 1 — Stability/Progress:
        r = Δlag·w_prog + uprightness·w_up − w_time − crash·w_crash
        NO contouring or lag penalties. Drone just learns to fly forward.
        Use: DroneRace_task_mpcc_stage1.yaml

    Stage 2 — Navigation/Contouring:
        Adds: − e_c²·w_cont + gate_bonus + centering·w_center + speed·w_spd
        Drone learns to stay on the racing line and pass through gates.
        Use: DroneRace_task_mpcc_stage2.yaml

    Stage 3 — Full MPCC Reward:
        Adds: − e_l²·w_lag − ‖Δa‖·w_sm + done·w_lap
        Full formula from Team 9 Slide 5. Load Stage 2 checkpoint.
        Use: DroneRace_task_mpcc_stage3.yaml
    """

    def __init__(self, cfg, headless):

        # ==================================================================
        # STAGE CONFIGURATION
        # ==================================================================
        # training_stage: 1 = progress only, 2 = +contouring, 3 = full MPCC
        self.training_stage = cfg.task.get("training_stage", 2)
        print(f"[DroneRaceEnv MPCC] Training stage: {self.training_stage}")

        # ==================================================================
        # DISTRIBUTED INITIALIZATION (Song et al. 2021)
        # ==================================================================
        self.distributed_init = cfg.task.get("distributed_init", False)
        print(f"[DroneRaceEnv MPCC] Distributed init: {self.distributed_init}")

        # sim_dt: seconds per environment step (used to scale velocity rewards)
        self.sim_dt = cfg.sim.dt * cfg.sim.substeps   # e.g. 0.002 × 5 = 0.01 s

        # ==================================================================
        # MPCC REWARD HYPERPARAMETERS
        # All loaded from task YAML. Defaults correspond to Stage 2 values.
        # ==================================================================

        # --- STAGE 1: STABILITY / PROGRESS ---
        # w_prog: weight on arc-length progress Δlag (metres/step).
        # This is the RL equivalent of the MPCC "maximise progress" objective.
        # INCREASE if drone doesn't move forward. DECREASE if drone rushes blindly.
        self.mpcc_w_prog  = cfg.task.get("mpcc_w_prog",  2.0)

        # w_up: uprightness reward weight (z-axis alignment).
        # Stage 1 only — teaches basic flight stability before adding MPCC terms.
        # DECREASE in later stages as the drone becomes more stable.
        self.mpcc_w_up    = cfg.task.get("mpcc_w_up",    0.5)

        # w_time: constant per-step time penalty (encourages speed).
        # This maps to the MPCC "time-optimal" component in the cost function.
        self.mpcc_w_time  = cfg.task.get("mpcc_w_time",  0.005)

        # w_crash: penalty when the drone crashes (ground, OOB, flip, contact).
        # Not in the original MPCC cost, but necessary for safe RL exploration.
        self.mpcc_w_crash = cfg.task.get("mpcc_w_crash", 20.0)

        # --- STAGE 2: NAVIGATION / CONTOURING ---
        # w_cont: contouring error weight. Penalises e_c² (quadratic lateral drift).
        # This is the CORE MPCC term — it shapes the drone onto the racing line.
        # INCREASE to keep the drone tighter to the centerline (slower but cleaner).
        # DECREASE if drone oscillates around the line (over-correction).
        self.mpcc_w_cont  = cfg.task.get("mpcc_w_cont",  0.3)

        # gate_bonus: flat reward for passing through any part of a gate.
        # Not in MPCC theory but necessary for sparse gate-passing signal.
        self.mpcc_gate_bonus   = cfg.task.get("mpcc_gate_bonus",   10.0)

        # w_center: weight on gate centering score (0=edge, 1=center).
        # From TOGT (Time-Optimal Gate Traversal) insight on the slide:
        # flying through the gate center reduces drag and enables higher exit speed.
        self.mpcc_w_center= cfg.task.get("mpcc_w_center", 3.0)

        # w_spd: speed reward weight. Rewards ‖v‖ (drone speed in m/s).
        # Encourages the drone to maintain high speed along the track.
        # INCREASE for Stage 3 to make the drone fly as fast as possible.
        # DECREASE if drone sacrifices accuracy for speed.
        self.mpcc_w_spd   = cfg.task.get("mpcc_w_spd",   0.05)

        # --- STAGE 3: FULL MPCC REWARD ---
        # w_lag: lag error weight. Penalises e_l² (falling behind in arc-length).
        # e_l = distance remaining to the next gate along the current segment.
        # This penalises the drone for lingering or drifting away from the gate.
        # Combined with w_prog (which rewards moving toward the gate), this
        # creates a "sandwich" that pushes the drone to progress efficiently.
        # INCREASE if drone is achieving progress but drifting between gates.
        self.mpcc_w_lag   = cfg.task.get("mpcc_w_lag",   0.01)

        # w_sm: action smoothness weight. Penalises ‖Δa‖ (action change per step).
        # MPCC achieves smoothness via the control cost term. We replicate this
        # using the difference between consecutive policy outputs.
        # INCREASE if drone exhibits chattering or oscillating control.
        self.mpcc_w_sm    = cfg.task.get("mpcc_w_sm",    0.01)

        # w_lap: lap completion bonus. One-time reward for completing a full lap.
        # Not in MPCC theory, but critical for the RL episode structure.
        self.mpcc_w_lap   = cfg.task.get("mpcc_w_lap",   100.0)

        # ==================================================================
        # CIRCULAR TRACK OUT-OF-BOUNDS
        # ==================================================================
        # Fixed for the 4-gate circular track.
        # If you use this with the race track YAML, override these in the YAML
        # (add bounds_x_min etc.) and update _compute_reward_and_done accordingly.
        self._oob_x_min = cfg.task.get("bounds_x_min", -5.0)
        self._oob_x_max = cfg.task.get("bounds_x_max", 15.0)
        self._oob_y_min = cfg.task.get("bounds_y_min", -10.0)
        self._oob_y_max = cfg.task.get("bounds_y_max", 10.0)
        self._oob_z_max = cfg.task.get("bounds_z_max",  8.0)

        # ==================================================================
        # GATE / TRACK CONFIGURATION
        # ==================================================================
        self.gate_scale = cfg.task.gate_scale
        self.gate_asset_path = cfg.task.get(
            "gate_asset_path", ASSET_PATH + "/gate/gate_a2rl.usd"
        )
        self.track_config = cfg.task.get("track_config", None)
        if self.track_config is not None:
            self.num_gates   = len(self.track_config)
            self.track_type  = "config"
            self.gate_height = cfg.task.get("gate_height", 2.0)
        else:
            self.num_gates    = int(cfg.task.num_gates)
            self.track_radius = cfg.task.track_radius
            self.gate_spacing = cfg.task.gate_spacing
            self.gate_height  = cfg.task.gate_height
            self.track_type   = "circular"

        import traceback, sys

        print(f"[DroneRaceEnv MPCC] Initializing, num_gates={self.num_gates}")
        try:
            super().__init__(cfg, headless)
        except Exception as e:
            traceback.print_exc(); raise

        try:
            self.drone.initialize(track_contact_forces=True)
        except Exception as e:
            traceback.print_exc(); raise

        # Per-environment gate tracking
        self.gate_indices    = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.gate_passed     = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.track_completed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.gate_width      = cfg.task.get("gate_width", 1.0)
        self.prev_drone_in_gate_frame = torch.zeros(self.num_envs, 3, device=self.device)

        # MPCC-specific buffers
        # prev_arc_length: global arc-length from previous step (for Δlag computation)
        self.prev_arc_length  = torch.zeros(self.num_envs, device=self.device)
        # last_action: policy output from previous step (for smoothness penalty)
        self.last_action = torch.zeros(
            self.num_envs, 1, self.drone.action_spec.shape[-1], device=self.device
        )
        self.effort = torch.zeros_like(self.last_action)

        # Gate view
        try:
            self.gates = RigidPrimView(
                "/World/envs/env_*/Gate_*",
                reset_xform_properties=False,
                prepare_contact_sensors=False,
                shape=[self.num_envs, self.num_gates],
            )
            self.gates.initialize()
        except Exception as e:
            traceback.print_exc(); raise

        # Drone initial state buffers
        self.init_vels      = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos  = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-1.0, -1.0, 1.5], device=self.device),
            torch.tensor([ 1.0,  1.0, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([ .2,  .2, 0.], device=self.device) * torch.pi
        )
        self.offset_local = torch.tensor(
            [-1.5, 0.0, self.gate_height / 2.0], device=self.device
        )
        self.alpha = 0.8

        # Debug visualization
        self.debug_gate_origins = cfg.task.get("debug_gate_origins", False)
        if self.debug_gate_origins and DEBUG_DRAW_AVAILABLE:
            self.draw = _debug_draw.acquire_debug_draw_interface()
            self.axis_length = cfg.task.get("debug_axis_length", 0.3)
        else:
            self.draw = None

        # ==================================================================
        # BUILD MPCC ARC-LENGTH REFERENCE FROM TRACK CONFIG
        # ==================================================================
        # This precomputes the piecewise-linear centerline and its arc-length
        # parameterization from the gate positions in the YAML config.
        # This is done ONCE at init time — no runtime cost per step.
        self._build_mpcc_reference()

    # =========================================================================
    # MPCC REFERENCE CONSTRUCTION
    # =========================================================================
    def _build_mpcc_reference(self):
        """Precompute the arc-length parameterized centerline from gate positions.

        The centerline is the piecewise-linear path connecting the center of
        each gate in sequence. For a gate at position [x,y,z] (gate origin =
        bottom-center of the gate aperture), the gate center is at [x,y,z+h/2]
        where h = gate_height. For yaw-only gates the z-offset is unaffected
        by rotation (yaw rotates around z-axis).

        Stores:
            self.mpcc_gate_centers: (num_gates, 3) gate centers in env/world frame.
            self.mpcc_arc_lengths:  (num_gates,) cumulative arc-length to each gate.
                arc_lengths[0] = 0.0 (start of track)
                arc_lengths[i] = total path length from gate 0 to gate i.
            self.mpcc_total_length: float — total track circumference.
        """
        gate_keys = sorted(self.track_config.keys(), key=lambda x: int(x))
        centers = []
        for key in gate_keys:
            gcfg = self.track_config[key]
            pos  = gcfg.get("pos", [0.0, 0.0, 1.0])
            x, y, z = pos
            # Gate center: gate origin is bottom-center; center is height/2 above.
            # For yaw rotations (rotate around z), the vertical offset stays vertical.
            gc = torch.tensor([x, y, z + self.gate_height / 2.0],
                              dtype=torch.float32, device=self.device)
            centers.append(gc)

        self.mpcc_gate_centers = torch.stack(centers)  # (num_gates, 3)

        # Cumulative arc-lengths
        arc = [0.0]
        for i in range(1, len(centers)):
            seg_len = torch.norm(centers[i] - centers[i - 1]).item()
            arc.append(arc[-1] + seg_len)

        self.mpcc_arc_lengths  = torch.tensor(arc, dtype=torch.float32, device=self.device)
        self.mpcc_total_length = float(arc[-1])

        print(f"[DroneRaceEnv MPCC] Centerline arc lengths: "
              f"{[round(a, 2) for a in arc]}")
        print(f"[DroneRaceEnv MPCC] Total track length: {self.mpcc_total_length:.2f}m")

    # =========================================================================
    # MPCC ERROR COMPUTATION (called every step)
    # =========================================================================
    def _compute_mpcc_errors(self, drone_pos_flat, gate_index_changed):
        """Compute the three core MPCC error signals from the drone's position.

        For each environment, the drone is on the segment from the PREVIOUS gate
        center (g1) to the CURRENT target gate center (g2). We project the
        drone's position onto this segment to compute:

          e_c  — contouring error: perpendicular distance from segment (metres).
                 = ‖drone_pos − closest_point_on_segment‖
                 Corresponds to MPCC contouring cost q_c · e_c²

          e_l  — lag error: remaining distance to the next gate along segment.
                 = segment_length − longitudinal_projection_of_drone
                 Corresponds to MPCC lag cost q_l · e_l²
                 Note: e_l = 0 when drone is exactly at the next gate center.

          Δlag — arc-length progress since the previous step (metres).
                 = θ(t) − θ(t−1)  where θ is the drone's global arc-length.
                 Corresponds to MPCC progress reward μ_v · θ̇
                 Positive when moving toward the next gate.

        Args:
            drone_pos_flat:    (N, 3) drone positions in env frame.
            gate_index_changed:(N,) bool — True for envs whose gate just advanced.
                               We zero Δlag for these to avoid jumps when the
                               segment changes (same as for distance progress reward).

        Returns:
            e_c        (N,) — contouring error in metres (unsigned, want small)
            e_l        (N,) — lag error in metres (unsigned, want small)
            delta_lag  (N,) — arc-length progress this step (want positive & large)
            global_arc (N,) — current global arc-length (stored for next step)
        """
        batch = torch.arange(self.num_envs, device=self.device)

        # For a circular track the last gate has the same position as gate 0.
        # We exclude it from the "unique" segment count so the wrap-around works.
        # num_unique = num_gates - 1 for a circular track.
        num_unique = self.num_gates - 1

        # Segment start = center of the PREVIOUS gate (where drone came from)
        prev_idx = (self.gate_indices - 1 + num_unique) % num_unique   # (N,)

        g1 = self.mpcc_gate_centers[prev_idx]             # (N, 3) segment start
        g2 = self.mpcc_gate_centers[self.gate_indices]    # (N, 3) segment end (target)

        # Segment tangent (unit vector in direction of travel along centerline)
        segment     = g2 - g1                                              # (N, 3)
        seg_len     = torch.norm(segment, dim=-1, keepdim=True).clamp_min(1e-6)  # (N, 1)
        T_hat       = segment / seg_len                                    # (N, 3) unit tangent
        seg_len_1d  = seg_len.squeeze(-1)                                  # (N,)

        # Vector from segment start to drone
        dp           = drone_pos_flat - g1                                 # (N, 3)

        # Longitudinal component: how far the drone has progressed along segment
        longitudinal = (dp * T_hat).sum(-1)                               # (N,)

        # Clamp to [0, seg_len] → closest point is within the segment
        long_clamped = longitudinal.clamp(0.0, seg_len_1d)                # (N,)

        # Closest point on segment to the drone
        closest_pt   = g1 + long_clamped.unsqueeze(-1) * T_hat            # (N, 3)

        # --- CONTOURING ERROR e_c ---
        # Perpendicular distance from drone to the centerline segment.
        # In MPCC theory this is the component of error normal to the path.
        # Here we use the full lateral distance (includes vertical component).
        lateral_vec  = drone_pos_flat - closest_pt                        # (N, 3)
        e_c          = torch.norm(lateral_vec, dim=-1)                    # (N,)

        # --- LAG ERROR e_l ---
        # Remaining distance to the next gate along the current segment.
        # When the drone is at the gate center, e_l = 0.
        # This penalises the drone for not making forward progress.
        # Combined with Δlag (which rewards progress), these two terms together
        # act as a tight centred constraint on the ideal forward velocity.
        e_l = (seg_len_1d - long_clamped).clamp_min(0.0)                 # (N,)

        # --- GLOBAL ARC-LENGTH θ ---
        # θ = cumulative arc-length to segment start + local progress within segment.
        global_arc = (self.mpcc_arc_lengths[prev_idx] + long_clamped)    # (N,)

        # --- ARC-LENGTH PROGRESS Δlag ---
        # Change in θ since the previous step.
        delta_lag = global_arc - self.prev_arc_length                     # (N,)

        # Zero out Δlag for environments where the gate index just changed.
        # When gate_index advances, global_arc jumps forward discontinuously
        # (because prev_idx also changes). The resulting large Δlag would be
        # a spurious reward spike, not actual drone progress.
        delta_lag = torch.where(gate_index_changed, torch.zeros_like(delta_lag), delta_lag)

        return e_c, e_l, delta_lag, global_arc

    # =========================================================================
    # SCENE DESIGN (unchanged from original)
    # =========================================================================
    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )
        if self.controller is not None:
            self.controller = self.controller.to(self.device)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        )

        scale = torch.ones(3) * self.gate_scale
        gate_positions_list, gate_orientations_list = [], []
        gate_keys = sorted(self.track_config.keys(), key=lambda x: int(x))

        for i, gate_key in enumerate(gate_keys):
            gcfg     = self.track_config[gate_key]
            pos      = gcfg.get("pos", (0.0, 0.0, 1.0))
            yaw      = gcfg.get("yaw", 0.0)
            gate_pos = torch.tensor(pos, device=self.device, dtype=torch.float32)
            gate_yaw = yaw.item() if isinstance(yaw, torch.Tensor) and yaw.numel() == 1 else float(yaw)
            gate_ori = euler_to_quaternion(torch.tensor([0., 0., gate_yaw], device=self.device))
            gate_positions_list.append(gate_pos)
            gate_orientations_list.append(gate_ori)

            gate_prim = prim_utils.create_prim(
                f"/World/envs/env_0/Gate_{i}",
                usd_path=self.gate_asset_path,
                translation=(gate_pos[0].item(), gate_pos[1].item(), gate_pos[2].item()),
                orientation=(gate_ori[0].item(), gate_ori[1].item(),
                             gate_ori[2].item(), gate_ori[3].item()),
                scale=scale
            )
            gate_prim_path = f"/World/envs/env_0/Gate_{i}"
            kit_utils.set_nested_rigid_body_properties(
                gate_prim_path, disable_gravity=True,
                linear_damping=1000.0, angular_damping=1000.0,
            )
            gate_prim_obj = prim_utils.get_prim_at_path(gate_prim_path)
            all_prims = [gate_prim_obj]
            while len(all_prims) > 0:
                child_prim = all_prims.pop(0)
                if child_prim.HasAttribute("physics:kinematicEnabled"):
                    child_prim.GetAttribute("physics:kinematicEnabled").Set(True)
                all_prims += child_prim.GetChildren()

        first_gate_pos = gate_positions_list[0]
        first_gate_rot = gate_orientations_list[0]
        _offset_local = torch.tensor(
            [-1.5, 0.0, self.gate_height / 2.0],
            device=self.cfg.sim.device
        )
        offset_world = quat_rotate(
            first_gate_rot.unsqueeze(0), _offset_local.unsqueeze(0)
        ).squeeze(0)
        start_pos = first_gate_pos + offset_world
        self.drone.spawn(translations=[(start_pos[0].item(),
                                        start_pos[1].item(),
                                        start_pos[2].item())])
        return ["/World/defaultGroundPlane"]

    # =========================================================================
    # SPECS (unchanged from original — same obs/action/reward shape)
    # =========================================================================
    def _set_specs(self):
        # 18-dim drone state + 3 (next gate rpos local) + 3 (n2n gate) + 6 (gate rot) = 30
        robot_state_dim = 3 + 9 + 3   # lin_vel + rot_mat + ang_vel = 18
        observation_dim = robot_state_dim + 3 + 3 + 6

        self.observation_spec = Composite({
            "agents": {
                "observation": Unbounded((1, observation_dim), device=self.device),
                "intrinsics":  self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            },
            "info": {"drone_state": Unbounded((1, robot_state_dim), device=self.device)},
        }).expand(self.num_envs).to(self.device)

        self.action_spec = Composite({
            "agents": {"action": self.drone.action_spec.unsqueeze(0)}
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = Composite({
            "agents": {"reward": Unbounded((1, 1))}
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = Composite({
            "return":            Unbounded(1),
            "episode_len":       Unbounded(1),
            "gates_passed":      Unbounded(1),
            "drone_uprightness": Unbounded(1),
            "collision":         Unbounded(1),
            "crashed_z":         Unbounded(1),
            "crashed_oob":       Unbounded(1),
            "success":           BinaryDiscreteTensorSpec(1, dtype=bool),
            "truncated":         Unbounded(1),
            # MPCC-specific stats for WandB monitoring
            "mpcc_contouring_err": Unbounded(1),   # e_c (lateral drift, want small)
            "mpcc_lag_err":        Unbounded(1),   # e_l (lag behind, want small)
            "mpcc_arc_progress":   Unbounded(1),   # Δlag per step (want large)
            "mpcc_speed":          Unbounded(1),   # drone speed magnitude (want high)
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    # =========================================================================
    # EPISODE RESET
    # =========================================================================
    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments. Supports distributed initialization."""
        import traceback

        self.drone._reset_idx(env_ids)
        self.gate_passed[env_ids]     = False
        self.track_completed[env_ids] = False
        self.last_action[env_ids]     = 0.0
        self.effort[env_ids]          = 0.0

        gate_velocities = torch.zeros(
            len(env_ids), self.num_gates, 6, device=self.device
        )
        self.gates.set_velocities(gate_velocities, env_indices=env_ids)

        # --- Distributed initialization ---
        if self.distributed_init and getattr(self, 'training', True):
            # Place each resetting env near a random gate (excluding last which = first)
            start_gate_indices = torch.randint(
                0, self.num_gates - 1, (len(env_ids),),
                device=self.device, dtype=torch.long
            )
        else:
            start_gate_indices = torch.zeros(
                len(env_ids), device=self.device, dtype=torch.long
            )

        self.gate_indices[env_ids] = start_gate_indices

        drone_rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)

        try:
            gate_world_pos, gate_world_rot = self.gates.get_world_poses()
            gate_env_pos, gate_env_rot = self.get_env_poses(
                (gate_world_pos, gate_world_rot)
            )
            gate_env_pos = gate_env_pos[env_ids]
            gate_env_rot = gate_env_rot[env_ids]

            env_range      = torch.arange(len(env_ids), device=self.device)
            start_gate_pos = gate_env_pos[env_range, start_gate_indices]
            start_gate_rot = gate_env_rot[env_range, start_gate_indices]

            offset_local_exp = self.offset_local.unsqueeze(0).expand(len(env_ids), -1)
            offset_world     = quat_rotate(start_gate_rot, offset_local_exp)
            drone_start_pos  = start_gate_pos + offset_world

            self.drone.set_world_poses(
                drone_start_pos.unsqueeze(1) + self.envs_positions[env_ids].unsqueeze(1),
                drone_rot, env_ids
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("DroneRaceEnv MPCC _reset_idx failed") from e

        self.drone.set_velocities(
            torch.zeros(len(env_ids), 1, 6, device=self.device), env_ids
        )
        self.drone.set_joint_positions(
            torch.zeros(len(env_ids), 1, 4, device=self.device), env_ids
        )
        self.drone.set_joint_velocities(
            torch.zeros(len(env_ids), 1, 4, device=self.device), env_ids
        )

        # Initialize gate-crossing state and arc-length for the chosen start gate
        gc_offset_local = torch.tensor(
            [0.0, 0.0, self.gate_height / 2.0], device=self.device
        ).unsqueeze(0).expand(len(env_ids), -1)
        gc_offset_world   = quat_rotate(start_gate_rot, gc_offset_local)
        start_gate_center = start_gate_pos + gc_offset_world

        drone_to_gate = drone_start_pos - start_gate_center
        self.prev_drone_in_gate_frame[env_ids] = quat_rotate_inverse(
            start_gate_rot, drone_to_gate
        )

        # Initialize prev_arc_length to the arc-length of the start gate.
        # This ensures Δlag on step 1 is sensible (not a huge jump from 0).
        num_unique = self.num_gates - 1
        prev_start = (start_gate_indices - 1 + num_unique) % num_unique
        # Arc-length = cumulative to previous gate + distance from start drone pos
        seg = self.mpcc_gate_centers[start_gate_indices] - self.mpcc_gate_centers[prev_start]
        seg_len = torch.norm(seg, dim=-1).clamp_min(1e-6)
        T = seg / seg_len.unsqueeze(-1)
        dp_start = drone_start_pos - self.mpcc_gate_centers[prev_start]
        long_init = (dp_start * T).sum(-1).clamp(0.0, seg_len)
        self.prev_arc_length[env_ids] = self.mpcc_arc_lengths[prev_start] + long_init

        self.stats.exclude("success")[env_ids] = 0.
        self.stats["success"][env_ids] = False

    # =========================================================================
    # SIMULATION STEP HOOKS
    # =========================================================================
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")].clone()
        if self.controller is not None:
            root_state  = self.drone.get_state()[..., :13]
            raw_actions = self.controller.scaled_to_raw(actions)
            rotor_cmds  = self.controller(root_state, *raw_actions)
            self.drone.apply_action(rotor_cmds)
        else:
            raise Exception("No controller found.")

    def _post_sim_step(self, tensordict: TensorDictBase):
        """Save current action for smoothness penalty next step."""
        self.effort = tensordict[("agents", "action")].clone()

    # =========================================================================
    # STATE / OBSERVATION BUILDING
    # =========================================================================
    def _build_robot_state(self) -> torch.Tensor:
        """18-dim state: [lin_vel_world(3) | rot_mat(9) | ang_vel_world(3)]."""
        self.drone.get_state()
        lin_vel = self.drone.get_linear_velocity()   # (N, 1, 3) world frame
        rot_mat = self.drone.get_rotation_matrix()   # (N, 1, 9)
        ang_vel = self.drone.get_angular_velocity()  # (N, 1, 3) world frame
        return torch.cat([lin_vel, rot_mat, ang_vel], dim=-1)

    def get_relative_gate_position(
        self, gate_indices, gate_env_pos, gate_env_rot, drone_pos, drone_rot
    ):
        batch = torch.arange(self.num_envs, device=self.device)
        next_gate_pos = gate_env_pos[batch, gate_indices].unsqueeze(1)
        next_gate_rot = gate_env_rot[batch, gate_indices].unsqueeze(1)
        next_gate_rpos_world = next_gate_pos - drone_pos
        next_gate_rpos_local = quat_rotate_inverse(
            drone_rot.squeeze(1), next_gate_rpos_world.squeeze(1)
        ).unsqueeze(1)
        return next_gate_pos, next_gate_rot, next_gate_rpos_world, next_gate_rpos_local

    def get_next_to_next_gate_position(
        self, n2n_indices, gate_env_pos, gate_env_rot, next_gate_indices
    ):
        batch = torch.arange(self.num_envs, device=self.device)
        ng_pos  = gate_env_pos[batch, next_gate_indices]
        ng_rot  = gate_env_rot[batch, next_gate_indices]
        n2n_pos = gate_env_pos[batch, n2n_indices]
        return quat_rotate_inverse(ng_rot, n2n_pos - ng_pos)

    def _compute_state_and_obs(self):
        import traceback
        try:
            self.drone_state = self._build_robot_state()
            drone_pos = self.drone.pos
            drone_rot = self.drone.rot
            gate_world_pos, gate_world_rot = self.gates.get_world_poses()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("_compute_state_and_obs failed") from e

        gate_env_pos, gate_env_rot = self.get_env_poses(
            (gate_world_pos, gate_world_rot)
        )
        target_gate_indices = self.gate_indices

        (next_gate_pos, next_gate_rot,
         next_gate_rpos_world, next_gate_rpos_local) = self.get_relative_gate_position(
            target_gate_indices, gate_env_pos, gate_env_rot, drone_pos, drone_rot
        )

        n2n_indices = torch.where(
            target_gate_indices == self.num_gates - 1,
            target_gate_indices, target_gate_indices + 1,
        )
        n2n_gate_pos = self.get_next_to_next_gate_position(
            n2n_indices, gate_env_pos, gate_env_rot, target_gate_indices
        )

        next_gate_rot_flat = next_gate_rot.squeeze(1)
        e_x = torch.tensor([1., 0., 0.], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        e_y = torch.tensor([0., 1., 0.], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        gate_col0 = quat_rotate(next_gate_rot_flat, e_x)
        gate_col1 = quat_rotate(next_gate_rot_flat, e_y)
        next_gate_rot_2col = torch.cat([gate_col0, gate_col1], dim=-1).unsqueeze(1)

        obs = torch.cat([
            self.drone_state,                   # (N, 1, 18)
            next_gate_rpos_local,               # (N, 1, 3)
            n2n_gate_pos.unsqueeze(1),          # (N, 1, 3)
            next_gate_rot_2col,                 # (N, 1, 6)
        ], dim=-1)

        return TensorDict(
            {
                "agents": {"observation": obs, "intrinsics": self.drone.intrinsics},
                "info":   {"drone_state": self.drone_state},
                "stats":  self.stats.clone(),
            },
            self.batch_size,
        )

    # =========================================================================
    # GATE HELPERS
    # =========================================================================
    def _get_gate_center(self, gate_pos, gate_rot):
        offset = torch.tensor(
            [0.0, 0.0, self.gate_height / 2.0], device=self.device
        ).unsqueeze(0).expand(gate_pos.shape[0], -1)
        return gate_pos + quat_rotate(gate_rot, offset)

    def _detect_gate_crossings(
        self, drone_pos_flat, current_gate_center, current_gate_rot,
        gate_env_pos, gate_env_rot, batch_indices
    ):
        """Plane-crossing + bounding-box gate detection. Returns:
           gate_passed_this_step (N,), gate_index_changed (N,), new_gate_center (N,3).
        """
        drone_to_gate = drone_pos_flat - current_gate_center
        curr_in_gate  = quat_rotate_inverse(current_gate_rot, drone_to_gate)

        prev_x = self.prev_drone_in_gate_frame[..., 0]
        curr_x = curr_in_gate[..., 0]
        crossed_plane = (prev_x < 0) & (curr_x > 0)
        within_y = torch.abs(curr_in_gate[..., 1]) < (self.gate_width  / 2.0)
        within_z = torch.abs(curr_in_gate[..., 2]) < (self.gate_height / 2.0)
        gates_passed_ok = crossed_plane & within_y & within_z

        gate_passed_this_step = gates_passed_ok & (~self.gate_passed)
        self.gate_passed[gate_passed_this_step] = True

        old_gate_indices = self.gate_indices.clone()
        last_gate_passed = gate_passed_this_step & (self.gate_indices + 1 >= self.num_gates)
        self.track_completed[last_gate_passed] = True
        self.gate_indices[gate_passed_this_step] = torch.clamp(
            self.gate_indices[gate_passed_this_step] + 1, max=self.num_gates - 1,
        )
        gate_index_changed = (self.gate_indices != old_gate_indices)
        self.gate_passed[gate_index_changed] = False

        new_gate_pos    = gate_env_pos[batch_indices, self.gate_indices]
        new_gate_rot    = gate_env_rot[batch_indices, self.gate_indices]
        new_gate_center = self._get_gate_center(new_gate_pos, new_gate_rot)
        new_in_gate     = quat_rotate_inverse(new_gate_rot, drone_pos_flat - new_gate_center)
        self.prev_drone_in_gate_frame = torch.where(
            gate_index_changed.unsqueeze(-1), new_in_gate, curr_in_gate,
        )

        return gate_passed_this_step, gate_index_changed, new_gate_center

    # =========================================================================
    # REWARD AND DONE — CORE MPCC-BASED REWARD
    # =========================================================================
    def _compute_reward_and_done(self):
        """
        Compute MPCC-derived reward and termination.

        ====================================================================
        REWARD FORMULA (Team 9 Week 2 Presentation, Slide 5):
        ====================================================================
          r = Δlag  · w_prog          STAGE 1+  arc-length progress
            + up    · w_up            STAGE 1   uprightness (stability)
            − w_time                  STAGE 1+  time penalty
            − crash · w_crash         STAGE 1+  crash penalty
            − e_c²  · w_cont          STAGE 2+  contouring error (lateral drift)
            + gate_bonus              STAGE 2+  flat gate passage reward
            + center· w_center        STAGE 2+  gate center quality bonus
            + speed · w_spd           STAGE 2+  speed reward
            − e_l²  · w_lag           STAGE 3   lag error (behind reference)
            − ‖Δa‖ · w_sm            STAGE 3   action smoothness penalty
            + done  · w_lap           STAGE 3   lap completion bonus

        Where:
          Δlag  = arc-length progress this step (see _compute_mpcc_errors)
          e_c   = perpendicular distance from drone to racing line (metres)
          e_l   = remaining distance to next gate along current segment (metres)
          up    = drone's z-axis z-component in world frame (1=level, -1=flipped)
          speed = ‖drone velocity‖ in m/s
          Δa    = L2 norm of policy action change since last step
          crash = bool indicator for any crash type
          done  = bool indicator for lap completion
        ====================================================================
        """
        import traceback

        try:
            drone_pos = self.drone.pos   # (N, 1, 3)
            drone_rot = self.drone.rot   # (N, 1, 4)
            gate_world_pos, gate_world_rot = self.gates.get_world_poses()
            gate_env_pos, gate_env_rot = self.get_env_poses(
                (gate_world_pos, gate_world_rot)
            )
            batch_indices    = torch.arange(self.num_envs, device=self.device)
            current_gate_pos = gate_env_pos[batch_indices, self.gate_indices]
        except Exception as e:
            traceback.print_exc(); raise

        current_gate_rot    = gate_env_rot[batch_indices, self.gate_indices]
        current_gate_center = self._get_gate_center(current_gate_pos, current_gate_rot)
        drone_pos_flat = drone_pos.squeeze(1)   # (N, 3)
        drone_rot_flat = drone_rot.squeeze(1)   # (N, 4)

        # Detect gate crossings — updates gate_indices, track_completed
        gate_passed_this_step, gate_index_changed, new_gate_center = (
            self._detect_gate_crossings(
                drone_pos_flat, current_gate_center, current_gate_rot,
                gate_env_pos, gate_env_rot, batch_indices,
            )
        )

        # ------------------------------------------------------------------
        # SHARED TENSORS
        # ------------------------------------------------------------------
        # Drone z-axis in world frame (uprightness signal)
        drone_up = quat_axis(drone_rot_flat, axis=2)                      # (N, 3)

        # World-frame velocity (confirmed world frame by multirotor.py line 316)
        drone_vel_world = self.drone.vel[..., :3].squeeze(1)              # (N, 3)
        speed           = torch.norm(drone_vel_world, dim=-1)             # (N,)

        # Body-frame angular velocity (for stability checks — unused in reward here)
        ang_vel_world = self.drone.vel[..., 3:].squeeze(1)                # (N, 3)

        # ------------------------------------------------------------------
        # MPCC ERRORS (arc-length projection)
        # ------------------------------------------------------------------
        e_c, e_l, delta_lag, global_arc = self._compute_mpcc_errors(
            drone_pos_flat, gate_index_changed
        )

        # ------------------------------------------------------------------
        # CRASH DETECTION
        # ------------------------------------------------------------------
        ground_crash   = drone_pos_flat[:, 2] < 0.15
        bounds_crash   = (
            (drone_pos_flat[:, 0] < self._oob_x_min)
            | (drone_pos_flat[:, 0] > self._oob_x_max)
            | (drone_pos_flat[:, 1] < self._oob_y_min)
            | (drone_pos_flat[:, 1] > self._oob_y_max)
            | (drone_pos_flat[:, 2] > self._oob_z_max)
        )
        flipped_crash  = drone_up[:, 2] < -0.1
        crashed = ground_crash | bounds_crash | flipped_crash

        # ==================================================================
        # REWARD COMPUTATION
        # ==================================================================
        reward = torch.zeros(self.num_envs, device=self.device)

        # ================================================================
        # STAGE 1: STABILITY / PROGRESS  (always active)
        # ================================================================
        # These terms are active for ALL stages. They teach basic flight
        # stability and forward motion before introducing MPCC-specific terms.

        # --- Δlag · w_prog: ARC-LENGTH PROGRESS REWARD ---
        # This is the RL equivalent of the MPCC "maximise progress" objective.
        # Δlag is positive when the drone moves toward the next gate along the
        # centerline. Zero or negative when stationary or moving backward.
        # This is more precise than "distance to gate" because it measures
        # progress along the ACTUAL RACING LINE, not just Euclidean distance.
        r_prog   = delta_lag * self.mpcc_w_prog
        reward  += r_prog

        # --- up · w_up: UPRIGHTNESS REWARD ---
        # Encourages keeping z-axis vertical. Important in Stage 1 when the
        # drone is learning basic attitude control. Reduce in later stages.
        r_up     = drone_up[:, 2] * self.mpcc_w_up
        reward  += r_up

        # --- -w_time: TIME PENALTY ---
        # Constant per-step cost. Encourages the drone to complete the lap
        # quickly rather than hovering. Equivalent to the time-optimal
        # component of the MPCC cost function.
        reward  -= self.mpcc_w_time

        # --- -crash · w_crash: CRASH PENALTY ---
        # Not part of MPCC theory, but necessary for RL exploration safety.
        r_crash  = crashed.float() * (-self.mpcc_w_crash)
        reward  += r_crash

        # ================================================================
        # STAGE 2: NAVIGATION / CONTOURING  (training_stage >= 2)
        # ================================================================
        if self.training_stage >= 2:

            # --- -e_c² · w_cont: CONTOURING ERROR PENALTY ---
            # THE CORE MPCC TERM. Penalises quadratic lateral drift from the
            # racing line (piecewise linear centerline).
            # In MPCC theory: this term in the cost function forces the solver
            # to keep the vehicle on the optimal racing line.
            # In RL: this shapes the policy to fly ALONG the centerline rather
            # than cutting corners, which is optimal for this piecewise linear
            # representation of the track.
            #
            # TUNING:
            #   Too high → drone flies timidly close to centerline, slow speeds.
            #   Too low  → drone cuts corners, misses gates.
            #   Good range: 0.1 – 0.5 for this track scale.
            r_cont   = -(e_c ** 2) * self.mpcc_w_cont
            reward  += r_cont

            # --- gate_bonus: GATE PASSAGE FLAT REWARD ---
            # Not in original MPCC but critical for sparse gate-passing signal.
            r_gate   = gate_passed_this_step.float() * self.mpcc_gate_bonus
            reward  += r_gate

            # --- centering · w_center: GATE CENTER QUALITY BONUS ---
            # TOGT (Time-Optimal Gate Traversal) insight from the slide:
            # passing through the gate center minimises drag and maximises exit
            # speed. We reward proximity to the gate center at crossing time.
            # centering score: 1.0 at center, 0.0 at edge.
            drone_in_gate = quat_rotate_inverse(
                current_gate_rot, drone_pos_flat - current_gate_center
            )  # (N, 3) in gate-local frame
            y_frac      = drone_in_gate[:, 1].abs() / (self.gate_width  / 2.0 + 1e-6)
            z_frac      = drone_in_gate[:, 2].abs() / (self.gate_height / 2.0 + 1e-6)
            center_frac = torch.clamp(1.0 - torch.max(y_frac, z_frac), min=0.0)
            r_center    = gate_passed_this_step.float() * center_frac * self.mpcc_w_center
            reward     += r_center

            # --- speed · w_spd: SPEED REWARD ---
            # Rewards maintaining high velocity. In MPCC the progress rate is
            # naturally speed-dependent (faster = more Δlag). Here we add an
            # explicit speed term to prevent the drone from slowing down when
            # near the centerline but far from a gate.
            r_spd    = speed * self.mpcc_w_spd
            reward  += r_spd

        # ================================================================
        # STAGE 3: FULL MPCC REWARD  (training_stage >= 3)
        # ================================================================
        if self.training_stage >= 3:

            # --- -e_l² · w_lag: LAG ERROR PENALTY ---
            # Penalises the drone for falling behind the reference progress
            # along the current segment. e_l = remaining distance to the next
            # gate center along the segment tangent direction.
            # When e_l = 0, the drone is at the gate center (optimal).
            # As e_l grows, the drone is falling behind — the quadratic penalty
            # strongly discourages lingering in the middle of a segment.
            #
            # Combined with r_prog (which rewards moving toward the gate),
            # these two terms form a complementary pair:
            #   r_prog rewards making FORWARD progress each step.
            #   r_lag  penalises BEING FAR from the next gate.
            # Together they incentivise both speed and position accuracy.
            r_lag    = -(e_l ** 2) * self.mpcc_w_lag
            reward  += r_lag

            # --- -‖Δa‖ · w_sm: ACTION SMOOTHNESS PENALTY ---
            # Penalises large changes in policy output between consecutive steps.
            # In MPCC theory: the control cost term u^T R u penalises large
            # actuator commands; ‖Δa‖ captures the change rather than magnitude.
            # Smoother control → more stable flight → less energy waste → faster laps.
            # FIXED: uses self.effort (policy output), NOT propeller RPMs.
            action_diff = torch.norm(
                (self.effort - self.last_action).squeeze(1), dim=-1
            )   # (N,)
            r_sm     = -action_diff * self.mpcc_w_sm
            reward  += r_sm

            # --- done · w_lap: LAP COMPLETION BONUS ---
            # Large one-time reward for completing a full lap (passing the last gate).
            # In MPCC this is analogous to a terminal cost or a task reward.
            r_lap    = self.track_completed.float() * self.mpcc_w_lap
            reward  += r_lap

        # ------------------------------------------------------------------
        # STATE UPDATES
        # ------------------------------------------------------------------
        # Update arc-length for next step's Δlag calculation
        self.prev_arc_length = global_arc.detach()

        # Update action history for smoothness penalty
        self.last_action = self.effort.clone()

        # ------------------------------------------------------------------
        # TERMINATION
        # ------------------------------------------------------------------
        truncated      = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        completed_task = self.track_completed
        done           = truncated | completed_task.unsqueeze(-1) | crashed.unsqueeze(-1)

        # ------------------------------------------------------------------
        # STATS
        # ------------------------------------------------------------------
        self.stats["truncated"].add_(truncated.float())
        self.stats["collision"].add_(crashed.float().unsqueeze(-1))
        self.stats["crashed_z"].add_(ground_crash.float().unsqueeze(-1))
        self.stats["crashed_oob"].add_(bounds_crash.float().unsqueeze(-1))
        self.stats["success"].bitwise_or_(completed_task.unsqueeze(-1))
        self.stats["return"].add_(reward.unsqueeze(-1))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["drone_uprightness"].lerp_(drone_up[:, 2].unsqueeze(-1), 1 - self.alpha)
        self.stats["gates_passed"][:] = (
            self.gate_indices + self.track_completed.long()
        ).float().unsqueeze(1)
        # MPCC-specific stats — EMA-smoothed for stable WandB curves
        self.stats["mpcc_contouring_err"].lerp_(e_c.unsqueeze(-1),      1 - self.alpha)
        self.stats["mpcc_lag_err"].lerp_(e_l.unsqueeze(-1),             1 - self.alpha)
        self.stats["mpcc_arc_progress"].lerp_(delta_lag.unsqueeze(-1),  1 - self.alpha)
        self.stats["mpcc_speed"].lerp_(speed.unsqueeze(-1),             1 - self.alpha)

        return TensorDict(
            {
                "agents":     {"reward": reward.unsqueeze(-1).unsqueeze(-1)},
                "done":       done,
                "terminated": crashed.unsqueeze(-1),
                "truncated":  truncated,
                "stats":      self.stats.clone(),
            },
            self.batch_size,
        )

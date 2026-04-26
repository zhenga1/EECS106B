# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    Unbounded,
    Composite,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
)

import isaacsim.core.utils.prims as prim_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import (
    euler_to_quaternion,
    quat_rotate,
    quat_rotate_inverse,
    quat_axis,
)
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView, XFormPrimView

from omni_drones.robots import ASSET_PATH

# from omni_drones.utils.global_trajectory_planner import (
#     generate_trajectory_from_gate_poses,
# )  # [KY] for generating global trajectory — replaced by gate-to-gate line segments

from pxr import UsdPhysics

# Debug visualization is provided by IsaacEnv.debug_draw (always available).


class DroneRaceEnv(IsaacEnv):
    r"""
    A drone racing task where the agent must navigate through a sequence of gates
    in a racing track. The gates are arranged in a track pattern and the agent
    must pass through them in order.

    ## Observation

    - `drone_state` (15): how t state vector `[lin_vel(3) | rot_mat_flat(9) | ang_vel(3)]`. Note that
      position is **not** included.
    - `next_gate_rpos` (3): The relative position of the next gate to the drone in the drone's local frame.
    - `next_to_next_gate_pos` (3): The position of the gate after the immediate next gate, expressed in
      the next gate's local frame. Clamped at the last gate (no wrap-around).
    - `next_gate_rot_mat_2col` (6): The first two columns of the next gate's rotation matrix in the world frame
      (i.e. the gate's local x- and y-axes expressed in world coordinates), flattened to a 6-vector.

    ## Reward  *(student implementation required)*

    **Your task is to implement the reward function in `_compute_reward_and_done`.**

    ## Episode End

    The episode ends when the drone crashes (physical contact or off-course),
    completes the full lap, or the maximum episode length is reached.
    **You should also implement the crash/termination condition** in
    `_compute_reward_and_done` (currently returns all-zeros as a placeholder).

    ## Config

    | Parameter               | Type  | Default       | Description                                                                                                                                                                                                                             |
    | ----------------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`           | str   | "Hummingbird" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `track_config`          | dict  | None          | Optional dictionary defining gate positions and orientations. If provided, gates are placed according to this config. Format: `{"1": {"pos": (x, y, z), "yaw": angle}, ...}`. If None, uses circular track.                            |
    | `num_gates`             | int   | 8             | Number of gates in the racing track (only used if `track_config` is None).                                                                                                                                                              |
    | `track_radius`          | float | 5.0           | Radius of the circular track (only used if `track_config` is None).                                                                                                                                                                     |
    | `gate_spacing`          | float | 3.0           | Spacing between gates along the track (only used if `track_config` is None).                                                                                                                                                           |
    | `gate_height`           | float | 2.0           | Height of the gates (only used if `track_config` is None).                                                                                                                                                                              |
    | `gate_scale`            | float | 1.0           | Scale of the gate assets.                                                                                                                                                                                                              |
    | `gate_asset_path`       | str   | None          | Path to the gate USD asset. Defaults to `ASSET_PATH/gate/gate.usd` (isaac_drone_racer style). Can be overridden in config.                                                                                                        |
    """

    def __init__(self, cfg, headless):
        # -----------------------------------------------------------------------
        # STUDENT TODO (1/3): Load your reward scaling hyperparameters.
        #
        # Each line reads one value from cfg/task/DroneRace.yaml.
        # The second argument to .get() is the default used if the key is absent.
        # Add matching entries in cfg/task/DroneRace.yaml for each one you add.
        #
        # Example:
        #   self.reward_progress_scale = cfg.task.get("reward_progress_scale", 1.0)
        #   self.reward_gate_passage   = cfg.task.get("reward_gate_passage", 10.0)
        #   self.reward_crash_scale    = cfg.task.get("reward_crash_scale", 5.0)
        # -----------------------------------------------------------------------
        # ----- ADD YOUR REWARD CONFIG LINES BELOW (replace / extend the example) -----

        # ── MPCC-style reward hyperparameters ──────────────────────────────
        # Derived from MPCC (Romero et al., TRO 2022), MPCC++ (Krinner et al.
        # 2024), Swift (Nature 2023), MonoRace (2026), Actor-Critic MPC
        # (Romero et al., TRO 2025), and MPPI reference-free cost (2024).
        #
        # Core idea: decompose the drone's tracking error into *lag* (along
        # the gate-to-gate racing line) and *contouring* (perpendicular to it).
        # Reward = maximise forward progress (Δlag) − penalise contouring².
        # This is the RL-reward analogue of MPCC's cost:
        #   J_MPCC = q_c·e_c² + q_l·e_l² − μ_v·Δθ
        #

        # Additional terms from Swift (gate bonus, smoothness, crash) and
        # MPPI (speed bonus, gate-centering reward at crossing time).
        self.w_progress = cfg.task.get(
            "w_progress", 4.0
        )  # μ_v: reward per metre of Δlag (forward progress along racing line)
        self.w_contouring = cfg.task.get(
            "w_contouring", 0.4
        )  # q_c: penalty on contouring error² (perpendicular deviation)
        # self.w_lag = cfg.task.get("w_lag", 0.05)  # pruned: redundant with progress + contouring
        self.w_gate = cfg.task.get(
            "w_gate", 10.0
        )  # one-time bonus for passing through a gate
        self.w_centering = cfg.task.get(
            "w_centering", 3.0
        )  # bonus scaled by how centred the gate crossing is (TOGT insight)
        self.w_speed = cfg.task.get(
            "w_speed", 0.04
        )  # reward for speed magnitude (MonoRace / MPPI encouragement)
        self.w_time = cfg.task.get(
            "w_time", 0.01
        )  # per-step cost to encourage speed (Swift)
        self.w_crash = cfg.task.get("w_crash", 10.0)  # flat crash penalty
        self.w_bypass = cfg.task.get(
            "w_bypass", 15.0
        )  # one-time penalty for bypassing a gate
        self.w_smooth = cfg.task.get(
            "w_smooth", 0.001
        )  # action-rate penalty (Swift r_cmd)
        self.w_completion = cfg.task.get("w_completion", 50.0)  # full-lap bonus
        self.w_approach = cfg.task.get(
            "w_approach", 2.0
        )  # [KY, Claude] bootstrap signal: reward per metre of approach toward current gate
        # ── Basic flight bootstrap rewards ─────────────────────────────────────
        # An untrained policy outputs near-zero thrust → drone falls under gravity
        # → all episodes crash in ~50 steps with identical returns → return variance
        # near zero → explained_var = 1 − MSE/Var blows up to −∞.  These dense
        # per-step rewards give PPO a non-zero gradient from step 1.
        # References: Swift (alive/crash terms), MPCC++ (upright), deep-RL-racing.
        # self.w_alive = cfg.task.get("w_alive", 0.5)  # pruned: covered by crash_penalty + stall
        self.w_upright = cfg.task.get(
            "w_upright", 1.0
        )  # bonus for keeping z-axis pointing up
        self.w_altitude = cfg.task.get(
            "w_altitude", 0.5
        )  # exp bonus for flying at gate height
        self.w_ang_rate = cfg.task.get("w_ang_rate", 0.015)
        self.w_yaw_rate = cfg.task.get("w_yaw_rate", 0.005)
        self.w_alignment = cfg.task.get(
            "w_alignment", 0.0
        )  # reward for velocity direction aligned with path tangent (cosine similarity)
        self.w_stall = cfg.task.get(
            "w_stall", 30.0
        )  # one-time penalty when stall fires
        self.stall_patience = int(
            cfg.task.get("stall_patience", 200)
        )  # steps before stall fires
        self.progress_k = cfg.task.get(
            "progress_k", 5
        )  # steps to accumulate for MPCC reward
        self.w_stall_step = cfg.task.get(
            "w_stall_step", 0.1
        )  # per-step stall penalty leading up to stall firing
        # Legacy aliases so the crash section and other code still works
        self.reward_crash_scale = self.w_crash
        # ----- END STUDENT CODE -----

        self.gate_scale = cfg.task.gate_scale

        # Gate asset path - default to isaac_drone_racer gate asset
        # User can override this in config: gate_asset_path: "path/to/gate.usd"
        # If not specified, defaults to gate/gate.usd (isaac_drone_racer style)
        # Gate asset is in ASSET_PATH/gate/gate.usd
        self.gate_asset_path = cfg.task.get(
            "gate_asset_path", ASSET_PATH + "/gate/gate_a2rl.usd"
        )

        # Track configuration: support both config-based and circular track
        self.track_config = cfg.task.get("track_config", None)
        if self.track_config is not None:
            # Config-based track: gates defined with positions and yaw angles
            self.num_gates = len(self.track_config)
            self.track_type = "config"
            # For config-based tracks, use gate_height from config or default
            self.gate_height = cfg.task.get("gate_height", 2.0)
        else:
            # Circular track (backward compatibility)
            self.num_gates = int(cfg.task.num_gates)
            self.track_radius = cfg.task.track_radius
            self.gate_spacing = cfg.task.gate_spacing
            self.gate_height = cfg.task.gate_height
            self.track_type = "circular"

        import traceback
        import sys

        print(f"[DroneRaceEnv] Initializing, num_gates={self.num_gates}")
        try:
            super().__init__(cfg, headless)
            print(f"[DroneRaceEnv] super().__init__ completed")
        except Exception as e:
            print("=" * 80)
            print("ERROR: Failed in super().__init__")
            print("=" * 80)
            traceback.print_exc()
            print("=" * 80)
            raise

        try:
            self.drone.initialize(track_contact_forces=True)
            print(f"[DroneRaceEnv] drone.initialize() completed")
        except Exception as e:
            print("=" * 80)
            print("ERROR: Failed to initialize drone")
            print("=" * 80)
            traceback.print_exc()
            print("=" * 80)
            raise

        self.hover_cmd_thrust = None

        print(f"[DroneRaceEnv] num_envs={self.num_envs}, num_gates={self.num_gates}")

        # Initialize gates view here — must come before any self._get_gate_world_poses() call.
        # Gates are static (triangle mesh), so use XFormPrimView instead of RigidPrimView.
        try:
            print(
                f"[DroneRaceEnv] Creating XFormPrimView with pattern='/World/envs/env_*/Gate_*', shape=[{self.num_envs}, {self.num_gates}]"
            )
            self.gates = XFormPrimView(
                "/World/envs/env_*/Gate_*",
                reset_xform_properties=False,
            )
            print(f"[DroneRaceEnv] XFormPrimView created, calling initialize()...")
            self.gates.initialize()
            print(f"[DroneRaceEnv] gates.initialize() completed")
        except Exception as e:
            print("=" * 80)
            print(
                f"ERROR: Failed to initialize gates view with num_envs={self.num_envs}, num_gates={self.num_gates}"
            )
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            sys.stderr.write("=" * 80 + "\n")
            sys.stderr.write(f"ERROR: Failed to initialize gates view\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("=" * 80 + "\n")
            raise

        # [KY] Build gate centers in env frame for line-segment MPCC decomposition.
        gate_world_pos, gate_world_rot = self._get_gate_world_poses()
        gate_env_pos, gate_env_rot = self.get_env_poses(
            (gate_world_pos, gate_world_rot)
        )

        # Convert gate origins (bottom-center) to gate centers before planning.
        gate_center_offset_local = torch.tensor(
            [0.0, 0.0, self.gate_height / 2.0],
            device=self.device,
            dtype=gate_env_pos.dtype,
        )
        gate_center_offset = gate_center_offset_local.view(1, 1, 3).expand(
            self.num_envs, self.num_gates, 3
        )
        gate_center_offset_world = quat_rotate(
            gate_env_rot.reshape(-1, 4), gate_center_offset.reshape(-1, 3)
        ).reshape(self.num_envs, self.num_gates, 3)
        gate_env_centers = gate_env_pos + gate_center_offset_world

        # trajectories = []
        # for env_idx in range(num_envs):
        #     positions = gate_env_pos[env_idx]         # (num_gates, 3)
        #     orientations = gate_env_rot[env_idx]      # (num_gates, 4)
        #     traj = generate_trajectory_from_gate_poses(
        #         positions,
        #         orientations,
        #         method="catmull_rom",    # or "linear"
        #         num_points=200
        #     )
        #     trajectories.append(traj)  # Each traj: (num_points, 3)
        # trajectories = torch.stack(trajectories)  # (num_envs, num_points, 3)
        # self.trajectories = trajectories  # Store the full trajectories if needed for visualization or analysis

        # Store gate centers for gate-to-gate line-segment MPCC decomposition.
        # Gates are static so this is valid for the whole training run.
        self.gate_env_centers = gate_env_centers  # (num_envs, num_gates, 3)
        # Cache env-frame gate poses so _compute_state_and_obs and
        # _compute_reward_and_done never call the expensive Isaac Sim
        # XFormPrimView.get_world_poses() at runtime (gates never move).
        self._gate_env_pos_cache = gate_env_pos  # (num_envs, num_gates, 3)
        self._gate_env_rot_cache = gate_env_rot  # (num_envs, num_gates, 4)
        print(
            f"[DroneRaceEnv] using gate-to-gate line segments for MPCC decomposition."
        )
        # [KY] END of trajectory setup

        # Track gate progress for each environment
        self.gate_indices = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.gate_passed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.gate_bypassed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.track_completed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        # Per-environment progress tracking for MPCC reward and trajectory following
        # Each environment tracks its own previous lag and distance to gate
        self.prev_distance_to_gate = torch.zeros(self.num_envs, device=self.device)
        self.prev_lag = torch.zeros(self.num_envs, device=self.device)
        self.k_recent_lags = torch.zeros(
            self.num_envs, 5, device=self.device
        )  # for computing delta_lag over k steps instead of 1 step
        self.stall_counter = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.start_gate_indices = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # Gate crossing detection: drone position in gate frame from previous step
        self.gate_width = cfg.task.get("gate_width", 1.0)
        self.prev_drone_in_gate_frame = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.last_action = torch.zeros(
            self.num_envs, 1, self.drone.action_spec.shape[-1], device=self.device
        )
        self.effort = torch.zeros(
            self.num_envs, 1, self.drone.action_spec.shape[-1], device=self.device
        )
        self.recent_squared_progress = torch.zeros(
            self.num_envs, self.progress_k, device=self.device
        )
        self.prev_drone_pos_flat = None

        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-1.0, -1.0, 1.5], device=self.device),
            torch.tensor([1.0, 1.0, 2.5], device=self.device),
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-0.2, -0.2, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 0.0], device=self.device) * torch.pi,
        )

        self.offset_local = torch.tensor(
            [-1.5, 0.0, self.gate_height / 2.0], device=self.device
        )
        self.alpha = 0.8

        # Debug visualization.
        # self.debug_draw is inherited from IsaacEnv and always available.
        self.debug_gate_origins = cfg.task.get("debug_gate_origins", False)
        self.axis_length = cfg.task.get("debug_axis_length", 0.3)

    def _draw_gate_origins(self, gate_world_pos, gate_world_rot):
        """Draw X/Y/Z axes for ALL gates in ALL envs — 3 batch draw_lines calls.

        Args:
            gate_world_pos: (E, G, 3) gate origins in world frame.
            gate_world_rot: (E, G, 4) gate quaternions in world frame.
        """
        if not self.debug_gate_origins:
            return

        E, G = gate_world_pos.shape[:2]
        L = self.axis_length

        # Local-frame axis directions scaled by axis_length: (3, 3)
        axis_dirs_local = torch.tensor(
            [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
            device=self.device,
            dtype=torch.float32,
        )

        # Flatten gate quats to (E*G, 4), tile axes to (E*G*3, 4) / (E*G*3, 3)
        gate_quat_flat = gate_world_rot.reshape(E * G, 4)
        gate_quat_2d = (
            gate_quat_flat.unsqueeze(1).expand(-1, 3, -1).reshape(E * G * 3, 4)
        )
        axis_dirs_2d = (
            axis_dirs_local.unsqueeze(0).expand(E * G, -1, -1).reshape(E * G * 3, 3)
        )

        # Single batched rotation → reshape to (E*G, 3, 3): dim 1 = axis index {X,Y,Z}
        axis_world = quat_rotate(gate_quat_2d, axis_dirs_2d).reshape(E * G, 3, 3)

        gate_origins_cpu = gate_world_pos.reshape(E * G, 3).cpu()
        axis_world_cpu = axis_world.cpu()
        n = gate_origins_cpu.shape[0]

        starts_list = [tuple(p) for p in gate_origins_cpu.tolist()]
        for axis_idx, color in enumerate(
            [
                (1.0, 0.0, 0.0, 1.0),  # X — red
                (0.0, 1.0, 0.0, 1.0),  # Y — green
                (0.0, 0.0, 1.0, 1.0),  # Z — blue
            ]
        ):
            ends = gate_origins_cpu + axis_world_cpu[:, axis_idx, :]  # (E*G, 3)
            ends_list = [tuple(p) for p in ends.tolist()]
            self.debug_draw._draw.draw_lines(
                starts_list, ends_list, [color] * n, [3.0] * n
            )

    def _draw_global_trajectory(self):
        """Draw the pre-computed global trajectory for ALL envs in one draw_lines call.

        Converts from env-local frame to world frame by broadcasting
        self.envs_positions (E, 3) across trajectory points (E, P, 3),
        then delegates to DebugDraw.plot_multi() for a single batched call.
        """
        if not self.debug_gate_origins:
            return

        # gate_env_centers: (E, G, 3) env-local frame — draw dots at gate centers
        traj_world = self.gate_env_centers + self.envs_positions.unsqueeze(1)
        self.debug_draw.plot_multi(traj_world, size=3.0, color=(0.0, 1.0, 1.0, 1.0))

    def _design_scene(self):
        print("Designing scene")
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )
        if self.controller is not None:
            self.controller = self.controller.to(self.device)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        # Create gates based on track configuration
        scale = torch.ones(3) * self.gate_scale
        gate_positions_list = []
        gate_orientations_list = []

        # Config-based track: gates defined with positions and yaw angles
        # Sort gate keys to ensure correct order
        gate_keys = sorted(self.track_config.keys(), key=lambda x: int(x))

        for i, gate_key in enumerate(gate_keys):
            gate_cfg = self.track_config[gate_key]
            pos = gate_cfg.get("pos", (0.0, 0.0, 1.0))
            yaw = gate_cfg.get("yaw", 0.0)

            # Convert to torch tensors
            if isinstance(pos, (list, tuple)):
                gate_pos = torch.tensor(pos, device=self.device, dtype=torch.float32)
            else:
                gate_pos = torch.tensor(
                    [pos[0], pos[1], pos[2]], device=self.device, dtype=torch.float32
                )

            if isinstance(yaw, torch.Tensor):
                gate_yaw = yaw.item() if yaw.numel() == 1 else yaw
            else:
                gate_yaw = float(yaw)

            # Create quaternion from yaw (rotation around z-axis)
            gate_orientation = euler_to_quaternion(
                torch.tensor([0.0, 0.0, gate_yaw], device=self.device)
            )

            gate_positions_list.append(gate_pos)
            gate_orientations_list.append(gate_orientation)

            # Spawn gate using configured gate asset
            gate_prim = prim_utils.create_prim(
                f"/World/envs/env_0/Gate_{i}",
                usd_path=self.gate_asset_path,
                translation=(
                    gate_pos[0].item(),
                    gate_pos[1].item(),
                    gate_pos[2].item(),
                ),
                orientation=(
                    gate_orientation[0].item(),
                    gate_orientation[1].item(),
                    gate_orientation[2].item(),
                    gate_orientation[3].item(),
                ),
                scale=scale,
            )
            # Make gate static: disable gravity and make kinematic
            gate_prim_path = f"/World/envs/env_0/Gate_{i}"
            kit_utils.set_nested_rigid_body_properties(
                gate_prim_path,
                disable_gravity=True,
                linear_damping=1000.0,  # Very high damping to prevent movement
                angular_damping=1000.0,
            )
            # Set kinematic on all nested rigid bodies to prevent movement from collisions
            gate_prim_obj = prim_utils.get_prim_at_path(gate_prim_path)
            all_prims = [gate_prim_obj]
            while len(all_prims) > 0:
                child_prim = all_prims.pop(0)
                if child_prim.HasAttribute("physics:kinematicEnabled"):
                    child_prim.GetAttribute("physics:kinematicEnabled").Set(True)
                all_prims += child_prim.GetChildren()

        # Gate positions and orientations will be retrieved from views at runtime
        # No need to store them manually since gates are static

        # Spawn drone at start position (behind first gate, in gate's local frame)
        # The gate's local x-axis points in the tangent direction (for circular) or forward (for config)
        # We want to spawn the drone behind the gate, so we offset backward along the gate's x-axis
        first_gate_pos = gate_positions_list[0]
        first_gate_rot = gate_orientations_list[0]
        # Offset backward in gate's local frame (negative x direction)
        offset_local = torch.tensor(
            [-1.5, 0.0, self.gate_height / 2.0], device=self.device
        )
        # Rotate offset to world frame - quat_rotate expects batched inputs, so add batch dimension
        offset_world = quat_rotate(
            first_gate_rot.unsqueeze(0), offset_local.unsqueeze(0)
        ).squeeze(0)
        start_pos = first_gate_pos + offset_world
        # Store first gate position as landing target after completing all laps
        self.first_gate_pos = first_gate_pos
        self.first_gate_rot = first_gate_rot
        self.drone.spawn(
            translations=[
                (start_pos[0].item(), start_pos[1].item(), start_pos[2].item())
            ]
        )

        return ["/World/defaultGroundPlane"]

    def _get_gate_world_poses(self):
        pos, rot = self.gates.get_world_poses()
        return (
            pos.reshape(self.num_envs, self.num_gates, 3),
            rot.reshape(self.num_envs, self.num_gates, 4),
        )

    def _set_specs(self):
        print(f"[_set_specs DEBUG] Loading from: {__file__}")
        # Custom robot state: linear_vel(3) + rotation_matrix_flat(9) + angular_vel(3) = 18
        robot_state_dim = 3 + 9 + 3  # 15
        # Observation: robot_state(15) + next_gate_rpos_local(3) + next_to_next_gate_pos(3)
        #              + next_gate_rot_mat_2col(6)
        observation_dim = robot_state_dim + 3 + 3 + 6
        print(f"[_set_specs DEBUG] observation_dim = {observation_dim}")
        print("The observation vector consists of:")
        print(f"  - Robot state: {robot_state_dim}")
        print(f"  - Next gate relative position: 3")
        print(f"  - Next to next gate position: 3")
        self.observation_spec = (
            Composite(
                {
                    "agents": {
                        "observation": Unbounded(
                            (1, observation_dim), device=self.device
                        ),
                        "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(
                            self.device
                        ),
                    },
                    "info": {
                        "drone_state": Unbounded(
                            (1, robot_state_dim), device=self.device
                        ),
                    },
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.action_spec = (
            Composite(
                {
                    "agents": {
                        "action": self.drone.action_spec.unsqueeze(0),
                    }
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.reward_spec = (
            Composite({"agents": {"reward": Unbounded((1, 1))}})
            .expand(self.num_envs)
            .to(self.device)
        )
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )
        stats_spec = (
            Composite(
                {
                    "return": Unbounded(1),
                    "episode_len": Unbounded(1),
                    "gates_passed": Unbounded(1),
                    "drone_uprightness": Unbounded(1),
                    "collision": Unbounded(1),
                    "success": BinaryDiscreteTensorSpec(1, dtype=bool),
                    "truncated": Unbounded(1),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        # [CK] Randomise spawn gate: pick any gate except the last (which duplicates gate 0).
        # Excludes num_gates-1 so the drone never spawns at the finish/start overlap gate.
        spawn_gate_idx = torch.randint(
            0, self.num_gates - 1, (len(env_ids),), dtype=torch.long, device=self.device
        )
        self.start_gate_indices[env_ids] = spawn_gate_idx
        self.gate_indices[env_ids] = spawn_gate_idx

        # Reset gate progress
        self.gate_passed[env_ids] = False
        self.gate_bypassed[env_ids] = False
        self.track_completed[env_ids] = False
        self.last_action[env_ids] = 0.0
        self.effort[env_ids] = 0.0
        self.stall_counter[env_ids] = 0
        # [KY, Claude] prev_lag is initialized below after drone_start_pos is known; zero-init caused a spurious -3.0 reward spike on the first step of every episode

        try:
            # Use cached static gate poses (gates never move).
            gate_env_pos = self._gate_env_pos_cache[env_ids]   # (M, num_gates, 3)
            gate_env_rot = self._gate_env_rot_cache[env_ids]   # (M, num_gates, 4)

            local_ids = torch.arange(len(env_ids), device=self.device)
            spawn_gate_pos = gate_env_pos[local_ids, spawn_gate_idx]  # (M, 3)
            spawn_gate_rot = gate_env_rot[local_ids, spawn_gate_idx]  # (M, 4)

            # Place drone 1.5 m in front of the spawn gate (gate local -x = approach side).
            offset_local_expanded = self.offset_local.unsqueeze(0).expand(
                len(env_ids), -1
            )  # (M, 3)
            offset_world = quat_rotate(spawn_gate_rot, offset_local_expanded)  # (M, 3)
            drone_start_pos = spawn_gate_pos + offset_world  # (M, 3)

            # Compute spawn gate center (origin is bottom-center; add height/2 in gate-local z).
            _gc_offset = torch.tensor(
                [0.0, 0.0, self.gate_height / 2.0], device=self.device
            )
            _gc_offset_world = quat_rotate(
                spawn_gate_rot, _gc_offset.unsqueeze(0).expand(len(env_ids), -1)
            )
            spawn_gate_center = spawn_gate_pos + _gc_offset_world  # (M, 3)

            # [CK] Point drone toward the spawn gate center so it gets a strong reward
            # signal from step 1. Compute yaw = atan2(dy, dx) from drone to gate center.
            to_gate = spawn_gate_center - drone_start_pos  # (M, 3)
            spawn_yaw = torch.atan2(to_gate[:, 1], to_gate[:, 0])  # (M,)

            # Sample small random roll/pitch perturbation for naturalness, use computed yaw.
            drone_rpy_noise = self.init_rpy_dist.sample((*env_ids.shape, 1))  # (M, 1, 3)
            drone_rpy = drone_rpy_noise.clone()
            drone_rpy[:, 0, 2] = spawn_yaw  # override yaw channel
            drone_rot = euler_to_quaternion(drone_rpy)

            # Initialize prev_lag using spawn gate's forward axis as tangent.
            gate_forward_reset = quat_rotate(
                spawn_gate_rot,
                torch.tensor([1.0, 0.0, 0.0], device=self.device)
                .unsqueeze(0)
                .expand(len(env_ids), -1),
            )  # (M, 3)
            drone_to_spawn_gate = drone_start_pos - spawn_gate_center  # (M, 3)
            self.prev_lag[env_ids] = (
                (drone_to_spawn_gate * gate_forward_reset).sum(dim=-1).detach()
            )

            # Aliases for the code below that uses these names.
            first_gate_center = spawn_gate_center
            first_gate_rot = spawn_gate_rot

            drone_start_pos_with_agent = drone_start_pos.unsqueeze(1)  # (M, 1, 3)
            env_positions_with_agent = self.envs_positions[env_ids].unsqueeze(1)  # (M, 1, 3)

            self.drone.set_world_poses(
                drone_start_pos_with_agent + env_positions_with_agent,
                drone_rot,
                env_ids,
            )
        except Exception as e:
            import traceback

            print("=" * 80)
            print(
                f"ERROR: Failed in _reset_idx (num_envs={self.num_envs}, num_gates={self.num_gates})"
            )
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            raise RuntimeError(
                f"DroneRaceEnv._reset_idx failed (num_envs={self.num_envs}, num_gates={self.num_gates})"
            ) from e

        self.drone.set_velocities(
            torch.zeros(len(env_ids), 1, 6, device=self.device), env_ids
        )

        self.drone.set_joint_positions(
            torch.zeros(len(env_ids), 1, 4, device=self.device), env_ids
        )
        self.drone.set_joint_velocities(
            torch.zeros(len(env_ids), 1, 4, device=self.device), env_ids
        )

        # first_gate_center already computed inside the try block above.
        self.prev_distance_to_gate[env_ids] = torch.norm(
            first_gate_center - drone_start_pos, dim=-1
        )

        # Initialise drone position in gate frame for crossing detection
        drone_to_first_gate_center = (
            drone_start_pos - first_gate_center
        )  # (len(env_ids), 3)
        self.prev_drone_in_gate_frame[env_ids] = quat_rotate_inverse(
            first_gate_rot, drone_to_first_gate_center
        )  # (len(env_ids), 3)

        self.stats.exclude("success")[env_ids] = 0.0
        self.stats["success"][env_ids] = False

    def _pre_sim_step(self, tensordict: TensorDictBase):
        """
        Input actions are in scaled units
        """
        actions = tensordict[("agents", "action")].clone()
        if self.controller is not None:
            root_state = self.drone.get_state()[..., :13]
            raw_actions = self.controller.scaled_to_raw(actions)
            rotor_cmds = self.controller(root_state, *raw_actions)
            _ = self.drone.apply_action(rotor_cmds)
        else:
            raise Exception("No controller found. This is not yet supported.")

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.effort = tensordict[("agents", "action")].clone()

    def _build_robot_state(self) -> torch.Tensor:
        """Builds the custom robot state vector used for observations.

        Calls ``drone.get_state()`` to refresh all cached kinematics, then
        concatenates along the feature dimension:

            [position (3) | linear_velocity (3) | rotation_matrix_flat (9) | angular_velocity (3)]

        Returns:
            Tensor of shape (N, 1, 18).
        """
        self.drone.get_state()  # refresh pos, rot, vel_w, vel_b caches
        lin_vel = self.drone.get_linear_velocity()  # (N, 1, 3)
        rot_mat = self.drone.get_rotation_matrix()  # (N, 1, 9)
        ang_vel = self.drone.get_angular_velocity()  # (N, 1, 3)
        return torch.cat([lin_vel, rot_mat, ang_vel], dim=-1)  # (N, 1, 18)

    def get_relative_gate_position(
        self, gate_indices, gate_env_pos, gate_env_rot, drone_pos, drone_rot
    ):
        """Return gate position/rotation relative to each drone, both in world and drone-local frames.

        Args:
            gate_indices: (N,) integer tensor – the target gate index per environment.
            gate_env_pos: (N, num_gates, 3) gate positions in env frame.
            gate_env_rot: (N, num_gates, 4) gate rotations in env frame.
            drone_pos:    (N, 1, 3) drone positions.
            drone_rot:    (N, 1, 4) drone rotations.

        Returns:
            next_gate_pos:        (N, 1, 3) selected gate position in env frame.
            next_gate_rot:        (N, 1, 4) selected gate rotation in env frame.
            next_gate_rpos_world: (N, 1, 3) gate position relative to drone, in world frame.
            next_gate_rpos_local: (N, 1, 3) gate position relative to drone, in drone-local frame.
        """
        batch_indices = torch.arange(self.num_envs, device=self.device)

        next_gate_pos = gate_env_pos[batch_indices, gate_indices]  # (N, 3)
        next_gate_rot = gate_env_rot[batch_indices, gate_indices]  # (N, 4)

        # Expand to match agent dimension for broadcasting
        next_gate_pos = next_gate_pos.unsqueeze(1)  # (N, 1, 3)
        next_gate_rot = next_gate_rot.unsqueeze(1)  # (N, 1, 4)

        # Relative position in world frame
        next_gate_rpos_world = next_gate_pos - drone_pos  # (N, 1, 3)

        # Relative position in drone-local frame
        drone_rot_flat = drone_rot.squeeze(1)  # (N, 4)
        next_gate_rpos_world_flat = next_gate_rpos_world.squeeze(1)  # (N, 3)
        next_gate_rpos_local_flat = quat_rotate_inverse(
            drone_rot_flat, next_gate_rpos_world_flat
        )  # (N, 3)
        next_gate_rpos_local = next_gate_rpos_local_flat.unsqueeze(1)  # (N, 1, 3)

        return next_gate_pos, next_gate_rot, next_gate_rpos_world, next_gate_rpos_local

    def get_next_to_next_gate_position(
        self, next_to_next_gate_indices, gate_env_pos, gate_env_rot, next_gate_indices
    ):
        """Return the position of the next-to-next gate expressed in the next gate's local frame.

        Args:
            next_to_next_gate_indices: (N,) index of the gate after the immediate next gate (clamped at last gate).
            gate_env_pos:              (N, num_gates, 3) gate positions in env frame.
            gate_env_rot:              (N, num_gates, 4) gate rotations in env frame.
            next_gate_indices:         (N,) index of the immediate next gate per environment.

        Returns:
            (N, 3) position of the next-to-next gate relative to the next gate, in the next gate's local frame.
        """
        batch_indices = torch.arange(self.num_envs, device=self.device)

        # Positions/orientations of the immediate next gate in env frame
        next_gate_pos_env = gate_env_pos[batch_indices, next_gate_indices]  # (N, 3)
        next_gate_rot_env = gate_env_rot[batch_indices, next_gate_indices]  # (N, 4)

        # Position of the next-to-next gate in env frame
        n2n_gate_pos_env = gate_env_pos[
            batch_indices, next_to_next_gate_indices
        ]  # (N, 3)

        # Relative position in env frame
        rpos_env = n2n_gate_pos_env - next_gate_pos_env  # (N, 3)

        # Rotate into the next gate's local frame
        rpos_next_gate_frame = quat_rotate_inverse(
            next_gate_rot_env, rpos_env
        )  # (N, 3)

        return rpos_next_gate_frame

    def _compute_state_and_obs(self):
        import traceback
        import sys

        try:
            # Build custom state: [pos(3) | lin_vel(3) | rot_mat_flat(9) | ang_vel(3)] -> (N, 1, 18)
            # Also refreshes drone.pos / drone.rot caches used below for gate computations.
            self.drone_state = self._build_robot_state()
            drone_pos = self.drone.pos  # (N, 1, 3) refreshed by _build_robot_state
            drone_rot = self.drone.rot  # (N, 1, 4) refreshed by _build_robot_state

        except Exception as e:
            print("=" * 80)
            print(
                f"ERROR: Failed in _compute_state_and_obs (num_envs={self.num_envs}, num_gates={self.num_gates})"
            )
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            sys.stderr.write("=" * 80 + "\n")
            sys.stderr.write(f"ERROR: Failed in _compute_state_and_obs\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("=" * 80 + "\n")
            raise RuntimeError(
                f"DroneRaceEnv._compute_state_and_obs failed (num_envs={self.num_envs}, num_gates={self.num_gates})"
            ) from e

        gate_env_pos = self._gate_env_pos_cache  # (N, num_gates, 3) — static, cached at init
        gate_env_rot = self._gate_env_rot_cache  # (N, num_gates, 4)

        # If track is completed, target first gate for landing
        track_completed = self.track_completed  # (N,)

        # Get current gate positions for each environment
        target_gate_indices = self.gate_indices  # (N,)

        next_gate_pos, next_gate_rot, next_gate_rpos_world, next_gate_rpos_local = (
            self.get_relative_gate_position(
                target_gate_indices, gate_env_pos, gate_env_rot, drone_pos, drone_rot
            )
        )

        # Get next-to-next gate positions. If the immediate next gate is the last gate, clamp so
        # next-to-next points at the same gate (no wrap-around).
        next_to_next_gate_indices = torch.where(
            target_gate_indices == self.num_gates - 1,
            target_gate_indices,
            target_gate_indices + 1,
        )  # (N,)

        next_to_next_gate_pos = self.get_next_to_next_gate_position(
            next_to_next_gate_indices, gate_env_pos, gate_env_rot, target_gate_indices
        )  # (N, 3)

        # Angle between drone->gate vector and gate normal (gate local +x axis in env frame)
        gate_normal_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        gate_normal_local_expanded = gate_normal_local.unsqueeze(0).expand(
            self.num_envs, -1
        )  # (N, 3)
        gate_normal_world = quat_rotate(
            next_gate_rot.squeeze(1), gate_normal_local_expanded
        )  # (N, 3)
        gate_vec_world = next_gate_rpos_world.squeeze(1)  # (N, 3)
        gate_vec_norm = torch.norm(gate_vec_world, dim=-1).clamp_min(1e-6)
        gate_normal_norm = torch.norm(gate_normal_world, dim=-1).clamp_min(1e-6)
        cos_angle = (gate_vec_world * gate_normal_world).sum(-1) / (
            gate_vec_norm * gate_normal_norm
        )
        gate_angle = torch.acos(cos_angle.clamp(-1.0, 1.0))  # (N,)

        # Gate progress: fraction of gates completed in the single lap
        gate_progress = self.gate_indices.float() / self.num_gates  # (N,)
        gate_progress = torch.where(
            track_completed, torch.ones_like(gate_progress), gate_progress
        )

        # Next gate orientation: first 2 columns of its rotation matrix in the world frame.
        # Rotate the gate-local x- and y-axes into world coordinates via quat_rotate,
        # then concatenate to a 6-vector per environment.
        next_gate_rot_flat = next_gate_rot.squeeze(1)  # (N, 4)
        e_x = (
            torch.tensor([1.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .expand(self.num_envs, -1)
        )  # (N, 3)
        e_y = (
            torch.tensor([0.0, 1.0, 0.0], device=self.device)
            .unsqueeze(0)
            .expand(self.num_envs, -1)
        )  # (N, 3)
        gate_col0 = quat_rotate(
            next_gate_rot_flat, e_x
        )  # (N, 3) - gate x-axis in world frame
        gate_col1 = quat_rotate(
            next_gate_rot_flat, e_y
        )  # (N, 3) - gate y-axis in world frame
        next_gate_rot_mat_2col = torch.cat([gate_col0, gate_col1], dim=-1).unsqueeze(
            1
        )  # (N, 1, 6)

        # Build observation
        # All components need to have the agent dimension (middle dimension) to match spec (N, 1, obs_dim)
        obs = [
            self.drone_state,  # (N, 1, state_dim) - already has agent dimension
            next_gate_rpos_local,  # (N, 1, 3) - already has agent dimension
            next_to_next_gate_pos.unsqueeze(1),  # (N, 1, 3)
            next_gate_rot_mat_2col,  # (N, 1, 6)
        ]

        # Concatenate along last dimension: (N, 1, obs_dim)
        obs = torch.cat(obs, dim=-1)  # (N, 1, obs_dim)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "info": {
                    "drone_state": self.drone_state,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _get_gate_center(self, gate_pos, gate_rot):
        """Compute gate center from gate origin (bottom-centre) by adding height/2 offset in gate frame.

        Args:
            gate_pos: (N, 3) gate origin positions in env frame.
            gate_rot: (N, 4) gate orientations (quaternions) in env frame.

        Returns:
            gate_center: (N, 3) gate centre positions in env frame.
        """
        offset_local = torch.tensor(
            [0.0, 0.0, self.gate_height / 2.0], device=self.device
        )
        offset_local_expanded = offset_local.unsqueeze(0).expand(gate_pos.shape[0], -1)
        offset_world = quat_rotate(gate_rot, offset_local_expanded)
        return gate_pos + offset_world

    def _detect_gate_crossings(
        self,
        drone_pos_flat,
        current_gate_center,
        current_gate_rot,
        gate_env_pos,
        gate_env_rot,
        batch_indices,
    ):
        """Detect whether the drone crossed through the current target gate this step.

        Uses plane-crossing (x: negative -> positive) plus bounding-box check
        (|y|, |z| < gate_width/2) in the gate frame centred at gate_center.

        Args:
            drone_pos_flat: (N, 3) current drone positions in env/world frame.
            current_gate_center: (N, 3) centre position of each env's current target gate
                in env/world frame.
            current_gate_rot: (N, 4) quaternion orientation of each env's current target gate
                in env/world frame.
            gate_env_pos: (N, G, 3) all gate origin positions for each env.
            gate_env_rot: (N, G, 4) all gate orientations (quaternions) for each env.
            batch_indices: (N,) environment indices used to gather the active gate per env.

        Side-effects: updates gate_indices, gate_passed, track_completed,
        prev_drone_in_gate_frame.

        Returns:
            gate_passed_this_step: (N,) bool — True for envs that just passed a gate.
            gate_index_changed:    (N,) bool — True for envs whose target gate advanced.
            new_gate_center:       (N, 3) — centre of the (possibly new) target gate.
        """
        drone_to_gate = drone_pos_flat - current_gate_center
        curr_in_gate = quat_rotate_inverse(current_gate_rot, drone_to_gate)  # (N, 3)

        # Drone position
        prev_x = self.prev_drone_in_gate_frame[..., 0]
        curr_x = curr_in_gate[..., 0]
        crossed_plane = (prev_x < 0) & (curr_x > 0)

        # TODO: Build `gates_passed_successfully` as a per-environment binary mask.
        # Shape: (N,) where N == self.num_envs.
        # Type: bool is preferred (True/False); equivalent to 1/0 when cast to int/float.
        # Gate width and height can be accessed by using self.gate_width and self.gate_height.
        # ----- ADD YOUR GATE-CROSSING MASK CODE BELOW (replace the placeholder) -----

        # Gate crossing: drone crossed the gate plane (x: neg->pos) AND is within the
        # gate's bounding box in the gate-local y-z plane.
        within_y = torch.abs(curr_in_gate[..., 1]) < (self.gate_width / 2.0)
        within_z = torch.abs(curr_in_gate[..., 2]) < (self.gate_height / 2.0)
        gates_passed_successfully = crossed_plane & within_y & within_z

        # ----- END STUDENT CODE -----

        gate_passed_this_step = gates_passed_successfully & (~self.gate_passed)
        self.gate_passed[gate_passed_this_step] = True

        # Bypass: crossed the gate plane but outside the opening, and haven't
        # already passed or bypassed this gate in the current approach.
        bypassed_gate = crossed_plane & ~gates_passed_successfully
        gate_bypassed_this_step = (
            bypassed_gate & (~self.gate_passed) & (~self.gate_bypassed)
        )
        self.gate_bypassed[gate_bypassed_this_step] = True

        old_gate_indices = self.gate_indices.clone()
        last_gate_passed = gate_passed_this_step & (
            self.gate_indices + 1 >= self.num_gates
        )
        self.track_completed[last_gate_passed] = True
        self.gate_indices[gate_passed_this_step] = torch.clamp(
            self.gate_indices[gate_passed_this_step] + 1,
            max=self.num_gates - 1,
        )
        gate_index_changed = self.gate_indices != old_gate_indices
        self.gate_passed[gate_index_changed] = False
        self.gate_bypassed[gate_index_changed] = False  # reset bypass flag for new gate

        # Recompute gate centre for envs whose target changed
        new_gate_pos = gate_env_pos[batch_indices, self.gate_indices]
        new_gate_rot = gate_env_rot[batch_indices, self.gate_indices]
        new_gate_center = self._get_gate_center(new_gate_pos, new_gate_rot)
        new_in_gate = quat_rotate_inverse(
            new_gate_rot, drone_pos_flat - new_gate_center
        )

        # old_prev_in_gate_frame = self.prev_drone_in_gate_frame.clone()
        self.prev_drone_in_gate_frame = torch.where(
            gate_index_changed.unsqueeze(-1),
            new_in_gate,
            curr_in_gate,
        )

        return (
            gate_passed_this_step,
            gate_index_changed,
            new_gate_center,
            gate_bypassed_this_step,
        )

    def _compute_reward_and_done(self):
        import traceback
        import sys

        track_completed = self.track_completed  # (N,)

        try:
            # Use cached pose tensors directly; self.drone_state does not include position/quaternion.
            drone_pos = self.drone.pos  # (N, 1, 3), env frame
            drone_rot = self.drone.rot  # (N, 1, 4), quaternion

            gate_env_pos = self._gate_env_pos_cache  # static, cached at init
            gate_env_rot = self._gate_env_rot_cache

            # Print debug positions in the same frame (env frame) for easier sanity checks.
            # print(f"Drone position (env frame): {torch.round(drone_pos[0, 0] * 100) / 100}")
            if self.debug_gate_origins:
                gate_world_pos, gate_world_rot = self._get_gate_world_poses()
                self.debug_draw.clear()
                self._draw_gate_origins(gate_world_pos, gate_world_rot)
                self._draw_global_trajectory()

            batch_indices = torch.arange(self.num_envs, device=self.device)
            current_gate_pos = gate_env_pos[batch_indices, self.gate_indices]  # (N, 3)
        except Exception as e:
            print("=" * 80)
            print(
                f"ERROR: Failed in _compute_reward_and_done (num_envs={self.num_envs}, num_gates={self.num_gates})"
            )
            print("=" * 80)
            traceback.print_exc()
            raise

        current_gate_rot = gate_env_rot[batch_indices, self.gate_indices]  # (N, 4)
        current_gate_center = self._get_gate_center(
            current_gate_pos, current_gate_rot
        )  # (N, 3)

        drone_pos_flat = drone_pos.squeeze(1)  # (N, 3)
        distance_to_gate = torch.norm(
            drone_pos_flat - current_gate_center, dim=-1
        )  # (N,)

        # --- gate crossing detection ---
        # You either _deteect_gate_crossings or _detect_gate_crossings_via_segments
        # This function call updates the gate indexes
        (
            gate_passed_this_step,
            gate_index_changed,
            new_gate_center,
            _gate_bypassed_this_step,
        ) = self._detect_gate_crossings(
            drone_pos_flat,
            current_gate_center,
            current_gate_rot,
            gate_env_pos,
            gate_env_rot,
            batch_indices,
        )

        # -----------------------------------------------------------------------
        # STUDENT TODO (2/3): Implement your reward function.
        #
        # Variables available in this scope:
        #   drone_pos_flat        (N, 3)    drone position in env frame
        #   drone_rot             (N, 1, 4) drone orientation quaternion (w, x, y, z)
        #   distance_to_gate      (N,)      Euclidean distance to current gate centre
        #   current_gate_center   (N, 3)    3-D centre of the current target gate
        #   gate_passed_this_step (N,)      bool – True when drone just passed a gate
        #   gate_index_changed    (N,)      bool – True when target gate index advanced
        #   new_gate_center       (N, 3)    centre of the (possibly new) target gate
        #   crashed_collision     (N,)      bool – True when contact forces are detected
        #   self.drone.vel        (N, 1, 6) [lin_vel(3) | ang_vel(3)] in body frame
        #   self.gate_indices     (N,)      index of current target gate (0…num_gates-1)
        #   self.track_completed  (N,)      bool – True when the full lap is done
        #
        # Useful helpers (already imported at top of file):
        #   quat_axis(q, axis)          extract body axis vector (0=x, 1=y, 2=z)
        #   quat_rotate(q, v)           rotate vector v by quaternion q
        #   quat_rotate_inverse(q, v)   rotate vector v by inverse of quaternion q
        #
        # Your reward scalars are loaded in __init__ (STUDENT TODO 1/3) and
        # configured in cfg/task/DroneRace.yaml.
        # -----------------------------------------------------------------------
        # ----- ADD YOUR REWARD CODE BELOW (replace the placeholder) -----
        #
        collision_forces = self.drone.base_link.get_net_contact_forces()  # (N, 1, 3)
        crashed = collision_forces.squeeze(1).norm(dim=-1) > 1.0  # (N,)

        #
        # ══════════════════════════════════════════════════════════════════
        # MPCC-STYLE CONTOURING REWARD
        # ══════════════════════════════════════════════════════════════════
        # Converts MPCC's cost  J = q_c·e_c² + q_l·e_l² − μ_v·Δθ  into
        # an RL reward by flipping the sign:
        #   R = w_progress·Δlag − w_contouring·e_c² − w_lag·e_l²
        #       + gate bonus + speed bonus + centering bonus
        #       − time penalty − smoothness penalty
        #
        # The "racing line" between consecutive gates is the straight
        # segment from prev_gate_center → current_gate_center.  The drone's
        # position is decomposed into:
        #   • lag   = projection onto the path tangent  (forward progress)
        #   • e_c   = perpendicular distance from path  (contouring error)
        #
        # References:
        #   MPCC  — Romero et al., TRO 2022 (lag/contouring decomposition)
        #   MPCC++ — Krinner et al., 2024 (safety corridors, TuRBO tuning)
        #   Swift  — Kaufmann et al., Nature 2023 (r_prog + r_cmd + r_crash)
        #   MonoRace — MavLab 2026 (speed incentive, tiny-net direct control)
        #   MPPI   — 2024 (reference-free gate progress, Gaussian near-gate weighting)
        #   AC-MPC — Romero et al., TRO 2025 (state-dependent cost adaptation)
        #   TOGT   — 2023 (gate as spatial volume, centering matters)
        # ══════════════════════════════════════════════════════════════════

        # ── 0. Gate-to-gate line-segment MPCC decomposition ─────────────────────
        # Racing line: straight segment from previous gate center → current gate center.
        # When at gate 0 (no previous gate), fall back to the gate's forward normal.
        # All computations stay in env frame (same frame as drone_pos_flat).
        prev_gate_idx = torch.clamp(self.gate_indices - 1, min=0)  # (N,)
        seg_start = self.gate_env_centers[batch_indices, prev_gate_idx]  # (N, 3)
        seg_end = self.gate_env_centers[batch_indices, self.gate_indices]  # (N, 3)
        seg_dir = seg_end - seg_start  # (N, 3)
        seg_len = torch.norm(seg_dir, dim=-1, keepdim=True).clamp(min=1e-6)
        path_tangent = seg_dir / seg_len  # (N, 3) unit tangent along segment

        # Degenerate case (gate_idx == 0): use gate's forward axis as tangent.
        is_first_gate = self.gate_indices == 0  # (N,)
        gate_forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        gate_forward = quat_rotate(
            current_gate_rot,
            gate_forward_local.unsqueeze(0).expand(self.num_envs, -1),
        )  # (N, 3)
        path_tangent = torch.where(
            is_first_gate.unsqueeze(-1), gate_forward, path_tangent
        )

        # Project drone position onto the racing line segment.
        drone_to_start = drone_pos_flat - seg_start  # (N, 3)
        lag = torch.sum(drone_to_start * path_tangent, dim=-1)  # (N,)
        contouring_vec = drone_to_start - lag.unsqueeze(-1) * path_tangent  # (N, 3)
        contouring_err = torch.norm(contouring_vec, dim=-1)  # (N,)

        # 2a. Progress reward — Δlag along the racing line (MPCC's μ_v·Δθ).
        # Zero out on gate-transition steps: the lag frame shifts, so delta is
        # meaningless; gate_bonus covers those steps.
        delta_lag = (lag - self.prev_lag).clamp(min=-1.0)
        progress_reward = delta_lag * self.w_progress * (~gate_index_changed).float()

        # 2a'. Approach reward — Euclidean distance shrinkage (weaker backup signal).
        approach_reward = (
            self.prev_distance_to_gate - distance_to_gate
        ) * self.w_approach  # (N,)

        # 2b. Contouring penalty  (MPCC's q_c·e_c²)
        contouring_penalty = -contouring_err.pow(2) * self.w_contouring  # (N,)

        # lag_penalty pruned: redundant with progress_reward + contouring_penalty

        # 2d. Gate passage bonus  (Swift / MonoRace)
        gate_bonus = gate_passed_this_step.float() * self.w_gate  # (N,)

        # 2f. Speed bonus  (MonoRace / MPPI: encourage fast flight)
        drone_vel_body = self.drone.vel[..., :3].squeeze(1)  # (N, 3)
        drone_rot_flat = drone_rot.squeeze(1)  # (N, 4)
        drone_vel_env = quat_rotate(drone_rot_flat, drone_vel_body)  # (N, 3)
        speed = torch.norm(drone_vel_env, dim=-1)  # (N,)
        speed_bonus = speed * self.w_speed  # (N,)

        # 2f2. Velocity alignment bonus — reward flying in the direction of the racing line.
        # Cosine similarity between velocity direction and path tangent: +1 when perfectly
        # aligned, 0 when perpendicular, -1 when flying backward. Complements speed_bonus
        # (which rewards magnitude) and progress_reward (which rewards displacement).
        # Other groups reported this as a large win: the drone learns direction quality
        # independently of how fast it's going. Default 0.0 (off) — set w_alignment in YAML.
        velocity_dir = drone_vel_env / (speed.unsqueeze(-1) + 1e-6)  # (N, 3) unit velocity
        alignment = (velocity_dir * path_tangent).sum(dim=-1)  # (N,) cosine in [-1, 1]
        alignment_reward = alignment * self.w_alignment  # (N,)

        # 2g. Time penalty  (Swift: small per-step cost)
        time_penalty = -self.w_time * self.w_stall_step # scalar

        # 2h. Action smoothness penalty  (Swift r_cmd)
        current_action = self.drone.get_joint_velocities()
        action_diff = torch.norm(
            (current_action - self.last_action).squeeze(1), dim=-1
        )  # (N,)
        smooth_penalty = -action_diff * self.w_smooth  # (N,)

        # 2i. Angular rate penalty — discourage spinning out of control.
        # drone.vel[..., 3:6] is angular velocity in body frame (rad/s).
        ang_vel_body = self.drone.vel[..., 3:6].squeeze(1)  # (N, 3)
        roll_rate = ang_vel_body[..., 0]  # (N,)
        pitch_rate = ang_vel_body[..., 1]  # (N,)
        yaw_rate = ang_vel_body[..., 2]  # (N,)
        angular_rate_penalty = -self.w_ang_rate * (
            roll_rate.pow(2) + pitch_rate.pow(2)
        ) - self.w_yaw_rate * yaw_rate.pow(2)  # (N,)

        # 2j. Lap completion bonus
        completion_bonus = self.track_completed.float() * self.w_completion  # (N,)

        # 2k. Gate bypass penalty — fires once per gate when drone crosses the gate
        # plane outside the opening (bypasses rather than flying through).
        # Must be larger than w_gate so bypassing is never a profitable strategy.
        # bypass_penalty = -gate_bypassed_this_step.float() * self.w_bypass  # (N,)

        # ── 2k–n. Basic flight bootstrap rewards ─────────────────────────
        # Without these, an untrained policy falls to the ground in ~50 steps,
        # all episodes get identical returns, return variance → 0, and
        # explained_var = 1 − MSE/Var(returns) blows up to −2,000,000.
        # These dense per-step terms give PPO a gradient signal from step 1.

        # alive_bonus pruned: covered by crash_penalty + stall termination

        # 2l. Upright bonus: reward keeping the drone's z-axis pointing up.
        drone_up = quat_axis(drone_rot.squeeze(1), axis=2)  # (N, 3) body z-axis in world frame
        upright_bonus = drone_up[:, 2] * self.w_upright  # (N,) — negative when inverted, positive when upright

        # 2n. Crash penalty — flat penalty on any collision.
        crashed = (collision_forces.squeeze(1).norm(dim=-1) > 1.0)
        crash_penalty = -crashed.float() * self.w_crash  # (N,)

        # 2o. Progress-based stall — counter increments every step, resets on gate passage.
        # Fires a large penalty and terminates when stall_patience steps pass without a gate.
        self.stall_counter = torch.where(
            gate_passed_this_step,
            torch.zeros_like(self.stall_counter),
            self.stall_counter + 1,
        )
        stall_triggered = self.stall_counter >= self.stall_patience  # (N,)
        stall_penalty = -stall_triggered.float() * self.w_stall  # (N,)

        # 2p. Altitude bonus — exp reward for flying near the current gate's height.
        # This is the critical bootstrap signal: without it, reward is identical
        # whether the drone sits on the ground or hovers at gate height.
        gate_z = current_gate_center[:, 2]  # (N,) target altitude
        altitude_err = drone_pos_flat[:, 2] - gate_z  # (N,)
        altitude_bonus = torch.exp(-altitude_err.pow(2) * 0.5) * self.w_altitude  # (N,)

        # ── 3. Total reward ──────────────────────────────────────────────
        reward = (
            progress_reward  # primary racing driver: Δlag along racing line
            + approach_reward  # backup Euclidean approach signal
            + contouring_penalty  # MPCC q_c·e_c² — stay on racing line
            + gate_bonus  # one-time bonus for passing a gate
            + speed_bonus  # reward fast flight — makes flying beat hovering
            + alignment_reward  # velocity direction aligned with racing line tangent
            + smooth_penalty  # action smoothness
            + crash_penalty  # penalise crashes
            + stall_penalty  # penalise hovering / flying too low too long
            + completion_bonus  # full-lap bonus
            + upright_bonus  # keep drone z-axis pointing up (bootstrap)
            + altitude_bonus  # fly at gate height (bootstrap)
            + time_penalty  # per-step cost — must exceed bootstrap to discourage hovering
        )  # (N,)

        # ── 4. Update state for next step ────────────────────────────────
        self.prev_distance_to_gate = torch.norm(
            drone_pos_flat - new_gate_center, dim=-1
        ).detach()
        # Zero prev_lag on gate transitions: avoids cross-frame subtraction at step t+1.
        self.prev_lag = torch.where(gate_index_changed, torch.zeros_like(lag), lag).detach()
        self.last_action = current_action.detach()

        # ----- END STUDENT CODE -----

        # -----------------------------------------------------------------------
        # Crash / termination condition — computed above (before the reward)
        # so that crash_penalty is included on the terminal transition.
        # Components: ground_crash | bounds_crash | flipped_crash | contact_crash
        # Variables ground_crash, bounds_crash, flipped_crash,
        # contact_crash, and crashed are all already defined above.
        # -----------------------------------------------------------------------
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        completed_task = self.track_completed
        done = (
            truncated
            | completed_task.unsqueeze(-1)
            | crashed.unsqueeze(-1)
            | stall_triggered.unsqueeze(-1)
        )

        # --- stats ---
        self.stats["truncated"].add_(truncated.float())
        self.stats["collision"].add_(crashed.float().unsqueeze(-1))
        self.stats["success"].bitwise_or_(completed_task.unsqueeze(-1))
        self.stats["return"].add_(reward.unsqueeze(-1))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        gates_since_spawn = (self.gate_indices - self.start_gate_indices).clamp(min=0)
        self.stats["gates_passed"][:] = (
            (
                gates_since_spawn
                + self.track_completed.long()
                * (self.num_gates - 1 - self.start_gate_indices)
            )
            .float()
            .unsqueeze(1)
        )
        # (optional) add extra stat tracking lines here if you add new metrics
        self.stats["drone_uprightness"][:] = (
            quat_axis(drone_rot.squeeze(1), axis=2)[:, 2].unsqueeze(1).detach()
        )

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1).unsqueeze(-1),
                },
                "done": done,
                "terminated": crashed.unsqueeze(-1),
                "truncated": truncated,
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )
